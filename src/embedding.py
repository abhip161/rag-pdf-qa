import logging
import numpy as np
from typing import List, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from sentence_transformers import SentenceTransformer

from src.data_loader import load_all_documents  # noqa: F401 (re-exported for callers)

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)

    @traceable(name="chunk_documents", run_type="tool")
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

        LangSmith captures:
          - input  : number of raw documents
          - output : number of chunks produced
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split %d document(s) into %d chunk(s).", len(documents), len(chunks))
        return chunks

    @traceable(
        name="generate_embeddings",
        run_type="tool",
        metadata={"embedding_framework": "sentence-transformers"},
    )
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Generate dense vector embeddings for all chunks.

        LangSmith captures:
          - input  : number of chunks to embed
          - output : embedding array shape
          - embedding_framework : 'sentence-transformers'
        """
        texts = [chunk.page_content for chunk in chunks]
        logger.info("Generating embeddings for %d chunk(s)…", len(texts))
        embeddings = self.model.encode(texts, show_progress_bar=False)
        logger.info("Embeddings shape: %s", embeddings.shape)
        return np.asarray(embeddings)