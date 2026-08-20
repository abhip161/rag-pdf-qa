import logging
import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Optional

from sentence_transformers import SentenceTransformer
from langsmith import traceable

from src.embedding import EmbeddingPipeline

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata: List[Any] = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info("Loaded embedding model: %s", embedding_model)

    def build_from_documents(self, documents: List[Any]) -> None:
        logger.info("Building vector store from %d raw document(s)…", len(documents))
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()
        logger.info("Vector store built and saved to %s", self.persist_dir)

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: Optional[List[Any]] = None,
    ) -> None:
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        logger.info("Added %d vector(s) to FAISS index.", embeddings.shape[0])

    def save(self) -> None:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if self.index is None:
            raise ValueError("Cannot save: FAISS index has not been built yet.")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Saved FAISS index and metadata to %s", self.persist_dir)

    def load(self) -> None:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info("Loaded FAISS index and metadata from %s", self.persist_dir)

    @traceable(name="faiss_vector_search", run_type="tool")
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """Raw FAISS index search. Traced as a tool span with distance metadata."""
        if self.index is None:
            raise ValueError("Cannot search: FAISS index has not been built or loaded yet.")
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    @traceable(
        name="retrieve_documents",
        run_type="retriever",
        metadata={"vector_store_type": "faiss"},
    )
    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        """
        Encode the query and run a FAISS nearest-neighbour search.

        Traced as a retriever span. Metadata includes:
          - query_text       : the raw user query
          - top_k            : number of results requested
          - vector_store_type: 'faiss' (set via decorator metadata)
        """
        logger.debug("Querying vector store: '%s' (top_k=%d)", query_text, top_k)
        query_emb = np.array(self.model.encode([query_text])).astype("float32")
        results = self.search(query_emb, top_k=top_k)
        logger.debug("Retrieved %d result(s).", len(results))
        return results
