import logging
import os
from dotenv import load_dotenv
from langsmith import traceable

from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger(__name__)


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b",
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vector store depending on whether a persisted index exists.
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(api_key=groq_api_key, model=llm_model)
        logger.info("Groq LLM initialized: %s", llm_model)

    @traceable(
        name="rag_pipeline",
        run_type="chain",
        tags=["rag", "pdf-qa"],
    )
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Full RAG pipeline: retrieve → build context → generate answer.

        This is the **root span** in LangSmith. All nested spans
        (retrieve_documents, faiss_vector_search, ChatGroq) attach here.

        LangSmith captures automatically:
          - inputs  : query, top_k
          - outputs : final answer string
          - latency : wall-clock time for the entire pipeline
          - tags    : ['rag', 'pdf-qa']
        """
        # ── Attach static run-level metadata ─────────────────────────────────
        try:
            from langsmith import get_current_run_tree
            run = get_current_run_tree()
            if run is not None:
                run.metadata.update(
                    {
                        "embedding_model": self.embedding_model,
                        "llm_model": self.llm_model,
                        "vector_store_type": "faiss",
                        "top_k": top_k,
                    }
                )
        except Exception:
            # Never break the pipeline if metadata attachment fails.
            pass

        # ── Retrieval ─────────────────────────────────────────────────────────
        results = self.vectorstore.query(query, top_k=top_k)

        # ── Attach dynamic metadata (num results) ─────────────────────────────
        try:
            run = get_current_run_tree()
            if run is not None:
                run.metadata["num_retrieved_docs"] = len(results)
        except Exception:
            pass

        # ── Context assembly ──────────────────────────────────────────────────
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."

        # ── Prompt construction ───────────────────────────────────────────────
        prompt = (
            f"Summarize the following context for the query: '{query}'\n\n"
            f"Context:\n{context}\n\nSummary:"
        )

        # ── LLM generation (ChatGroq auto-traced by LangChain integration) ────
        response = self.llm.invoke(prompt)
        return str(response.content)