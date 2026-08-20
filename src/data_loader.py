import logging
from pathlib import Path
from typing import List, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="load_documents", run_type="tool")
def load_all_documents(data_dir: str) -> List[Any]:
    """
    Recursively load all PDF and TXT files from *data_dir* and return them
    as a flat list of LangChain Document objects.

    LangSmith captures:
      - input  : data_dir path
      - output : total number of documents loaded
      - errors : per-file failures are logged and do not raise
    """
    data_path = Path(data_dir).resolve()
    logger.debug("Data path: %s", data_path)
    documents: List[Any] = []

    # ── PDF files ─────────────────────────────────────────────────────────────
    pdf_files = list(data_path.glob("**/*.pdf"))
    logger.info("Found %d PDF file(s)", len(pdf_files))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            logger.debug("Loaded %d page(s) from %s", len(loaded), pdf_file)
            documents.extend(loaded)
        except Exception as exc:
            logger.error("Failed to load PDF %s: %s", pdf_file, exc)

    # ── TXT files ─────────────────────────────────────────────────────────────
    txt_files = list(data_path.glob("**/*.txt"))
    logger.info("Found %d TXT file(s)", len(txt_files))
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            logger.debug("Loaded %d doc(s) from %s", len(loaded), txt_file)
            documents.extend(loaded)
        except Exception as exc:
            logger.error("Failed to load TXT %s: %s", txt_file, exc)

    logger.info("Total documents loaded: %d", len(documents))
    return documents