"""
RAG PDF Q&A — Streamlit Dashboard
Run: streamlit run dashboard.py
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "RAG PDF Q&A — FAISS · Groq · LangSmith"},
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #12122a 60%, #0f1a2e 100%);
    min-height: 100vh;
}

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(13, 13, 26, 0.85) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Hero title ── */
.hero-title {
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.4rem;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 4px;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 0;
}

/* ── Status badges ── */
.badge-ok {
    display: inline-block;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.badge-err {
    display: inline-block;
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #f87171;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.badge-warn {
    display: inline-block;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fbbf24;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 500;
}

/* ── Section labels ── */
.sidebar-section {
    color: #64748b !important;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    margin-bottom: 10px;
    padding: 4px 8px;
}

/* ── Source card ── */
.source-card {
    background: rgba(99, 102, 241, 0.07);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #cbd5e1;
    line-height: 1.65;
}
.source-num {
    color: #818cf8;
    font-weight: 600;
    font-size: 0.78rem;
    margin-bottom: 4px;
    display: block;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 10px 14px;
}
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.45) !important;
}
.stButton > button:disabled {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #475569 !important;
}

/* ── Dividers ── */
hr { border-color: rgba(255, 255, 255, 0.07) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px dashed rgba(99, 102, 241, 0.3);
    border-radius: 10px;
    padding: 4px;
}

/* ── Selectbox / Slider ── */
[data-baseweb="select"] { background: rgba(255, 255, 255, 0.06) !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #475569;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 12px; }
.empty-text { font-size: 1.05rem; color: #64748b; }
.empty-hint { font-size: 0.85rem; color: #475569; margin-top: 6px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _env_badge(var: str) -> str:
    """Return an HTML badge showing whether an env var is configured."""
    val = os.getenv(var, "")
    ok = bool(val and not val.startswith("your_"))
    cls = "badge-ok" if ok else "badge-err"
    icon = "✓" if ok else "✗"
    label = "configured" if ok else "missing"
    return f'<span class="{cls}">{icon} {label}</span>'


def _index_exists(persist_dir: str) -> bool:
    return (
        Path(persist_dir, "faiss.index").exists()
        and Path(persist_dir, "metadata.pkl").exists()
    )


# ── Cached RAGSearch loader ────────────────────────────────────────────────────
    
@st.cache_resource(show_spinner=False)
def _load_rag(persist_dir: str, embedding_model: str, llm_model: str) -> Any:
    """Load RAGSearch once per unique (persist_dir, embedding_model, llm_model) combo."""
    from src.search import RAGSearch
    return RAGSearch(
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )


# ── Constants ──────────────────────────────────────────────────────────────────

PERSIST_DIR = "faiss_store"
LLM_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
EMBEDDING_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Branding
    st.markdown("### 📄 RAG PDF Q&A")
    st.markdown('<p class="hero-subtitle">Powered by FAISS · Groq · LangSmith</p>', unsafe_allow_html=True)
    st.divider()

    # API status
    st.markdown('<p class="sidebar-section">🔑 API Status</p>', unsafe_allow_html=True)
    st.markdown(
        f"GROQ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        f"{_env_badge('GROQ_API_KEY')}<br>"
        f"LangSmith &nbsp;{_env_badge('LANGSMITH_API_KEY')}",
        unsafe_allow_html=True,
    )
    st.divider()

    # Model config
    st.markdown('<p class="sidebar-section">🤖 Model Configuration</p>', unsafe_allow_html=True)
    llm_model: str = st.selectbox("LLM Model (Groq)", LLM_MODELS, index=0)
    embedding_model: str = st.selectbox("Embedding Model", EMBEDDING_MODELS, index=0)
    top_k: int = st.slider("Top-K chunks", min_value=1, max_value=10, value=3)
    st.divider()

    # Vector store status
    st.markdown('<p class="sidebar-section">🗄️ Vector Store</p>', unsafe_allow_html=True)
    idx_ready = _index_exists(PERSIST_DIR)
    if idx_ready:
        st.markdown('<span class="badge-ok">✓ Index ready</span>', unsafe_allow_html=True)
        # Show rough size
        idx_size = Path(PERSIST_DIR, "faiss.index").stat().st_size / 1024
        st.caption(f"faiss.index: {idx_size:.1f} KB")
    else:
        st.markdown('<span class="badge-err">✗ No index found</span>', unsafe_allow_html=True)
        st.caption("Upload documents and rebuild to get started.")

    col_rb, col_cl = st.columns(2)
    with col_rb:
        if st.button("🔄 Rebuild", use_container_width=True):
            with st.spinner("Rebuilding…"):
                st.cache_resource.clear()
                if Path(PERSIST_DIR).exists():
                    shutil.rmtree(PERSIST_DIR)
                try:
                    _load_rag(PERSIST_DIR, embedding_model, llm_model)
                    st.success("Done!")
                except Exception as exc:
                    st.error(f"Error: {exc}")
            st.rerun()
    with col_cl:
        if st.button("🗑️ Drop", use_container_width=True, disabled=not idx_ready):
            st.cache_resource.clear()
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
            st.warning("Index dropped.")
            st.rerun()
    st.divider()

    # Document upload
    st.markdown('<p class="sidebar-section">📁 Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF / TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        saved_names: list[str] = []
        for uf in uploaded_files:
            ext = Path(uf.name).suffix.lower()
            dest_dir = Path("data/pdf") if ext == ".pdf" else Path("data/text_files")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / uf.name
            dest_path.write_bytes(uf.read())
            saved_names.append(uf.name)
        st.success(f"Saved {len(saved_names)} file(s).")
        st.caption("Click **Rebuild** to include them in the index.")
    st.divider()

    # Chat controls
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # LangSmith link
    ls_project = os.getenv("LANGSMITH_PROJECT", "rag-pdf-qa")
    ls_configured = bool(
        os.getenv("LANGSMITH_API_KEY", "").strip()
        and not os.getenv("LANGSMITH_API_KEY", "").startswith("your_")
    )
    if ls_configured:
        st.markdown(
            f'<br><a href="https://smith.langchain.com/projects/{ls_project}" '
            f'target="_blank" style="color:#818cf8;font-size:0.8rem;">🔗 Open LangSmith traces</a>',
            unsafe_allow_html=True,
        )


# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown('<h1 class="hero-title">📄 RAG PDF Q&A</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Ask questions about your PDF and text documents. '
    "Answers are grounded in retrieved content via FAISS similarity search and Groq LLM.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div class="empty-text">No conversation yet</div>
            <div class="empty-hint">Type a question below to get started.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Sources expander
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} source chunks", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        preview = src[:450] + ("…" if len(src) > 450 else "")
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-num">Chunk {i}</span>{preview}'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Metrics
            if msg.get("metrics"):
                m = msg["metrics"]
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱ Latency", f"{m['latency']:.2f}s")
                c2.metric("📎 Chunks", m["num_chunks"])
                c3.metric("🤖 Model", m["llm_model"])


# ── Query input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    # Render user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Guard: GROQ key required
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key.startswith("your_"):
        st.error("⚠️ **GROQ_API_KEY** is not configured. Add it to your `.env` file.")
        st.stop()

    # Generate answer
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        try:
            rag = _load_rag(PERSIST_DIR, embedding_model, llm_model)

            t0 = time.perf_counter()

            # Retrieve source chunks (FAISS is in-memory — negligible cost)
            raw_results = rag.vectorstore.query(prompt, top_k=top_k)
            sources: list[str] = [
                r["metadata"].get("text", "")
                for r in raw_results
                if r.get("metadata")
            ]

            # Full traced RAG pipeline
            answer: str = rag.search_and_summarize(prompt, top_k=top_k)
            latency = time.perf_counter() - t0

            placeholder.markdown(answer)

            # Source chunks
            if sources:
                with st.expander(f"📚 {len(sources)} source chunks", expanded=False):
                    for i, src in enumerate(sources, 1):
                        preview = src[:450] + ("…" if len(src) > 450 else "")
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-num">Chunk {i}</span>{preview}'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Metrics row
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱ Latency", f"{latency:.2f}s")
            c2.metric("📎 Chunks", len(sources))
            c3.metric("🤖 Model", llm_model)

            # Persist to session
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metrics": {
                        "latency": latency,
                        "num_chunks": len(sources),
                        "llm_model": llm_model,
                    },
                }
            )

        except FileNotFoundError:
            placeholder.empty()
            st.error(
                "⚠️ No FAISS index found. Upload documents using the sidebar and click **Rebuild**."
            )
        except ValueError as exc:
            placeholder.empty()
            st.error(f"⚠️ Configuration error: {exc}")
            logger.error("RAG ValueError: %s", exc, exc_info=True)
        except Exception as exc:
            placeholder.empty()
            st.error(f"❌ Unexpected error: {exc}")
            logger.error("Dashboard query error: %s", exc, exc_info=True)
