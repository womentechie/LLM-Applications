# ============================================================
# CONCEPT: Streamlit Multi-Page App with st.navigation()
# ============================================================
# This app resolves TWO sets of credentials at startup:
#
#   1. EMBEDDING KEY  (embeddings_utils.py)
#      Used by: embed_similarity.py, 08_chroma_job_search.py,
#               rag_demo.py (to vectorise documents + query)
#      Stored in session_state as: "embed_api_key", "embed_provider",
#                                  "embed_model"
#
#   2. CHAT MODEL KEY  (utils.py)
#      Used by: rag_demo.py (to generate answers from retrieved context)
#      Stored in session_state as: "api_key", "provider", "model"
#
# WHY BOTH ARE NEEDED FOR RAG:
#   RAG has two distinct steps that use different model types:
#     Step 1 — Retrieve: embed the user's query → search vector store
#                        → needs an EMBEDDING model
#     Step 2 — Generate: pass retrieved chunks to LLM → get an answer
#                        → needs a CHAT model
#   Embedding pages (similarity, job search) only need the embedding key.
#   RAG pages need both. Resolving both here means every page just reads
#   from session_state without triggering any auth UI.
#
# IMPORTANT — AVOID THIS MISTAKE:
#   Never use [...] (ellipsis) as a placeholder for future pages.
#   st.navigation() raises:
#     StreamlitAPIException: Invalid page type: <class 'ellipsis'>
#   Comment out entire sections until their pages are built instead.
#
# FILE STRUCTURE:
#   app.py                          ← this file (entry point)
#   embeddings_utils.py             ← embedding key + factory
#   utils.py                        ← chat model key + factory
#   pipeline_utils.py               ← document loading + chunking
#   pages/
#     home.py                       ← landing page
#     embed_similarity.py           ← text similarity
#     08_chroma_job_search.py       ← semantic job search
#     rag_demo.py                   ← RAG — needs both keys
#
# RUN WITH:  streamlit run app.py
# ============================================================

import streamlit as st
from embeddings_utils import get_or_set_embedding_key
from utils import get_or_set_api_key

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# ── App-wide page config ───────────────────────────────────────
# Called once here — individual pages must NOT call st.set_page_config()
st.set_page_config(
    page_title="Embeddings Demo Hub",
    page_icon="🔢",
    layout="centered",
)

# ── Step 1: Resolve EMBEDDING key ─────────────────────────────
# get_or_set_embedding_key() checks session_state first (fast path),
# then shows the embedding provider/model/key UI if not yet entered.
# Stored under: "embed_api_key", "embed_provider", "embed_model"
#
# !! SECURITY: Keys stored ONLY in session_state — never os.environ !!
# os.environ is process-level, shared across all users on the server.
# session_state is per-browser-tab and cleared when the tab closes.
embed_key, embed_provider, embed_model = get_or_set_embedding_key()

# ── Step 2: Resolve CHAT MODEL key ────────────────────────────
# get_or_set_api_key() checks session_state first (fast path),
# then shows the chat provider/model/key UI if not yet entered.
# Stored under: "api_key", "provider", "model"
#
# This is needed by rag_demo.py to generate answers.
# Embedding-only pages (similarity, job search) ignore this key —
# it's already in session_state and costs nothing to have resolved.
chat_key, chat_provider, chat_model = get_or_set_api_key()

# ── Page Definitions ───────────────────────────────────────────
# Only include pages whose .py files actually exist.
# Comment out sections until the pages are built — never use [...].

pages = {

    # ── Home ──────────────────────────────────────────────────
    "🏠 Home": [
        st.Page("pages/home.py", title="Getting Started", icon="🏠", default=True),
    ],

    # ── Embedding demos ───────────────────────────────────────
    "🔢 Embeddings": [
        st.Page("pages/01_embed_similarity.py", title="Text Similarity", icon="📊"),
    ],

    # ── Vector DB demos ───────────────────────────────────────
    "🗄️ Vector DBs": [
        st.Page("pages/08_chroma_job_search.py", title="Job Search (Chroma)", icon="💼"),
    ],

    # ── RAG demos — needs both embedding + chat keys ──────────
    "🧠 RAG": [
        st.Page("pages/10_rag_demo.py", title="RAG Demo", icon="🧠"),
    ],

    # ── Chunking — uncomment as you build each page ───────────
    # "✂️ Chunking": [
    #     st.Page("pages/04_chunk_fixed.py",     title="Fixed size",     icon="✂️"),
    #     st.Page("pages/05_chunk_recursive.py", title="Recursive",      icon="🔄"),
    #     st.Page("pages/06_chunk_semantic.py",  title="Semantic",       icon="🧠"),
    #     st.Page("pages/07_loaders.py",         title="Loaders",        icon="📂"),
    # ],

}

# ── Run Navigation ─────────────────────────────────────────────
pg = st.navigation(pages)
pg.run()