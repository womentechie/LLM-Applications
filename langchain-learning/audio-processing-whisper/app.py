# ============================================================
# CONCEPT: Streamlit Multi-Page App with st.navigation()
# ============================================================
# This app resolves ONE set of credentials at startup:
#
#   CHAT MODEL KEY  (utils.py)
#      Used by: all pages in this module
#      Stored in session_state as: "api_key", "provider", "model"
#
# WHY NO EMBEDDING KEY?
#   Image processing sends images directly to a vision-capable LLM
#   as base64-encoded data URIs — no vector store or similarity search
#   is involved. Only a chat model is needed. There is no embedding
#   step, so embeddings_utils.py is not imported here.
#
# IMPORTANT — AVOID THIS MISTAKE:
#   Never use [...] (ellipsis) as a placeholder for future pages.
#   st.navigation() raises:
#     StreamlitAPIException: Invalid page type: <class 'ellipsis'>
#   Comment out entire sections until their pages are built instead.
#
# FILE STRUCTURE:
#   app.py                              ← this file (entry point)
#   utils.py                            ← chat model key + factory
#   pages/
#     home.py                           ← landing page
#     01_embed_similarity.py            ← text similarity (uses chat only)
#     20_describe_image_assistant.py    ← image Q&A — needs chat key only
#
# RUN WITH:  streamlit run app.py
# ============================================================

import streamlit as st
from utils import get_or_set_api_key

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# ── App-wide page config ───────────────────────────────────────
# Called once here — individual pages must NOT call st.set_page_config()
#st.set_page_config(
#    page_title="Image Processing Demo Hub",
#    page_icon="🖼️",
#    layout="centered",
#)

# ── Step 1: Resolve CHAT MODEL key ────────────────────────────
# get_or_set_api_key() checks session_state first (fast path),
# then shows the chat provider/model/key UI if not yet entered.
# Stored under: "api_key", "provider", "model"
#
# !! SECURITY: Keys stored ONLY in session_state — never os.environ !!
# os.environ is process-level, shared across all users on the server.
# session_state is per-browser-tab and cleared when the tab closes.
#
# NOTE: Select a vision-capable model in the auth UI:
#   OpenAI    → gpt-4o ✅, gpt-4-turbo ✅
#   Anthropic → claude-3-5-sonnet-20241022 ✅, claude-3-opus ✅
#   Google    → gemini-2.0-flash ✅, gemini-1.5-pro ✅
chat_key, chat_provider, chat_model = get_or_set_api_key()

# ── Page Definitions ───────────────────────────────────────────
# Only include pages whose .py files actually exist.
# Comment out sections until the pages are built — never use [...].

pages = {

    # ── Home ──────────────────────────────────────────────────
    "🏠 Home": [
        st.Page("pages/home.py", title="Getting Started", icon="🏠", default=True),
    ],

    # ── Image Processing demos — needs chat key only ──────────
    # All pages here send images directly to a vision-capable LLM
    # as base64 data URIs. No embedding model or vector store is used.
    "🎙️ Audio Processing": [
        st.Page("pages/30_audio_assistant.py", title="Audio Q&A", icon="🎙️"),
    ],


}

# ── Run Navigation ─────────────────────────────────────────────
pg = st.navigation(pages)
pg.run()