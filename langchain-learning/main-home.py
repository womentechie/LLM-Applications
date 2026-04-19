# ============================================================
# Hub Home — unified landing page for LangChain Learning Hub
# ============================================================
# This is the root landing page for the whole project.
# It also hosts the MODULE SELECTOR — a set of pills that let
# the user filter the sidebar to show only one module's pages.
#
# WHY THE SELECTOR LIVES HERE (not in the sidebar):
#   st.navigation() always occupies the top of the sidebar.
#   Any widget added via `with st.sidebar:` renders BELOW the
#   nav list regardless of code order — so a sidebar dropdown
#   ends up at the bottom of the screen (exactly the bug we saw).
#   Putting the selector in the main content area avoids this
#   entirely. The selection is persisted in st.query_params so
#   it survives clicking between pages without resetting.
# ============================================================

import streamlit as st

st.title("🧠 LangChain Learning Hub")
st.markdown(
    "A self-contained collection of LangChain + Streamlit demos — "
    "embeddings, RAG, vision, and core chain patterns."
)

# ── Module selector ────────────────────────────────────────────
# Clicking a button sets ?module=<key> in the URL query params.
# main-app.py reads this param to filter the sidebar nav list.
# The page then reloads with the updated sidebar automatically.
st.markdown("#### Filter sidebar by module")

MODULE_OPTIONS = {
    "🏠 All modules":        "all",
    "🔢 Embeddings & RAG":   "embeddings",
    "🖼️ Image Processing":   "image",
    "🎙️ Audio Processing":   "audio",      # ← add this
    "⚡ Streamlit Patterns": "streamlit",
}

current = st.query_params.get("module", "all")

cols = st.columns(len(MODULE_OPTIONS))
for col, (label, key) in zip(cols, MODULE_OPTIONS.items()):
    with col:
        is_active = current == key
        # Active module button gets a filled appearance via type="primary"
        if st.button(
            label,
            key=f"mod_{key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.query_params["module"] = key
            st.rerun()

st.divider()

# ── Active credentials banner ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    embed_provider = st.session_state.get("embed_provider", "—")
    embed_model    = st.session_state.get("embed_model",    "—")
    embed_key      = st.session_state.get("embed_api_key",  None)
    st.success(f"🔢 Embedding: **{embed_provider}** · `{embed_model}`")
    if embed_key:
        st.caption(f"🔑 `{embed_key[:8]}...`")
    elif embed_provider == "HuggingFace (Local)":
        st.caption("🖥️ Running locally — no key needed")

with col2:
    chat_provider = st.session_state.get("provider", "—")
    chat_model    = st.session_state.get("model",    "—")
    chat_key      = st.session_state.get("api_key",  None)
    st.success(f"💬 Chat LLM: **{chat_provider}** · `{chat_model}`")
    if chat_key:
        st.caption(f"🔑 `{chat_key[:8]}...`")

st.divider()

# ── Module index ───────────────────────────────────────────────
st.subheader("📚 What's inside")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
**🔢 Embeddings**
Convert text to vectors and measure semantic similarity.

**🗄️ Vector DBs**
Store and search vectors with ChromaDB.

**🧠 RAG**
Retrieval-Augmented Generation — ground LLM answers in your documents.
""")

with col_b:
    st.markdown("""
**🖼️ Image Processing**
Multimodal vision — image Q&A, KYC verification, product comparison.

**🖼️ Audio Processing**
Multimodal audio — audio Q&A.

**⚡ Streamlit Patterns**
Core LangChain patterns — prompt templates, LCEL chains, chat memory, TTS.
""")

st.divider()

# ── How to add new pages / modules ────────────────────────────
st.subheader("➕ Adding new demos")
st.markdown("""
**New page in an existing module:**
1. Drop the `.py` file into `<module>/pages/`
2. Add one `st.Page(...)` line with a unique `url_path` in `main-app.py`

**New module entirely:**
1. Create `new-module/utils.py` (copy from an existing module)
2. Add `"new-module"` to `MODULE_DIRS` in `main-app.py`
3. Add an entry to `MODULE_OPTIONS` in both `main-app.py` and `main-home.py`
4. Add a new key → sections dict to `ALL_PAGES` in `main-app.py`
""")

st.divider()
st.caption("Built with LangChain · Streamlit · Python")