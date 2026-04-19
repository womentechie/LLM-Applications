# ============================================================
# TOP-LEVEL ENTRY POINT — LangChain Learning Hub
# ============================================================
# Single entry point for the entire langchain-learning project.
# Unifies all three modules under one st.navigation() call.
#
# ── MODULE MAP ────────────────────────────────────────────────
#   embeddings/        text similarity, vector DBs, RAG demos
#   image-processing/  vision / multimodal demos
#   streamlit/         core LangChain + Streamlit pattern demos
#
# ── HOW TO ADD A NEW PAGE ─────────────────────────────────────
#   1. Drop your .py file into the relevant module's pages/ folder
#   2. Add an st.Page() entry in ALL_PAGES below under its module key
#   3. Give it a unique url_path  (convention: <prefix>-<description>)
#   That's it — credentials and sys.path are handled here.
#
# ── HOW TO ADD A NEW MODULE ───────────────────────────────────
#   1. Create  new-module/utils.py  (copy from an existing module)
#   2. Create  new-module/pages/    and add your page files
#   3. Add  "new-module"  to the MODULE_DIRS list below
#   4. Add the module label to MODULE_OPTIONS
#   5. Add a new key → sections dict to ALL_PAGES
#
# ── CREDENTIAL FLOW ───────────────────────────────────────────
#   All credentials are resolved HERE before any page loads.
#   Pages just read from st.session_state — no auth UI mid-nav.
#
#   EMBEDDING KEY  →  "embed_api_key", "embed_provider", "embed_model"
#   CHAT KEY       →  "api_key", "provider", "model"
#
# ── ROOT-LEVEL FILE REQUIRED ──────────────────────────────────
#   hub_utils.py — single utility at repo root that loads credentials
#   from embeddings/ using importlib (explicit file paths, no sys.path
#   dependency). Unique name avoids any conflict with module-level
#   utils.py files inside embeddings/, image-processing/, streamlit/.
#   Individual pages do NOT import from hub_utils — they use their
#   own module's utils.py resolved via sys.path injection below.
#
# ── URL PATH RULE ─────────────────────────────────────────────
#   Every st.Page() MUST have a unique url_path.
#   Without it Streamlit infers from filename — three home.py files
#   all claim "/home" and st.navigation() raises StreamlitAPIException.
#   Convention:  <module-prefix>-<short-description>
#     emb-  → embeddings    img-  → image-processing
#     rag-  → RAG           st-   → streamlit patterns
#     vdb-  → vector DBs    ag-   → agents (future)
#
# RUN WITH:
#   streamlit run main-app.py
# ============================================================

import sys
import os
import streamlit as st
# load_dotenv reads a local .env file for development.
# On Streamlit Cloud there is no .env file — credentials are set via
# the Streamlit Cloud Secrets manager (Settings → Secrets) and are
# already present as environment variables, so load_dotenv is a no-op.
# The try/except ensures the app still starts even if python-dotenv
# is not installed in the deployment environment.
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass  # Streamlit Cloud: secrets injected as env vars — dotenv not needed

# ── sys.path injection ────────────────────────────────────────
# Each module folder is prepended to sys.path so pages can do:
#   from utils import build_llm          → resolves their own utils.py
#   from embeddings_utils import ...     → resolves embeddings/embeddings_utils.py
# Add new module folder names here when you create them.
MODULE_DIRS = [
    "embeddings",
    "image-processing",
    "streamlit",
    # "agents",       ← add when ready
    # "fine-tuning",  ← add when ready
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for module_dir in MODULE_DIRS:
    path = os.path.join(BASE_DIR, module_dir)
    if path not in sys.path:
        sys.path.insert(0, path)

# ── App-wide config ───────────────────────────────────────────
# Called ONCE here. Individual pages must NOT call st.set_page_config().
st.set_page_config(
    page_title="LangChain Learning Hub",
    page_icon="🧠",
    layout="centered",
)

# ── Sidebar styling ───────────────────────────────────────────
# Targets stable data-testid attributes — survives minor Streamlit updates.
# Module selector pills are styled via .module-pill classes below.
st.markdown("""
<style>
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.12);
}
/* Title injected via CSS ::before — renders above the nav list
   because it is part of the sidebar container, not a widget.
   Widgets added via st.sidebar always render BELOW st.navigation()
   items regardless of code order — ::before bypasses that. */
[data-testid="stSidebar"] > div:first-child::before {
    content: "🧠  LangChain Hub";
    display: block;
    padding: 16px 20px 12px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.2px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
    margin-bottom: 6px;
}
[data-testid="stSidebarNavSeparator"] span {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.9px !important;
    text-transform: uppercase !important;
    color: rgba(128,128,128,0.6) !important;
    padding: 10px 20px 4px !important;
    display: block !important;
}
[data-testid="stSidebarNavItems"] a {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 6px 12px !important;
    margin: 1px 8px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    text-decoration: none !important;
    transition: background 0.15s ease !important;
    line-height: 1.3 !important;
}
[data-testid="stSidebarNavItems"] a:hover {
    background: rgba(128,128,128,0.08) !important;
}
[data-testid="stSidebarNavItems"] a[aria-current="page"] {
    background: rgba(49,130,206,0.1) !important;
    color: #2b6cb0 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebarNavItems"] {
    padding: 0 0 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Credential resolution ─────────────────────────────────────
# Both keys resolved upfront — pages hit session_state fast-path.
#
# hub_utils.py (at repo root) uses importlib to load credentials
# from their real locations inside the embeddings/ module folder.
# This avoids name collisions with module-level utils.py files and
# works reliably on Streamlit Cloud regardless of sys.path state.
from hub_utils import get_embedding_key, get_chat_key

embed_key, embed_provider, embed_model = get_embedding_key()
chat_key,  chat_provider,  chat_model  = get_chat_key()

# ── Module selector ───────────────────────────────────────────
# IMPORTANT: st.navigation() always occupies the TOP of the sidebar.
# Any st.sidebar widget added via `with st.sidebar:` renders BELOW
# the navigation list — which is why the title and dropdown appeared
# at the bottom of the screen in the previous version.
#
# Fix: use st.query_params to persist the active module across page
# navigations, and render the module switcher in the MAIN content area
# (top of main-home.py) rather than the sidebar. The sidebar title is
# injected via CSS ::before so it has no widget render order issue.
#
# The module filter still works: selecting a module rebuilds the pages
# dict passed to st.navigation(), collapsing the sidebar to only that
# module's sections. The selection is stored in st.query_params so it
# survives clicking between pages without resetting to "all".
#
# MODULE_OPTIONS maps display label → internal key used in ALL_PAGES.
# Add a new entry here whenever you add a new module.
MODULE_OPTIONS = {
    "🏠 All modules":        "all",
    "🔢 Embeddings & RAG":   "embeddings",
    "🖼️ Image Processing":   "image",
    "🎙️ Audio Processing":   "audio",      # ← add this
    "⚡ Streamlit Patterns": "streamlit",
}

# Read the active module from query params (persists across navigation).
# Falls back to "all" if the param is missing or invalid.
raw_param = st.query_params.get("module", "all")
selected_module = raw_param if raw_param in MODULE_OPTIONS.values() else "all"

# ── Page registry ─────────────────────────────────────────────
# ALL_PAGES groups every st.Page() by module key.
# The active module selection above filters this down to the relevant
# sections before passing to st.navigation().
#
# Structure:
#   ALL_PAGES = {
#       "module_key": {
#           "Section Label": [ st.Page(...), ... ],
#           ...
#       },
#       ...
#   }
#
# To add a new page:   add one st.Page() to the right module + section.
# To add a new section inside a module: add a new key → list pair.
# To add a new module: add a new top-level key matching MODULE_OPTIONS.

ALL_PAGES = {

    # ── Shared home ───────────────────────────────────────────
    # Always shown regardless of which module is selected.
    "home": {
        "🏠 Home": [
            st.Page("main-home.py", title="Hub Overview", icon="🏠",
                    default=True, url_path="hub"),
        ],
    },

    # ── Embeddings module ─────────────────────────────────────
    # Needs EMBEDDING KEY. Pages: similarity, vector search, RAG.
    "embeddings": {
        "🔢 Embeddings": [
            st.Page("embeddings/pages/home.py",
                    title="Embeddings Overview", icon="🔢", url_path="emb-home"),
            st.Page("embeddings/pages/01_embed_similarity.py",
                    title="Text Similarity",     icon="📊", url_path="emb-similarity"),
        ],
        "🗄️ Vector DBs": [
            st.Page("embeddings/pages/08_chroma_job_search.py",
                    title="Job Search (Chroma)", icon="💼", url_path="vdb-chroma-jobs"),
        ],
        "🧠 RAG": [
            st.Page("embeddings/pages/10_rag_demo.py",
                    title="RAG — Text",          icon="🧠", url_path="rag-text"),
            st.Page("embeddings/pages/11_rag_demo_history_aware.py",
                    title="RAG — History Aware", icon="🧠", url_path="rag-history"),
            st.Page("embeddings/pages/12_pdf-rag-demo.py",
                    title="RAG — PDF",           icon="🧠", url_path="rag-pdf"),
        ],
    },

    # ── Image Processing module ───────────────────────────────
    # Needs CHAT KEY only — no embeddings.
    "image": {
        "🖼️ Image Processing": [
            st.Page("image-processing/pages/home.py",
                    title="Vision Overview",      icon="🖼️", url_path="img-home"),
            st.Page("image-processing/pages/20_describe_image_assistant.py",
                    title="Image Q&A",            icon="🖼️", url_path="img-qa"),
            st.Page("image-processing/pages/21_kyc_verification.py",
                    title="KYC Verification",     icon="🪪",  url_path="img-kyc"),
            st.Page("image-processing/pages/22_product_comparison.py",
                    title="Product Comparison",   icon="🥗",  url_path="img-products"),
        ],
    },

    # ── Audio Processing module ───────────────────────────────────
    # Needs CHAT KEY only — no embeddings.
    # Whisper runs locally — no API key needed for transcription.
    "audio": {
        "🎙️ Audio Processing": [
            st.Page("audio-processing-whisper/pages/home.py",
                    title="Audio Overview", icon="🎙️", url_path="aud-home"),
            st.Page("audio-processing-whisper/pages/30_audio_assistant.py",
                    title="Audio Q&A", icon="🎧", url_path="aud-qa"),
        ],
    },

    # ── Streamlit Patterns module ─────────────────────────────
    # Needs CHAT KEY only. Core LangChain + Streamlit patterns.
    "streamlit": {
        "📝 Prompt Templates": [
            st.Page("streamlit/pages/home.py",
                    title="Patterns Overview",        icon="⚡", url_path="st-home"),
            st.Page("streamlit/pages/1_prompt_template_no_variable.py",
                    title="Prompt — No Variable",     icon="1️⃣",  url_path="st-prompt-basic"),
            st.Page("streamlit/pages/2_prompt_template_with_variable.py",
                    title="Prompt — With Variable",   icon="2️⃣",  url_path="st-prompt-vars"),
            st.Page("streamlit/pages/3_travelApp.py",
                    title="Travel App",               icon="✈️",  url_path="st-travel"),
        ],
        "⛓️ Chains": [
            st.Page("streamlit/pages/4_simplechain_lcel.py",
                    title="Simple Chain (LCEL)",      icon="⛓️",  url_path="st-chain"),
            st.Page("streamlit/pages/7_generate_blog_post.py",
                    title="Blog Post Generator",      icon="✍️",  url_path="st-blog"),
        ],
        "💬 Chat & Memory": [
            st.Page("streamlit/pages/8_chatPromptTemplate.py",
                    title="Chat Prompt Template",     icon="💬",  url_path="st-chat"),
            st.Page("streamlit/pages/9_streamlit_history_chatprompttemplate.py",
                    title="Chat with History",        icon="🕘",  url_path="st-history"),
        ],
        "🛠️ Advanced": [
            st.Page("streamlit/pages/5_mindful_morning_coach.py",
                    title="Mindful Morning Coach",    icon="🧘",  url_path="st-mindful"),
            st.Page("streamlit/pages/6_TextToSpeech.py",
                    title="Text to Speech",           icon="🔊",  url_path="st-tts"),
        ],
    },

    # ── Future modules — add full block when ready ────────────
    # "agents": {
    #     "🤖 Agents": [
    #         st.Page("agents/pages/home.py",
    #                 title="Agents Overview", icon="🤖", url_path="ag-home"),
    #         st.Page("agents/pages/01_react_agent.py",
    #                 title="ReAct Agent",     icon="🔄", url_path="ag-react"),
    #     ],
    # },
    # "fine-tuning": {
    #     "🎯 Fine-Tuning": [
    #         st.Page("fine-tuning/pages/home.py",
    #                 title="Fine-Tune Overview", icon="🎯", url_path="ft-home"),
    #     ],
    # },

}

# ── Build filtered pages dict ─────────────────────────────────
# Always include the shared home section.
# Then merge in the selected module's sections (or all modules if "all").
active_pages = dict(ALL_PAGES["home"])   # always present

if selected_module == "all":
    # Merge every module's sections in order
    for mod_key in ["embeddings", "image", "audio", "streamlit"]:
        active_pages.update(ALL_PAGES.get(mod_key, {}))
else:
    active_pages.update(ALL_PAGES.get(selected_module, {}))

# ── Run navigation ────────────────────────────────────────────
pg = st.navigation(active_pages)
pg.run()