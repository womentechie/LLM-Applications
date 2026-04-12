# ============================================================
# CONCEPT: Streamlit Multi-Page App with st.navigation()
# ============================================================
# st.navigation() is the modern (Streamlit 1.36+) way to build
# multi-page apps. It replaces the older "pages/" folder convention.
#
# HOW IT WORKS:
#   1. Define each page with st.Page(), pointing to a .py file
#   2. Optionally group pages into sections using a dict
#   3. Call st.navigation(...).run() to render the selected page
#
# WHY THIS IS BETTER THAN THE "pages/" FOLDER CONVENTION:
#   - Full control over page titles, icons, and order
#   - Pages can be grouped into labelled sections in the sidebar
#   - The entry-point file (this file) can render a home page
#     when no sub-page is selected
#   - Works cleanly with shared utils.py (one key for all pages)
#
# FILE STRUCTURE EXPECTED:
#   app.py                          ← this file (entry point)
#   utils.py                        ← shared key + LLM factory
#   pages/
#     home.py
#     1_prompt_template_no_variable.py
#     2_prompt_template_with_variable.py
#     3_travelApp.py
#     4_simplechain_lcel.py
#     5_mindful_morning_coach.py
#     6_TextToSpeech.py
#     7_generate_blog_post.py
#     8_chatPromptTemplate.py
#     9_streamlit_history_chatprompttemplate.py
#
# RUN WITH:  streamlit run app.py
# ============================================================

import streamlit as st
from utils import get_or_set_api_key

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# ── App-wide page config (set once here, not in individual pages) ─
st.set_page_config(
    page_title="LangChain Demo Hub",
    page_icon="🤖",
    layout="centered"
)

# ── Resolve API key ONCE for the whole app ────────────────────────
# Because all pages import utils.py and check session_state first,
# resolving the key here means sub-pages never show the auth screen.
# The key is stored in st.session_state and shared across all pages.
api_key, provider, model = get_or_set_api_key()

# ── Page Definitions — grouped into logical sections ──────────────
# st.Page(path, title, icon) registers a sub-page.
# Grouping via a dict adds labelled section headers in the sidebar.
# Pages are shown in the order they appear in the lists.

pages = {
    "🏠 Home": [
        st.Page("pages/home.py", title="Getting Started", icon="🏠", default=True),
    ],
    "📝 Prompt Templates": [
        st.Page("pages/1_prompt_template_no_variable.py", title="Single Variable", icon="🎈"),
        st.Page("pages/2_prompt_template_with_variable.py", title="Multiple Variables", icon="✨"),
        st.Page("pages/3_travelApp.py", title="Travel App", icon="🚢"),
    ],
    "⛓️ LCEL Chains": [
        st.Page("pages/4_simplechain_lcel.py", title="Simple Chain", icon="⛓️"),
        st.Page("pages/7_generate_blog_post.py", title="Blog Post Generator", icon="👩‍💻"),
    ],
    "🤖 Chat & Memory": [
        st.Page("pages/8_chatPromptTemplate.py", title="Agile Coach", icon="🏃"),
        st.Page("pages/9_streamlit_history_chatprompttemplate.py", title="Science Coach (Memory)", icon="🔬"),
    ],
    "🛠️ Advanced": [
        st.Page("pages/5_mindful_morning_coach.py", title="Mindful Morning Coach", icon="🧘"),
        st.Page("pages/6_TextToSpeech.py", title="Text To Speech", icon="💬"),
    ],
}

# ── Run Navigation ────────────────────────────────────────────────
# st.navigation() renders the sidebar with all grouped pages.
# .run() executes whichever page the user selects.
pg = st.navigation(pages)
pg.run()
