# ============================================================
# Home Page — shown by default when the app first loads
# ============================================================
# This page serves two purposes:
#   1. Confirms the active LLM provider and model to the user
#   2. Acts as a visual index/guide to all the demo pages
#
# NOTE: The API key is already resolved in app.py (the entry point)
# before any page runs. By the time this page loads, session_state
# already contains "api_key", "provider", and "model".
# We just read them here for display — no auth logic needed.
# ============================================================

import streamlit as st

st.title("🤖 LangChain Demo Hub")
st.markdown("A collection of LangChain + Streamlit demos, from basic prompt templates to memory-enabled chatbots.")

# ── Active Provider Banner ────────────────────────────────────
# Read from session_state — set once in app.py, shared everywhere.
provider = st.session_state.get("provider", "Unknown")
model    = st.session_state.get("model",    "Unknown")
api_key  = st.session_state.get("api_key",  "")

st.success(f"✅ Active provider: **{provider}** · `{model}`")
if api_key:
    st.caption(f"Key starts with: `{api_key[:8]}...`")

st.divider()

# ── App Index ────────────────────────────────────────────────
st.subheader("📚 What's inside")

st.markdown("""
**📝 Prompt Templates**
- 🎈 **Single Variable** — Basic `PromptTemplate` with one `{placeholder}`
- ✨ **Multiple Variables** — Injecting several values into one template
- 🚢 **Travel App** — Real-world multi-variable template with constrained inputs

**⛓️ LCEL Chains**
- ⛓️ **Simple Chain** — Two chained prompts using the `|` pipe operator
- 👩‍💻 **Blog Post Generator** — `RunnablePassthrough.assign()` to prevent variable loss

**🤖 Chat & Memory**
- 🏃 **Agile Coach** — `ChatPromptTemplate` with system/human/ai roles
- 🔬 **Science Coach** — Full conversational memory with `RunnableWithMessageHistory`

**🛠️ Advanced**
- 🧘 **Mindful Morning Coach** — Dynamic prompts, UI customisation, download button
- 💬 **Text To Speech** — LLM output piped into gTTS for in-browser audio playback
""")

st.divider()
st.caption("Built with LangChain · Streamlit · Python")
