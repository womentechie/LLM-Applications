# ============================================================
# Home Page — landing page for the Audio Processing Demo Hub
# ============================================================
# By the time this page loads, app.py has already resolved:
#
#   CHAT model (utils.py):
#     session_state["provider"]  → e.g. "OpenAI"
#     session_state["model"]     → e.g. "gpt-4o"
#     session_state["api_key"]   → user's chat key
#
# WHY NO EMBEDDING KEY?
#   Audio processing transcribes speech to text using Whisper
#   (a local model — no API key needed for transcription), then
#   passes the transcript to a chat LLM for Q&A. No vector store
#   or similarity search is involved, so no embedding model is needed.
#
# This page reads the chat model from session_state for display only —
# no auth logic or key prompts needed here.
# ============================================================

import streamlit as st

st.title("🎙️ Audio Processing Demo Hub")
st.markdown(
    "A hands-on collection of audio + LLM demos — "
    "from speech transcription to intelligent audio Q&A."
)

# ── Active model banner ────────────────────────────────────────
# Show the active chat model so users know what's configured
# before navigating to any demo page.
# Whisper runs locally — it does not appear here because it
# requires no API key and is loaded on demand inside each page.

chat_provider = st.session_state.get("provider", "Unknown")
chat_model    = st.session_state.get("model",    "Unknown")
chat_key      = st.session_state.get("api_key",  None)

col1, col2 = st.columns(2)
with col1:
    st.success(f"💬 Chat LLM: **{chat_provider}** · `{chat_model}`")
    st.caption("Analyses and answers questions about transcripts")
    if chat_key:
        st.caption(f"🔑 Key: `{chat_key[:8]}...`")
with col2:
    st.info("🎙️ Whisper (local) · `faster-whisper`")
    st.caption("Transcribes audio — runs on CPU, no API key needed")

st.divider()

# ── How the audio pipeline works ──────────────────────────────
st.subheader("🧠 How audio Q&A works")
st.markdown("""
The pipeline has two distinct stages — a local transcription step
and a cloud LLM step. They use completely different models:

```
Uploaded audio file  (WAV / MP3 / M4A)
          │
          ▼
  Whisper (local, CPU)          ← faster-whisper, no API key
  speech → text transcript
          │
          ▼
  ChatPromptTemplate
    system: "You analyse audio transcriptions."
    human:  transcript + user's question
          │
          ▼
  Chat LLM  (gpt-4o / claude / gemini ...)
          │
          ▼
  Natural-language answer grounded in the transcript
```

**No audio is sent to the LLM** — only the text transcript is.
Whisper runs entirely on your machine so your audio stays local.
""")

st.divider()

# ── App index ──────────────────────────────────────────────────
st.subheader("📚 What's inside")
st.markdown("""
**🎙️ Audio Processing**
- 🎙️ **Audio Q&A** (`30_audio_assistant.py`) — upload any WAV, MP3 or M4A
  file, get an instant transcript via Whisper, then ask the LLM anything
  about what was said.

*Coming soon:*
- 📋 **Meeting summariser** — extract action items and key decisions
- 🌐 **Multilingual transcription** — Whisper auto-detects language
- 🔍 **Audio + RAG** — transcribe then search across multiple recordings
""")

st.divider()

# ── Whisper model sizes ────────────────────────────────────────
st.subheader("⚙️ Whisper model sizes")
st.markdown("""
`faster-whisper` is a re-implementation of OpenAI Whisper that runs
2–4× faster on CPU with the same accuracy. Choose the model size
in `30_audio_assistant.py` based on your speed vs. accuracy needs:

| Model | Size | Speed | Best for |
|---|---|---|---|
| `tiny` | 39M params | Fastest | Quick demos, short clips |
| `base` | 74M params | Fast | Good default — used in this demo |
| `small` | 244M params | Moderate | Better accuracy, still CPU-friendly |
| `medium` | 769M params | Slow | High accuracy for noisy audio |
| `large-v3` | 1.5B params | Slowest | Production, multilingual |

`compute_type="int8"` keeps memory low on CPU — no GPU needed.
""")

st.divider()

# ── Supported audio formats ────────────────────────────────────
st.subheader("🎵 Supported audio formats")
st.markdown("""
| Format | Notes |
|---|---|
| WAV | Best quality — no compression artefacts |
| MP3 | Most common — fully supported |
| M4A | Apple / iPhone recordings — fully supported |
| FLAC | Lossless — supported via ffmpeg |
| OGG | Supported via ffmpeg |

> **Requires ffmpeg** — install with `brew install ffmpeg` (macOS)
> or `apt install ffmpeg` (Linux). For Streamlit Cloud add `ffmpeg`
> to `packages.txt`.
""")

st.divider()
st.caption("Built with faster-whisper · LangChain · Streamlit · Python")