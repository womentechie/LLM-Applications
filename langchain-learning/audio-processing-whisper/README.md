# 🎙️ Audio Processing (Whisper)

A self-contained module for transcribing audio and querying it with LLMs.
Upload an audio file — Whisper transcribes it locally on your machine,
then ask any question about the transcript using any chat LLM.

---

## 📁 Folder Structure

```
audio-processing-whisper/
├── app.py                              # Entry point — run this standalone
├── utils.py                            # Chat model key + factory (chat only)
├── requirements.txt                    # Dependencies
├── README.md                           # This file
├── .env                                # API keys (never commit this)
├── data/                               # Sample audio files for testing
└── pages/
    ├── home.py                         # Landing page
    └── 30_audio_assistant.py           # Audio transcription + Q&A — core demo
```

---

## 🚀 Getting Started

### Step 1 — Activate your virtual environment

```bash
cd LLM-Applications/langchain-learning/audio-processing-whisper

# Mac / Linux
source ../.venv/bin/activate

# Windows
..\.venv\Scripts\activate
```

### Step 2 — Verify pip3 points inside your venv

```bash
which pip3    # ✅ should show .venv/bin/pip3
which python  # ✅ should show .venv/bin/python
```

> **Rule for Intel Mac:** always use `pip3` (not `pip`) when your venv
> is active in PyCharm. `pip3` reliably points inside `.venv`.

### Step 3 — Install system dependency (ffmpeg)

faster-whisper uses ffmpeg to decode audio formats (MP3, M4A, FLAC etc.).
Install it before running the app:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Step 4 — Install Python dependencies

```bash
pip3 install -r requirements.txt
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔑 Key System (`utils.py`)

### How it works

This module only needs **one key** — the chat model key. No embedding key
is required. Whisper runs locally with no API key. The chat LLM is only
used for Q&A over the transcript text, not for transcription itself.

```
1. st.session_state  →  already entered this session (fast path)
2. Streamlit UI      →  provider + model dropdown + key input
```

Keys are stored **only** in `st.session_state` — never in `os.environ`.

### Supported chat providers

Any text-capable model works here — the LLM only sees the transcript,
not the audio file itself.

| Provider | Models | Key prefix |
|---|---|---|
| OpenAI | `gpt-4o` ✅, `gpt-4o-mini` ✅ | `sk-` |
| Anthropic | `claude-3-5-sonnet-20241022` ✅, `claude-3-haiku` ✅ | `sk-ant-` |
| Google Gemini | `gemini-2.0-flash` ✅, `gemini-1.5-pro` ✅ | `AIza` |
| Groq | `llama-3.3-70b-versatile` ✅ — fast inference | `gsk_` |
| Mistral | `mistral-small-latest` ✅ | — |

---

## 📐 Core Concepts

### Why no embeddings?

| Feature | Needs embeddings? | Why |
|---|---|---|
| RAG / document Q&A | ✅ Yes | Text → vectors → similarity search |
| Audio Q&A (this module) | ❌ No | Transcript sent directly in the chat prompt |

The transcript is injected straight into the LangChain `ChatPromptTemplate`
as plain text. No vector store is involved.

### Two-stage pipeline

```
Uploaded audio file  (WAV / MP3 / M4A / FLAC / OGG)
          │
          ▼
  faster-whisper  (local, CPU, no API key)
  speech → text transcript
          │
          ▼
  ChatPromptTemplate:
    system: "You analyse audio transcriptions."
    human:  "{transcription}\n\nQuestion: {query}"
          │
          ▼
  Chat LLM  (gpt-4o / claude / gemini ...)
          │
          ▼
  Natural-language answer grounded in the transcript
```

**Audio never leaves your machine** — only the text transcript is sent
to the cloud LLM. Whisper runs entirely locally.

### Why faster-whisper instead of openai-whisper?

`faster-whisper` is a re-implementation of OpenAI Whisper using CTranslate2.
It runs **2–4× faster on CPU** and uses significantly less memory, while
producing identical transcription output.

`compute_type="int8"` quantises model weights to 8-bit integers — further
reducing RAM usage with negligible accuracy loss, making it practical on a
laptop without a GPU.

### Whisper model sizes

| Model | Parameters | Speed | Best for |
|---|---|---|---|
| `tiny` | 39M | Fastest | Quick demos, short clips |
| `base` | 74M | Fast | Good default — used in this module |
| `small` | 244M | Moderate | Better accuracy, still CPU-friendly |
| `medium` | 769M | Slow | High accuracy, noisy audio |
| `large-v3` | 1.5B | Slowest | Production, multilingual |

Change the model size in `30_audio_assistant.py`:
```python
WhisperModel("base", compute_type="int8")   # ← change "base" to your choice
```

---

## 📋 Usage Pattern

### Standard audio page pattern

```python
import os
import tempfile
import streamlit as st
from faster_whisper import WhisperModel
from langchain_core.prompts import ChatPromptTemplate
from utils import get_or_set_api_key, build_llm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # suppresses macOS OpenMP warning

# ── Auth ──────────────────────────────────────────────────────
chat_key, chat_provider, chat_model = get_or_set_api_key()
llm = build_llm(chat_provider, chat_model, chat_key)

# ── Load Whisper once ─────────────────────────────────────────
# @st.cache_resource keeps the model in memory across reruns
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", compute_type="int8")

whisper_model = load_whisper_model()

# ── Transcribe ────────────────────────────────────────────────
def get_transcription(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path)
    return " ".join(seg.text.strip() for seg in segments).strip()

# ── Q&A ───────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You analyse audio transcriptions. Answer based only on the transcript."),
    ("human", "Transcript:\n{transcription}\n\nQuestion: {query}"),
])
chain = prompt | llm

# ── UI ────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name
    try:
        transcript = get_transcription(audio_path)
        st.write(transcript)
        query = st.text_input("Ask a question:")
        if query:
            response = chain.invoke({"transcription": transcript, "query": query})
            st.write(response.content)
    finally:
        os.remove(audio_path)   # always clean up temp file
```

---

## 📂 Pages

### `30_audio_assistant.py` — Audio Q&A

The core demo. Upload any WAV, MP3 or M4A file and ask questions about it.

**How it works:**
1. User uploads an audio file via `st.file_uploader`
2. File is written to a temp path — faster-whisper needs a real file path, not an in-memory buffer
3. Whisper transcribes the audio into text locally on your machine
4. `ChatPromptTemplate` combines the transcript + user's question
5. `chain.invoke()` sends both to the chat LLM
6. `response.content` is displayed as the answer
7. Temp file is deleted in a `finally` block — guaranteed cleanup

**Key features:**
- Provider-agnostic — works with any chat model in `utils.py`
- Session-safe auth — key stored only in `session_state`
- Cached Whisper model — loads once, reused across all reruns via `@st.cache_resource`
- Safe temp file handling — unique path per upload, always cleaned up
- Empty transcript guard — friendly warning if no speech detected

---

## 🎵 Supported Audio Formats

| Format | Notes |
|---|---|
| WAV | Best quality — no compression artefacts |
| MP3 | Most common — fully supported |
| M4A | Apple / iPhone recordings — fully supported |
| FLAC | Lossless — requires ffmpeg |
| OGG | Vorbis audio — requires ffmpeg |

All formats require `ffmpeg` to be installed on the system.

---

## ⚠️ Platform Notes (Intel Mac x86_64)

| Issue | Cause | Fix |
|---|---|---|
| `pip` installs to wrong Python | System `pip` vs venv `pip3` | Always use `pip3` in PyCharm venv |
| `KMP_DUPLICATE_LIB_OK` warning | PyTorch + numpy share OpenMP | Set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` before model load |
| `ffmpeg not found` | Not installed or not on PATH | `brew install ffmpeg` |
| Slow transcription | Large model on CPU | Switch to `"tiny"` or `"base"` |

**Golden rule for Intel Mac + PyCharm:**
```bash
which pip3    # must show .venv/bin/pip3
pip3 install -r requirements.txt
```

---

## ➕ Adding a New Audio Page

1. Create `pages/NN_your_page.py` following the usage pattern above.
2. Import `get_or_set_api_key` and `build_llm` from `utils`.
3. Use `@st.cache_resource` for any model you load — never load inside a function called on every rerun.
4. Always use `tempfile.NamedTemporaryFile` — never hardcode a filename.
5. Add the page to the `"🎙️ Audio Processing"` section in `app.py`:

```python
"🎙️ Audio Processing": [
    st.Page("pages/30_audio_assistant.py", title="Audio Q&A",        icon="🎙️"),
    st.Page("pages/31_your_new_page.py",   title="Your New Feature", icon="🎧"),
],
```

---

## 📦 Dependencies

```
# Core
streamlit
python-dotenv

# Transcription (local — no API key needed)
faster-whisper          # Whisper re-implementation via CTranslate2

# LangChain
langchain-core
langchain-community

# Chat providers (install only what you need)
langchain-openai        # gpt-4o, gpt-4o-mini
langchain-anthropic     # claude-3-5-sonnet, claude-3-haiku
langchain-google-genai  # gemini-2.0-flash, gemini-1.5-pro
langchain-groq          # llama-3.3-70b (fast inference)
langchain-mistralai     # mistral-small
```

**System dependency (not pip):**
```
ffmpeg   ← required for MP3, M4A, FLAC decoding
```

Add to `packages.txt` for Streamlit Cloud deployment:
```
ffmpeg
```

---

## 🗒️ Notes

- `@st.cache_resource` is essential for `WhisperModel` — loading it takes several seconds and would run on every widget interaction without caching.
- Temp files use `tempfile.NamedTemporaryFile(delete=False, suffix=...)` because faster-whisper requires a real disk path, not an in-memory buffer. The `finally` block guarantees cleanup even if transcription raises an exception.
- `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` suppresses a harmless warning on macOS where PyTorch and numpy both link the system OpenMP library.
- Whisper auto-detects the spoken language — access it via `info.language` returned by `model.transcribe()` if you need to display or log it.
- For Streamlit Cloud: add `ffmpeg` to `packages.txt` and `faster-whisper` to `requirements.txt`.