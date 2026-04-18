# 🖼️ Image Processing

A self-contained module for working with vision-capable LLMs using LangChain.
Upload images and ask questions — the image is passed directly to the model
as a base64-encoded data URI, no vector store or embeddings needed.

---

## 📁 Folder Structure

```
image-processing/
├── app.py                              # Entry point — run this
├── utils.py                            # Chat model key + factory (chat only)
├── requirements.txt                    # Dependencies
├── README.md                           # This file
├── .env                                # API keys (never commit this)
├── data/                               # Sample images for testing
└── pages/
    ├── home.py                         # Landing page
    ├── 01_embed_similarity.py          # Text similarity (optional)
    └── 20_describe_image_assistant.py  # Image Q&A — core demo
```

---

## 🚀 Getting Started

### Step 1 — Activate your virtual environment

```bash
cd LLM-Applications/langchain-learning/image-processing

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

### Step 3 — Install dependencies

```bash
pip3 install -r requirements.txt
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔑 Key System (`utils.py`)

### How it works

This module only needs **one key** — the chat model key. No embedding key
is required because images are sent directly to the LLM, not stored in a
vector database.

```
1. st.session_state  →  already entered this session (fast path)
2. Streamlit UI      →  provider + model dropdown + key input
```

Keys are stored **only** in `st.session_state` — never in `os.environ`.

### Supported vision providers

| Provider | Vision-capable models | Key prefix |
|---|---|---|
| OpenAI | `gpt-4o` ✅, `gpt-4-turbo` ✅ | `sk-` |
| Anthropic | `claude-3-5-sonnet-20241022` ✅, `claude-3-opus` ✅ | `sk-ant-` |
| Google Gemini | `gemini-2.0-flash` ✅, `gemini-1.5-pro` ✅ | `AIza` |
| Groq | ⚠️ Limited — check model docs | `gsk_` |
| Cohere / Mistral | ❌ Text-only in current LangChain wrappers | — |

> **Tip:** `gpt-4o` and `claude-3-5-sonnet` give the best results for
> detailed image understanding. Use `gemini-2.0-flash` for the lowest cost.

---

## 📐 Core Concepts

### Why no embeddings?

| Feature | Needs embeddings? | Why |
|---|---|---|
| RAG / document Q&A | ✅ Yes | Text → vectors → similarity search |
| Image Q&A (this module) | ❌ No | Image sent directly in the prompt |

The image never touches a vector store. It is base64-encoded and injected
straight into the LangChain message as a `data:image/jpeg;base64,...` URI,
which the vision LLM reads natively alongside the user's question.

### How the image reaches the LLM

```
User uploads image (JPG / PNG)
          │
          ▼
  base64.b64encode(image_file.read()).decode()
          │
          ▼
  ChatPromptTemplate — multimodal human message:
    [
      {"type": "text",      "text": "{input}"},
      {"type": "image_url", "image_url": {
          "url":    "data:image/jpeg;base64,{image}",
          "detail": "low"   ← or "high" for fine detail
      }}
    ]
          │
          ▼
  chain.invoke({"input": question, "image": base64_string})
          │
          ▼
  AIMessage.content  →  natural-language answer
```

### The `detail` parameter

| Setting | Speed | Cost | Best for |
|---|---|---|---|
| `"low"` | Fast | Cheap | General description, object ID, colours |
| `"high"` | Slower | Higher | Reading text, charts, fine detail |

---

## 📋 Usage Pattern

### Standard vision page pattern

```python
import base64
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from utils import get_or_set_api_key, build_llm

# ── Auth ──────────────────────────────────────────────────────
# Keys stored only in session_state — never os.environ
chat_key, chat_provider, chat_model = get_or_set_api_key()
llm = build_llm(chat_provider, chat_model, chat_key)

# ── Encode image ──────────────────────────────────────────────
def encode_image(image_file) -> str:
    return base64.b64encode(image_file.read()).decode()

# ── Build multimodal prompt ───────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can describe images."),
    ("human", [
        {"type": "text",      "text": "{input}"},
        {"type": "image_url", "image_url": {
            "url":    "data:image/jpeg;base64,{image}",
            "detail": "low",
        }},
    ]),
])

chain = prompt | llm

# ── UI ────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
question      = st.text_input("Ask a question about the image")

if question:
    if not uploaded_file:
        st.warning("Please upload an image first.")
    else:
        image    = encode_image(uploaded_file)
        response = chain.invoke({"input": question, "image": image})
        st.write(response.content)
```

---

## 📂 Pages

### `20_describe_image_assistant.py` — Image Q&A

The core demo. Upload any JPG or PNG and ask any question about it.

**How it works:**
1. User uploads an image via `st.file_uploader`
2. Image is base64-encoded with `encode_image()`
3. A multimodal `ChatPromptTemplate` combines the question + image
4. `chain.invoke()` sends both to the vision LLM
5. `response.content` is displayed as the answer

**Key features:**
- Provider-agnostic — works with any vision-capable model in `utils.py`
- Session-safe auth — key stored only in `session_state`
- Guard against missing image — friendly warning if no file uploaded

---

## ⚠️ Platform Notes (Intel Mac x86_64)

| Issue | Cause | Fix |
|---|---|---|
| `pip` installs to wrong Python | System `pip` vs venv `pip3` | Always use `pip3` in PyCharm venv |
| Wrong provider error | Key prefix mismatch | `utils.py` warns if key doesn't match selected provider |

**Golden rule for Intel Mac + PyCharm:**
```bash
which pip3    # must show .venv/bin/pip3
pip3 install -r requirements.txt
```

---

## ➕ Adding a New Vision Page

1. Create `pages/NN_your_page.py` following the usage pattern above.
2. Import `get_or_set_api_key` and `build_llm` from `utils` — no other imports needed.
3. Add the page to the `"🖼️ Image Processing"` section in `app.py`:

```python
"🖼️ Image Processing": [
    st.Page("pages/20_describe_image_assistant.py", title="Image Q&A",       icon="🖼️"),
    st.Page("pages/21_your_new_page.py",            title="Your New Feature", icon="🔍"),
],
```

---

## 📦 Dependencies

```
# Core
streamlit
python-dotenv
langchain-core
langchain-community

# Chat / vision providers (install only what you need)
langchain-openai        # gpt-4o, gpt-4-turbo
langchain-anthropic     # claude-3-5-sonnet, claude-3-opus
langchain-google-genai  # gemini-2.0-flash, gemini-1.5-pro
langchain-groq          # limited vision support
```

---

## 🗒️ Notes

- `encode_image()` calls `.read()` on the Streamlit `UploadedFile` object, which **consumes the buffer**. Call it once and store the result — do not call it twice on the same upload.
- The `data:image/jpeg;base64,...` URI format works for both JPG and PNG inputs. The MIME type in the URI (`image/jpeg`) is accepted by all providers regardless of the actual file extension.
- For production use, add image size validation before encoding — very large images can exceed provider context limits or significantly increase cost.
- `app.py` in this module does **not** import `embeddings_utils` — keep it that way. If a future page needs embeddings, add `get_or_set_embedding_key()` to `app.py` at that point.