# ============================================================
# Home Page — landing page for the Image Processing Demo Hub
# ============================================================
# By the time this page loads, app.py has already resolved:
#
#   CHAT model (utils.py):
#     session_state["provider"]  → e.g. "OpenAI"
#     session_state["model"]     → e.g. "gpt-4o"
#     session_state["api_key"]   → user's chat key
#
# WHY NO EMBEDDING KEY?
#   Image processing sends images directly to a vision-capable LLM
#   as base64-encoded data URIs — no vector store or similarity search
#   is involved, so no embedding model is needed here.
#
# This page reads the chat model from session_state for display only —
# no auth logic or key prompts needed here.
# ============================================================

import streamlit as st

st.title("🖼️ Image Processing Demo Hub")
st.markdown(
    "A hands-on collection of LangChain vision demos — "
    "from basic image description to multimodal Q&A."
)

# ── Active model banner ────────────────────────────────────────
# Show the active chat model so users know what's configured before
# navigating to any demo page.
# NOTE: Only the chat model is shown here — no embedding model is
# needed for vision tasks in this module.

chat_provider = st.session_state.get("provider", "Unknown")
chat_model    = st.session_state.get("model",    "Unknown")
chat_key      = st.session_state.get("api_key",  None)

st.success(f"✅ Chat LLM: **{chat_provider}** · `{chat_model}`")
st.caption("Used by all image demos to generate descriptions and answers")
if chat_key:
    st.caption(f"🔑 Key: `{chat_key[:8]}...`")

st.divider()

# ── What is multimodal vision? ─────────────────────────────────
st.subheader("🧠 What is multimodal vision?")
st.markdown("""
A **vision-capable LLM** can accept both text and images as input.
The image is encoded as a base64 string and embedded directly in the
prompt payload — the model reads pixels alongside your question and
generates a grounded natural-language response.

```
User question + uploaded image
          │
          ▼
  base64-encode image
          │
          ▼
  Build multimodal prompt  ←  {"type": "text",      "text": question}
                               {"type": "image_url", "url":  "data:image/jpeg;base64,..."}
          │
          ▼
  Vision LLM (gpt-4o / claude-3-5-sonnet / gemini-2.0-flash ...)
          │
          ▼
  Natural-language answer about the image
```

**No embeddings or vector stores are used** — the image goes straight
into the model context, not into a retrieval pipeline.
""")

st.divider()

# ── App Index ──────────────────────────────────────────────────
st.subheader("📚 What's inside")

st.markdown("""
**🖼️ Image Processing**
- 🖼️ **Image Q&A** (`20_describe_image_assistant.py`) — upload any image
  and ask questions about it. The app encodes the image as base64 and
  passes it directly to the vision LLM alongside your question.

*Coming soon:*
- 🗂️ **Batch image description** — describe multiple images at once
- 🔍 **Image + document RAG** — combine vision with retrieval
- 🏷️ **Image classification** — label and tag images with an LLM
""")

st.divider()

# ── How vision inputs work ─────────────────────────────────────
st.subheader("🔄 How the image reaches the LLM")

st.markdown("""
LangChain's `ChatPromptTemplate` supports multimodal messages natively.
The human turn contains two content blocks sent together in one API call:

```python
ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can describe images."),
    ("human", [
        {"type": "text",      "text": "{input}"},
        {"type": "image_url", "image_url": {
            "url":    "data:image/jpeg;base64,{image}",
            "detail": "low",   # "low" = fast/cheap  |  "high" = fine detail
        }},
    ]),
])
```

The `detail` parameter controls how carefully the model inspects the image:

| Setting | Speed | Cost | Best for |
|---|---|---|---|
| `"low"` | Fast | Cheap | General description, object identification |
| `"high"` | Slower | Higher | Reading text, fine details, charts |
""")

st.divider()

# ── Supported vision models ────────────────────────────────────
st.subheader("🌐 Supported vision providers")

st.markdown("""
| Provider | Vision-capable models |
|---|---|
| OpenAI | `gpt-4o` ✅, `gpt-4-turbo` ✅ |
| Anthropic | `claude-3-5-sonnet-20241022` ✅, `claude-3-opus` ✅, `claude-3-haiku` ✅ |
| Google Gemini | `gemini-2.0-flash` ✅, `gemini-1.5-pro` ✅ |
| Groq | ⚠️ Limited vision support — check model docs |
| Cohere / Mistral | ❌ Text-only in current LangChain wrappers |

> **Tip:** `gpt-4o` and `claude-3-5-sonnet` give the best results for
> detailed image understanding. Use `gemini-2.0-flash` for the fastest
> response at lowest cost.
""")

st.divider()
st.caption("Built with LangChain · Streamlit · Python")