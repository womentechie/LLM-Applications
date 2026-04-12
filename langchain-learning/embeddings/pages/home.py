# ============================================================
# Home Page — landing page for the Embeddings Demo Hub
# ============================================================
# By the time this page loads, app.py has already resolved:
#
#   EMBEDDING model (embeddings_utils.py):
#     session_state["embed_provider"]  → e.g. "OpenAI"
#     session_state["embed_model"]     → e.g. "text-embedding-3-small"
#     session_state["embed_api_key"]   → user's embedding key
#
#   CHAT model (utils.py):
#     session_state["provider"]        → e.g. "Anthropic"
#     session_state["model"]           → e.g. "claude-sonnet-4-5"
#     session_state["api_key"]         → user's chat key
#
# This page reads both from session_state for display only —
# no auth logic or key prompts needed here.
# ============================================================

import streamlit as st
from embeddings_utils import get_embedding_dimensions

st.title("🔢 Embeddings Demo Hub")
st.markdown(
    "A hands-on collection of LangChain embedding demos — "
    "from basic vector generation to semantic search and RAG."
)

# ── Active model banners ───────────────────────────────────────
# Show both active models so users know what's configured before
# navigating to any demo page.

col1, col2 = st.columns(2)

with col1:
    # Embedding model — used by similarity, job search, and RAG retrieval
    embed_provider = st.session_state.get("embed_provider", "Unknown")
    embed_model    = st.session_state.get("embed_model",    "Unknown")
    embed_key      = st.session_state.get("embed_api_key",  None)

    st.success(f"✅ Embedding: **{embed_provider}** · `{embed_model}`")

    dims = get_embedding_dimensions(embed_provider, embed_model)
    if dims:
        st.caption(f"📐 {dims}-dimensional vectors")
    if embed_key:
        st.caption(f"🔑 Key: `{embed_key[:8]}...`")
    elif embed_provider == "HuggingFace (Local)":
        st.caption("🖥️ Running locally — no key needed")

with col2:
    # Chat model — used by RAG to generate answers from retrieved context
    chat_provider = st.session_state.get("provider", "Unknown")
    chat_model    = st.session_state.get("model",    "Unknown")
    chat_key      = st.session_state.get("api_key",  None)

    st.success(f"✅ Chat LLM: **{chat_provider}** · `{chat_model}`")
    st.caption("Used by RAG to generate answers")
    if chat_key:
        st.caption(f"🔑 Key: `{chat_key[:8]}...`")

st.divider()

# ── What are embeddings? ───────────────────────────────────────
st.subheader("🧠 What are embeddings?")
st.markdown("""
An **embedding model** converts text into a list of numbers (a **vector**)
that captures its semantic meaning. Similar texts produce vectors that are
mathematically close together in vector space.

```
"The cat sat on the mat"   →  [0.23, -0.81,  0.44, ...]  # 1536 numbers
"A feline rested on a rug" →  [0.22, -0.80,  0.43, ...]  # very similar!
"The stock market crashed"  →  [0.91,  0.14, -0.63, ...]  # very different
```

Embeddings power:
- 🔍 **Semantic search** — find similar documents, not just keyword matches
- 🧠 **RAG** — give LLMs long-term memory by retrieving relevant context
- 🎯 **Recommendations** — surface similar content
- 🗂️ **Clustering** — group similar texts automatically
""")

st.divider()

# ── App Index ──────────────────────────────────────────────────
st.subheader("📚 What's inside")

st.markdown("""
**🔢 Embeddings**
- 🔡 **Text Similarity** — embed two texts, measure cosine similarity,
  see the raw vectors and an interpreted score

**🗄️ Vector DBs**
- 💼 **Job Search (Chroma)** — semantic search over job listings using
  ChromaDB. Find roles by meaning, not just keywords

**🧠 RAG**
- 🧠 **RAG Demo** — ask questions about your own documents.
  Uses the embedding model to retrieve relevant chunks, then the
  chat LLM to generate a grounded answer with source references

*Coming soon:*
- ✂️ **Chunking strategies** — compare fixed, recursive, and semantic splitting
- ⚡ **FAISS search** — fast local vector search for larger datasets
- 💬 **RAG with memory** — conversational RAG with full chat history
""")

st.divider()

# ── How RAG uses both models ───────────────────────────────────
st.subheader("🔄 How RAG uses both models")

st.markdown("""
RAG (Retrieval-Augmented Generation) is why this app needs two models:

```
User question
     │
     ▼
Embed question          ← EMBEDDING model (embed_provider)
     │
     ▼
Search vector store     ← finds top-k most similar document chunks
     │
     ▼
Retrieved chunks + question → CHAT model (chat_provider)
     │
     ▼
Grounded answer with sources
```

The **embedding model** handles retrieval — it finds relevant context.
The **chat model** handles generation — it writes the answer.
You can mix providers freely: e.g. OpenAI embeddings + Claude for answers.
""")

st.divider()

# ── Provider quick reference ───────────────────────────────────
st.subheader("🌐 Supported embedding providers")

st.markdown("""
| Provider | Best for | Dimensions |
|---|---|---|
| OpenAI `text-embedding-3-small` | General purpose, best value | 1536 |
| OpenAI `text-embedding-3-large` | Highest accuracy | 3072 |
| Google Gemini `text-embedding-004` | Multilingual | 768 |
| Cohere `embed-multilingual-v3.0` | 100+ languages | 1024 |
| Mistral `mistral-embed` | Fast, compact | 1024 |
| HuggingFace `all-MiniLM-L6-v2` | Free, local, offline | 384 |

> ❌ **Anthropic** and **Groq** do not offer embedding models.
""")

st.divider()
st.caption("Built with LangChain · Streamlit · Python")