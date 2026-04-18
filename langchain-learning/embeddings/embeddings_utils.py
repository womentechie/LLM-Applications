# embeddings_utils.py
# ============================================================
# CONCEPT: Embeddings — turning text into vectors
# ============================================================
# An embedding model converts a piece of text into a list of
# numbers (a "vector") that captures its semantic meaning.
# Texts with similar meaning will have vectors that are
# mathematically close to each other in vector space.
#
# WHY EMBEDDINGS MATTER:
#   They are the foundation of:
#     - Semantic search  (find similar documents, not just keyword matches)
#     - RAG              (Retrieval-Augmented Generation — give LLMs long-term memory)
#     - Recommendations  (find similar items)
#     - Clustering       (group similar texts automatically)
#
# HOW THIS FILE RELATES TO utils.py:
#   utils.py     → manages CHAT models  (generate text)
#   this file    → manages EMBEDDING models (vectorise text)
#
#   They share the same:
#     - Provider registry pattern (EMBEDDING_PROVIDERS dict)
#     - Session-state-only security (no os.environ reads/writes)
#     - build_*() factory function pattern
#     - Auto-detection of provider from key prefix
#
#   A user typically needs BOTH:
#     llm       = build_llm(...)        ← from utils.py
#     embeddings = build_embeddings(...)← from this file
#
# PROVIDER SUPPORT FOR EMBEDDINGS:
#   ✅ OpenAI        — text-embedding-3-small / large (best general purpose)
#   ✅ Google Gemini — models/embedding-001 (strong multilingual)
#   ✅ Cohere        — embed-english-v3.0 / multilingual-v3.0
#   ✅ Mistral       — mistral-embed (compact, fast)
#   ✅ HuggingFace   — free, local, no API key needed
#   ❌ Anthropic     — does NOT offer embedding models (Claude is chat-only)
#   ❌ Groq          — does NOT offer embedding models (inference-only)
#
# USAGE EXAMPLE:
#   from embeddings_utils import get_or_set_embedding_key, build_embeddings
#
#   api_key, provider, model = get_or_set_embedding_key()
#   embeddings = build_embeddings(provider, model, api_key)
#
#   # Embed a single string
#   vector = embeddings.embed_query("What is LangChain?")
#   print(len(vector))  # e.g. 1536 dimensions for OpenAI small
#
#   # Embed multiple documents at once
#   vectors = embeddings.embed_documents(["doc one", "doc two", "doc three"])
# ============================================================

import streamlit as st


# ── Embedding Provider Registry ───────────────────────────────────────────────
# Same structure as PROVIDERS in utils.py:
#   key_prefix   : for auto-detecting provider from a pasted API key
#   env_var      : reference name only — NOT read at runtime (security)
#   models       : available embedding models for this provider
#   default_model: pre-selected in the dropdown
#   dimensions   : output vector size per model (useful for configuring
#                  vector stores like Pinecone, Chroma, FAISS)
#
# NOTE: Anthropic and Groq are intentionally excluded.
# Anthropic's Claude is a chat-only model with no embedding endpoint.
# Groq is an inference engine (runs other models fast) — it has no
# proprietary embedding model.

EMBEDDING_PROVIDERS = {
    "OpenAI": {
        "key_prefix":    "sk-",
        "env_var":       "OPENAI_API_KEY",
        "models": [
            "text-embedding-3-small",   # 1536 dims — best cost/performance balance
            "text-embedding-3-large",   # 3072 dims — highest accuracy
            "text-embedding-ada-002",   # 1536 dims — legacy, still widely used
        ],
        "default_model": "text-embedding-3-small",
        "dimensions": {
            "text-embedding-3-small":  1536,
            "text-embedding-3-large":  3072,
            "text-embedding-ada-002":  1536,
        },
    },
    "Google Gemini": {
        "key_prefix":    "AIza",
        "env_var":       "GOOGLE_API_KEY",
        "models": [
            "models/text-embedding-004",   # latest, 768 dims
            "models/embedding-001",        # 768 dims — stable, widely supported
        ],
        "default_model": "models/text-embedding-004",
        "dimensions": {
            "models/text-embedding-004": 768,
            "models/embedding-001":      768,
        },
    },
    "Cohere": {
        "key_prefix":    None,
        "env_var":       "COHERE_API_KEY",
        "models": [
            "embed-english-v3.0",           # 1024 dims — English only, high quality
            "embed-multilingual-v3.0",      # 1024 dims — 100+ languages
            "embed-english-light-v3.0",     # 384 dims  — faster, smaller
            "embed-multilingual-light-v3.0",# 384 dims  — fast multilingual
        ],
        "default_model": "embed-english-v3.0",
        "dimensions": {
            "embed-english-v3.0":            1024,
            "embed-multilingual-v3.0":       1024,
            "embed-english-light-v3.0":       384,
            "embed-multilingual-light-v3.0":  384,
        },
    },
    "Mistral": {
        "key_prefix":    None,
        "env_var":       "MISTRAL_API_KEY",
        "models": [
            "mistral-embed",   # 1024 dims — Mistral's only embedding model
        ],
        "default_model": "mistral-embed",
        "dimensions": {
            "mistral-embed": 1024,
        },
    },
    "HuggingFace (Local)": {
        # HuggingFace sentence-transformers run entirely locally.
        # No API key is required — models are downloaded from HuggingFace Hub
        # and cached on your machine. Free, private, offline-capable.
        "key_prefix":    None,
        "env_var":       None,   # No key needed
        "models": [
            "all-MiniLM-L6-v2",            # 384 dims  — fast, great for search
            "all-mpnet-base-v2",           # 768 dims  — more accurate, slower
            "paraphrase-MiniLM-L6-v2",     # 384 dims  — optimised for paraphrase
            "multi-qa-MiniLM-L6-cos-v1",   # 384 dims  — optimised for Q&A retrieval
        ],
        "default_model": "all-MiniLM-L6-v2",
        "dimensions": {
            "all-MiniLM-L6-v2":           384,
            "all-mpnet-base-v2":          768,
            "paraphrase-MiniLM-L6-v2":    384,
            "multi-qa-MiniLM-L6-cos-v1":  384,
        },
    },
}


def get_embedding_dimensions(provider: str, model: str) -> int | None:
    """
    Returns the output vector dimension for a given provider + model.
    Useful when initialising a vector store that requires knowing the
    dimension upfront (e.g. Pinecone, Chroma, FAISS).

    Example:
        dims = get_embedding_dimensions("OpenAI", "text-embedding-3-small")
        # → 1536
    """
    cfg = EMBEDDING_PROVIDERS.get(provider)
    if cfg:
        return cfg.get("dimensions", {}).get(model)
    return None


def build_embeddings(provider: str, model: str, api_key: str | None = None):
    """
    ============================================================
    CONCEPT: Embedding Model Factory
    ============================================================
    Instantiates and returns the correct LangChain embedding object
    for the given provider and model.

    All embedding classes share the same interface:
      .embed_query(text: str)         → list[float]  (one vector)
      .embed_documents(texts: list)   → list[list[float]]  (many vectors)

    This means you can swap providers without changing any downstream
    code — the vector store, the retriever, and the RAG chain all
    work identically regardless of which embedding model produced the vectors.

    PARAMETERS:
      provider  : provider name string matching a key in EMBEDDING_PROVIDERS
      model     : model name string from that provider's models list
      api_key   : the user's API key from session_state.
                  None is valid for HuggingFace (no key required).

    RETURNS:
      A LangChain Embeddings object ready to call .embed_query() or
      .embed_documents() on.
    """
    pkg_map = {
        "OpenAI":              "langchain-openai",
        "Google Gemini":       "langchain-google-genai",
        "Cohere":              "langchain-cohere",
        "Mistral":             "langchain-mistralai",
        "HuggingFace (Local)": "langchain-huggingface sentence-transformers",
    }

    try:
        if provider == "OpenAI":
            # OpenAIEmbeddings wraps the /v1/embeddings endpoint.
            # The model parameter selects which embedding model to use.
            # api_key is passed explicitly — never read from os.environ.
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model, api_key=api_key)

        elif provider == "Google Gemini":
            # GoogleGenerativeAIEmbeddings wraps the Gemini embedding API.
            # task_type can be set to "retrieval_document" or "retrieval_query"
            # for better retrieval performance — defaults to general embedding.
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

        elif provider == "Cohere":
            # CohereEmbeddings supports both English and multilingual models.
            # input_type distinguishes search queries from documents —
            # Cohere uses this to optimise the embedding direction.
            # "search_document" → use when embedding docs to store
            # "search_query"    → use when embedding a user's search query
            # Default here uses "search_document" (most common use case).
            from langchain_cohere import CohereEmbeddings
            return CohereEmbeddings(
                model=model,
                cohere_api_key=api_key,
                input_type="search_document",
            )

        elif provider == "Mistral":
            # MistralAIEmbeddings wraps Mistral's single embedding endpoint.
            from langchain_mistralai import MistralAIEmbeddings
            return MistralAIEmbeddings(model=model, api_key=api_key)

        elif provider == "HuggingFace (Local)":
            # HuggingFaceEmbeddings runs the model LOCALLY using the
            # sentence-transformers library. No API call is made.
            # First run downloads the model (~90MB for MiniLM) and caches it.
            # Subsequent runs load from cache — no internet needed.
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model)

        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    except ImportError:
        pkg = pkg_map.get(provider, "the relevant langchain package")
        st.error(
            f"Missing package for **{provider}** embeddings. Install it with:\n"
            f"```\npip install {pkg}\n```"
        )
        st.stop()


def get_or_set_embedding_key() -> tuple[str | None, str, str]:
    """
    ============================================================
    CONCEPT: Embedding Auth — per-user, session-isolated
    ============================================================
    Same security model as get_or_set_api_key() in utils.py:
      - Keys stored ONLY in st.session_state (per browser tab)
      - os.environ is NEVER read or written
      - Every user must provide their own key

    Special case — HuggingFace (Local):
      No API key is required. The function returns (None, provider, model)
      and skips the key input field entirely.

    Uses separate session_state keys ("embed_*") so a user can have
    BOTH a chat model key AND an embedding model key active at the
    same time without conflict.

    Returns:
        (api_key, provider, model)
        api_key is None for HuggingFace (Local).
    """

    # ── 1. Already resolved this session (fast path) ────────────────────────
    # Uses "embed_*" keys to avoid colliding with chat model keys
    # stored as "api_key", "provider", "model" in utils.py.
    if all(k in st.session_state for k in ("embed_api_key", "embed_provider", "embed_model")):
        return (
            st.session_state["embed_api_key"],
            st.session_state["embed_provider"],
            st.session_state["embed_model"],
        )

    # ── 2. Show embedding auth UI ────────────────────────────────────────────
    st.title("🔢 Embedding Model Configuration")
    st.markdown("Choose an embedding provider and model to vectorise your text.")
    st.info(
        "🔒 Your API key is stored only in your browser session and is never "
        "shared with other users. It clears automatically when you close the tab."
    )

    col1, col2 = st.columns(2)

    with col1:
        chosen_provider = st.selectbox(
            "Embedding Provider",
            list(EMBEDDING_PROVIDERS.keys()),
            key="embed_provider_select",
        )

    cfg = EMBEDDING_PROVIDERS[chosen_provider]

    with col2:
        chosen_model = st.selectbox(
            "Embedding Model",
            cfg["models"],
            index=cfg["models"].index(cfg["default_model"]),
            key="embed_model_select",
        )

    # Show the output dimension as a helpful info badge
    dims = cfg["dimensions"].get(chosen_model)
    if dims:
        st.caption(f"📐 Output dimensions: **{dims}** — configure your vector store to match this.")

    # ── HuggingFace: skip key input entirely ──────────────────────────────────
    if chosen_provider == "HuggingFace (Local)":
        st.info(
            "✅ No API key needed — this model runs locally on your machine using "
            "sentence-transformers. The model will be downloaded on first use (~90MB for MiniLM)."
        )
        if st.button("✅ Confirm & Continue", use_container_width=True, key="embed_confirm"):
            # Store None as the key — build_embeddings() handles the None case
            st.session_state["embed_api_key"]  = None
            st.session_state["embed_provider"] = chosen_provider
            st.session_state["embed_model"]    = chosen_model
            st.rerun()
    else:
        placeholder = f"{cfg['key_prefix']}..." if cfg["key_prefix"] else "Paste your API key here"
        user_key = st.text_input(
            f"{chosen_provider} API Key",
            type="password",
            placeholder=placeholder,
            help="Your key is stored only in your session and cleared when you close the tab.",
            key="embed_key_input",
        )

        if st.button("✅ Confirm & Continue", use_container_width=True, key="embed_confirm"):
            if not user_key.strip():
                st.error("API key cannot be empty.")
            else:
                # ✅ Store ONLY in session_state — never in os.environ
                st.session_state["embed_api_key"]  = user_key.strip()
                st.session_state["embed_provider"] = chosen_provider
                st.session_state["embed_model"]    = chosen_model
                st.rerun()

    st.stop()


# ── Convenience helper ────────────────────────────────────────────────────────
def get_embeddings_from_session() -> object:
    """
    One-liner shortcut for apps that just want a ready-to-use
    embeddings object without managing the 3-tuple themselves.

    Usage:
        from embeddings_utils import get_embeddings_from_session
        embeddings = get_embeddings_from_session()
        vector = embeddings.embed_query("hello world")
    """
    api_key, provider, model = get_or_set_embedding_key()
    return build_embeddings(provider, model, api_key)
