# ============================================================
# CONCEPT: Text Similarity using Embeddings
# ============================================================
# This app demonstrates how embedding models convert text into
# vectors and how we can measure semantic similarity between them.
#
# WHAT CHANGED FROM THE ORIGINAL:
#   Before: Hardcoded to OpenAI only, read key from os.environ
#   After:  Works with any of the 6 supported embedding providers
#           (OpenAI, Google Gemini, Cohere, Mistral, HuggingFace)
#           Key comes from st.session_state — never os.environ
#
# HOW SIMILARITY IS MEASURED — Cosine Similarity:
#   Most embedding providers return normalized vectors (length = 1).
#   For normalized vectors, the dot product of two vectors equals
#   their cosine similarity — a measure of the angle between them.
#
#   Score interpretation:
#     ~1.00 = identical meaning    ("dog" vs "canine")
#     ~0.70 = related meaning      ("king" vs "queen")
#     ~0.30 = loosely related      ("apple" vs "fruit")
#     ~0.00 = unrelated            ("cat" vs "democracy")
#    negative = opposite meaning   (rare in practice)
#
# PROVIDER NOTE:
#   Anthropic and Groq do NOT support embeddings.
#   Anthropic's Claude is a chat-only model.
#   Groq is an inference engine, not an embedding provider.
#   Both are excluded from the EMBEDDING_PROVIDERS registry
#   in embeddings_utils.py.
# ============================================================

import numpy as np          # For dot product (cosine similarity calculation)
import streamlit as st

# Import the shared embedding factory — works for all providers
# get_or_set_embedding_key() : resolves provider + model + key via UI
# build_embeddings()         : returns the correct LangChain embedding object
from embeddings_utils import get_or_set_embedding_key, build_embeddings, get_embedding_dimensions

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# ── App Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Text Similarity", layout="centered", page_icon="🔢")
st.title("🔢 Text Similarity using Embeddings")
st.markdown(
    "Enter two pieces of text and see how semantically similar they are "
    "based on their embedding vectors."
)

# ── Embedding Model Setup ─────────────────────────────────────
# get_or_set_embedding_key() checks (in order):
#   1. st.session_state — already entered this session (fast path)
#   2. Streamlit UI     — shows provider + model dropdown + key input
#
# Returns (api_key, provider, model) — all stored per user, per tab.
# os.environ is NEVER read or written — prevents key leakage between users.
api_key, provider, model = get_or_set_embedding_key()

# build_embeddings() instantiates the correct LangChain embeddings class:
#   OpenAI        → OpenAIEmbeddings(model=model, api_key=api_key)
#   Google Gemini → GoogleGenerativeAIEmbeddings(...)
#   Cohere        → CohereEmbeddings(...)
#   Mistral       → MistralAIEmbeddings(...)
#   HuggingFace   → HuggingFaceEmbeddings(model_name=model)  ← no key needed
embeddings = build_embeddings(provider, model, api_key)

# Show active provider + model + vector dimensions
dims = get_embedding_dimensions(provider, model)
st.success(f"✅ Ready — **{provider}** · `{model}`")
if dims:
    st.caption(f"📐 This model produces **{dims}-dimensional** vectors.")

st.divider()

# ── User Inputs ───────────────────────────────────────────────
# Using st.text_area instead of input() because:
#   - input() is a terminal function — it blocks Streamlit's event loop
#   - st.text_area works natively in the browser with multi-line support
col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area(
        "📝 Text 1",
        placeholder="e.g. The cat sat on the mat",
        height=120,
    )

with col2:
    text2 = st.text_area(
        "📝 Text 2",
        placeholder="e.g. A feline rested on a rug",
        height=120,
    )

# ── Compute Similarity ────────────────────────────────────────
if st.button("🔍 Compare Texts", use_container_width=True):
    if not text1.strip() or not text2.strip():
        st.warning("Please enter text in both fields before comparing.")
    else:
        with st.spinner(f"Embedding with {provider} · {model}..."):

            # embed_query() converts a single string into a vector (list of floats).
            # Each call makes one API request (or local inference for HuggingFace).
            vector1 = embeddings.embed_query(text1)
            vector2 = embeddings.embed_query(text2)

        # ── Cosine Similarity Calculation ─────────────────────
        # Most providers return L2-normalized vectors (magnitude = 1.0).
        # For normalized vectors: cosine_similarity = dot_product(a, b)
        # because: cos(θ) = (a · b) / (|a| × |b|) and |a| = |b| = 1
        #
        # np.dot() computes the sum of element-wise products:
        #   dot(a, b) = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
        similarity_score = np.dot(vector1, vector2)

        # For providers that don't normalize (rare), compute true cosine:
        # similarity_score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        similarity_pct = similarity_score * 100

        # ── Interpret the score ────────────────────────────────
        if similarity_pct >= 90:
            label, color = "Nearly identical meaning", "🟢"
        elif similarity_pct >= 70:
            label, color = "Highly similar", "🟢"
        elif similarity_pct >= 50:
            label, color = "Moderately similar", "🟡"
        elif similarity_pct >= 30:
            label, color = "Loosely related", "🟠"
        else:
            label, color = "Unrelated or opposite", "🔴"

        # ── Display Results ────────────────────────────────────
        st.divider()
        st.subheader("📊 Results")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Raw score", f"{similarity_score:.4f}")
        with m2:
            st.metric("Similarity %", f"{similarity_pct:.2f}%")
        with m3:
            st.metric("Interpretation", f"{color} {label}")

        # Visual similarity bar
        st.progress(
            min(max(similarity_score, 0.0), 1.0),
            text=f"{similarity_pct:.1f}% similar"
        )

        # ── Debug: show raw vectors (first 5 values) ──────────
        with st.expander("🔬 Show raw vectors (first 10 values)"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Vector 1")
                st.write([round(v, 6) for v in vector1[:10]])
            with col_b:
                st.caption("Vector 2")
                st.write([round(v, 6) for v in vector2[:10]])
            st.caption(
                f"Full vectors: {len(vector1)} dimensions each. "
                f"Showing first 10 of {len(vector1)}."
            )