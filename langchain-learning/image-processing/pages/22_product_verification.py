# ============================================================
# CONCEPT: Product Nutrition Comparison using Multimodal Vision LLM
# ============================================================
# This app lets users upload two product nutrition fact images and
# ask questions comparing them. The vision LLM reads both images
# simultaneously and answers the question grounded in what it sees.
#
# HOW IT WORKS:
#   1. User uploads two product images (JPG / PNG)
#   2. User types a comparison question
#   3. Both images are base64-encoded and sent in one multimodal prompt
#   4. The vision LLM reads both images together and answers
#
# WHY NO EMBEDDINGS?
#   Images are sent directly to a vision-capable chat model as base64
#   data URIs — no vector store or similarity search is involved.
#   Only the chat key from utils.py is needed here.
#
# FILE LOCATION:
#   image-processing/pages/22_product_comparison.py
# Provide which of the product is more nutritious?
# ============================================================

import base64
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

# ── Import shared utilities ───────────────────────────────────
# utils → resolves provider/model/key, builds the chat model object.
# No embeddings_utils needed — this is a direct vision call, not RAG.
from utils import get_or_set_api_key, build_llm


# ── Helper: encode uploaded image to base64 ───────────────────
def encode_image(image_file) -> str:
    """
    Read an uploaded file and return it as a base64-encoded string.

    IMPORTANT: .read() consumes the file buffer in one call.
    Store the result — do not call encode_image() twice on the same
    UploadedFile or the second call will return an empty string.

    Args:
        image_file: a Streamlit UploadedFile object

    Returns:
        str — base64-encoded image content
    """
    return base64.b64encode(image_file.read()).decode()


# ── App Setup ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Nutrition Comparison",
    layout="centered",
    page_icon="🥗",
)
st.title("🥗 Product Nutrition Comparison")
st.markdown(
    "Upload two product nutrition labels and ask any comparison question. "
    "The AI reads both labels simultaneously and answers based on what it sees."
)

# ── Step 1: Resolve chat model key ────────────────────────────
# get_or_set_api_key() checks st.session_state first (fast path),
# then shows the provider/model/key UI if not yet entered.
# Keys are stored ONLY in session_state — never in os.environ.
#
# NOTE: Select a vision-capable model in the UI:
#   OpenAI    → gpt-4o ✅, gpt-4-turbo ✅
#   Anthropic → claude-3-5-sonnet-20241022 ✅, claude-3-opus ✅
#   Google    → gemini-2.0-flash ✅, gemini-1.5-pro ✅
chat_key, chat_provider, chat_model = get_or_set_api_key()

# build_llm() returns the correct LangChain chat model object
# for whichever provider + model the user selected.
llm = build_llm(chat_provider, chat_model, chat_key)

st.success(f"💬 Using: **{chat_provider}** · `{chat_model}`")
st.divider()

# ── Step 2: Build the comparison prompt ───────────────────────
# The human turn contains THREE content blocks sent in one API call:
#   1. {"type": "text"}      → user's question        ({question})
#   2. {"type": "image_url"} → product 1 image        ({product_image_1})
#   3. {"type": "image_url"} → product 2 image        ({product_image_2})
#
# "detail": "high" ensures fine nutrition label text (calories,
# serving size, ingredients) is readable by the LLM.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful food inspector and nutrition expert. "
        "You are given two product nutrition label images. "
        "Answer the user's question by comparing both labels accurately. "
        "Always refer to the products as 'Product 1' and 'Product 2' "
        "to match the layout the user sees on screen.",
    ),
    (
        "human",
        [
            {"type": "text",      "text": "{question}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{product_image_1}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{product_image_2}", "detail": "high"}},
        ],
    ),
])

# ── Step 3: Build the chain ───────────────────────────────────
# chain.invoke({"question": ..., "product_image_1": ..., "product_image_2": ...})
# fills all three placeholders, sends the multimodal message to the LLM,
# and returns an AIMessage with .content holding the comparison answer.
chain = prompt | llm

# ── Step 4: Upload + preview — images stay inside their columns ──
# Both file uploaders and their previews live inside the same column
# block so the image always renders directly below its own uploader.
st.subheader("📦 Upload product labels")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Product 1**")
    upload_image_1 = st.file_uploader(
        "Choose product 1",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
        key="product_1",                       # explicit key avoids widget ID collisions
    )
    # Preview renders immediately inside col1 — directly below its uploader
    if upload_image_1 is not None:
        st.image(
            upload_image_1,
            caption="Product 1",
            use_container_width=True,
        )

with col2:
    st.markdown("**Product 2**")
    upload_image_2 = st.file_uploader(
        "Choose product 2",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
        key="product_2",                       # explicit key avoids widget ID collisions
    )
    # Preview renders immediately inside col2 — directly below its uploader
    if upload_image_2 is not None:
        st.image(
            upload_image_2,
            caption="Product 2",
            use_container_width=True,
        )

st.divider()

# ── Step 5: Question input ─────────────────────────────────────
st.subheader("❓ Ask a comparison question")
question = st.text_input(
    "Your question",
    placeholder="e.g. Which product has fewer calories? Which has more protein?",
)

# ── Step 6: Run comparison ─────────────────────────────────────
# Auto-trigger: fires as soon as all three inputs are present.
# No button needed — the user has already done the explicit actions
# of uploading both images and typing a question.
# NOTE: Streamlit reruns on every widget interaction, so this also
# fires when the question text changes mid-sentence. If you want to
# prevent API calls while the user is still typing, swap this back
# to: if st.button("🔍 Compare", use_container_width=True):
if question and upload_image_1 and upload_image_2:
    with st.spinner("Reading both labels and comparing..."):
        # encode_image() consumes the file buffer in one call.
        # Both images are encoded here and stored before invoke()
        # so neither buffer is read twice.
        image_1 = encode_image(upload_image_1)
        image_2 = encode_image(upload_image_2)

        response = chain.invoke({
            "question":        question,
            "product_image_1": image_1,
            "product_image_2": image_2,
        })

    st.markdown("### 💡 Comparison Result")
    st.write(response.content)