# ============================================================
# CONCEPT: Multimodal Vision — Describing Images with an LLM
# ============================================================
# This app lets users upload an image and ask questions about it.
# The image is encoded as a base64 string and sent directly to a
# vision-capable chat model (e.g. GPT-4o) alongside the question.
#
# WHY NO EMBEDDINGS?
#   Embeddings are used to convert TEXT into vectors for similarity
#   search against a document store (RAG). This app skips that step
#   entirely — the image is passed directly to the LLM as part of
#   the prompt payload, not stored or retrieved from a vector DB.
#   Only a chat model (build_llm) is needed here.
#
# HOW VISION INPUTS WORK IN LANGCHAIN:
#   OpenAI's vision API accepts images as base64-encoded strings
#   embedded in the message content under the "image_url" type.
#   The "detail" field controls resolution:
#     "low"  → faster, cheaper, good for general description
#     "high" → slower, more accurate for fine detail / text in images
#
# WHAT CHANGED FROM THE ORIGINAL:
#   Before → Hardcoded OpenAI only, raw os.getenv() key, no provider choice
#   After  → Any vision-capable provider via utils.py, session-safe key,
#             get_or_set_api_key() handles auth UI automatically
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

    OpenAI's vision API does not accept raw bytes — it expects the
    image to be base64-encoded and embedded in the message as a
    data URI: `data:image/jpeg;base64,<encoded_string>`.

    Args:
        image_file: a Streamlit UploadedFile object (has a .read() method)

    Returns:
        str — base64-encoded image content
    """
    return base64.b64encode(image_file.read()).decode()


# ── App Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Image Q&A", layout="centered", page_icon="🖼️")
st.title("🖼️ Image Q&A")
st.markdown(
    "Upload an image and ask any question about it. "
    "Powered by a vision-capable LLM."
)

# ── Step 1: Resolve chat model key ───────────────────────────
# get_or_set_api_key() checks st.session_state first (fast path),
# then shows the provider/model/key UI if not yet entered.
# Keys are stored ONLY in session_state — never in os.environ.
#
# NOTE: Make sure to select a vision-capable model in the UI.
#   OpenAI   → gpt-4o, gpt-4-turbo  ✅
#   Anthropic→ claude-3-5-sonnet, claude-3-opus  ✅
#   Others   → check provider docs for vision support
chat_key, chat_provider, chat_model = get_or_set_api_key()

# build_llm() returns the correct LangChain chat model object
# for whichever provider + model the user selected.
llm = build_llm(chat_provider, chat_model, chat_key)

st.success(f"💬 Using: **{chat_provider}** · `{chat_model}`")
st.divider()

# ── Step 2: Build the vision prompt ──────────────────────────
# ChatPromptTemplate with a multimodal human message.
# The human turn contains TWO content blocks:
#   1. {"type": "text"}      → the user's question  ({input})
#   2. {"type": "image_url"} → the base64 image      ({image})
#
# LangChain passes this list structure directly to the provider's
# chat completions API, which handles the multimodal encoding.



prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can describe images."),
        ("human",[
                {"type": "text", "text": "{input}"},
                {
                    "type": "image_url",
                    "image_url": {
                        # data URI format required by OpenAI vision API.
                        # The {image} placeholder is filled at invoke() time
                        # with the base64 string from encode_image().
                        "url": "data:image/jpeg;base64,{image}",
                        "detail": "low",  # "low" = faster/cheaper; "high" = fine detail
                    },
                },
            ],
        ),
    ]
)

# ── Step 3: Build the chain ───────────────────────────────────
# The | operator pipes the prompt into the llm.
# chain.invoke({"input": ..., "image": ...}) will:
#   1. Fill the prompt template placeholders
#   2. Send the completed multimodal message to the LLM
#   3. Return an AIMessage with .content holding the answer
chain = prompt | llm

# ── Step 4: UI — file upload + question ──────────────────────
uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

question = st.text_input(
    "Ask a question about the image",
    placeholder="e.g. What objects are in this image? What colour is the car?",
)

# ── Step 5: Run the chain on submit ──────────────────────────
if question:
    # Guard: ensure an image has actually been uploaded before running.
    # Without this check, encode_image(None) would raise an AttributeError.
    if not uploaded_file:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Analysing image..."):
            # Encode the uploaded file to base64 so the LLM can read it.
            # uploaded_file is a Streamlit UploadedFile — calling .read()
            # consumes the buffer, so encode_image must be called once here
            # and the result stored in `image` for the invoke call.
            image = encode_image(uploaded_file)

            # invoke() fills both placeholders and calls the LLM.
            # response is an AIMessage; .content holds the reply string.
            response = chain.invoke({"input": question, "image": image})

        st.markdown("### 💡 Answer")
        st.write(response.content)