# ============================================================
# CONCEPT: KYC Verification using Multimodal Vision LLM
# ============================================================
# This app lets users upload an identification document and
# provide their name and date of birth. The LLM cross-checks
# the supplied details against the document image.
#
# HOW IT WORKS:
#   1. User uploads an ID document image (JPG / PNG)
#   2. User enters their name and date of birth
#   3. Image is base64-encoded and embedded in a multimodal prompt
#   4. The vision LLM receives the image + supplied details together
#   5. The LLM verifies whether the details match the document
#
# WHY NO EMBEDDINGS?
#   The image is sent directly to a vision-capable chat model as a
#   base64 data URI — no vector store or similarity search is involved.
#   Only the chat key from utils.py is needed here.
#
# WHAT CHANGED FROM THE ORIGINAL:
#   Before → Hardcoded OpenAI only, raw os.getenv() key, no provider choice
#   After  → Any vision-capable provider via utils.py, session-safe key,
#             get_or_set_api_key() handles auth UI automatically,
#             fixed broken f-string in image_url,
#             guard added for missing upload before processing
#
# FILE LOCATION:
#   image-processing/pages/21_kyc_verification.py
# ============================================================

import base64
from datetime import date
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

    IMPORTANT: .read() consumes the file buffer in one call.
    Store the result in a variable — do not call encode_image() twice
    on the same UploadedFile or the second call will return an empty string.

    Args:
        image_file: a Streamlit UploadedFile object (has a .read() method)

    Returns:
        str — base64-encoded image content
    """
    return base64.b64encode(image_file.read()).decode()


# ── App Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="KYC Verification", layout="centered", page_icon="🪪")
st.title("🪪 KYC Verification Application")
st.markdown(
    "Upload your identification document and enter your details. "
    "The AI will verify whether the information matches the document."
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

# ── Step 2: Build the KYC verification prompt ─────────────────
# The human turn contains FOUR content blocks sent in one API call:
#   1. {"type": "text"} → verification instruction
#   2. {"type": "text"} → user-supplied name     ({user_name})
#   3. {"type": "text"} → user-supplied DOB      ({user_dob})
#   4. {"type": "image_url"} → the ID document   ({image})
#
# The LLM reads all four blocks together and cross-checks the
# supplied details against what it can see in the document image.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can verify identification documents. "
            "Compare the user-supplied details against the document image and clearly "
            "state whether each field matches, does not match, or cannot be verified.",
        ),
        (
            "human",
            [
                {"type": "text", "text": "Verify the identification details."},
                {"type": "text", "text": "Name: {user_name}"},
                {"type": "text", "text": "DOB: {user_dob}"},
                {
                    "type": "image_url",
                    "image_url": {
                        # data URI format required by OpenAI vision API.
                        # {image} is filled at invoke() time with the
                        # base64 string produced by encode_image().
                        # FIX: was f"data:image/jpeg;base64,""{image}" —
                        # broken f-string concatenation; corrected to a
                        # plain template string that LangChain fills itself.
                        "url": "data:image/jpeg;base64,{image}",
                        "detail": "high",  # "high" recommended for ID docs —
                                           # ensures text (name, DOB) is readable
                    },
                },
            ],
        ),
    ]
)

# ── Step 3: Build the chain ───────────────────────────────────
# The | operator pipes the prompt into the llm.
# chain.invoke({"user_name": ..., "user_dob": ..., "image": ...}) will:
#   1. Fill all prompt template placeholders
#   2. Send the completed multimodal message to the LLM
#   3. Return an AIMessage with .content holding the verification result
chain = prompt | llm

# ── Step 4: UI — document upload + user details ───────────────
st.subheader("📋 Enter your details")

user_name = st.text_input(
    "Full name (as it appears on your ID)",
    placeholder="e.g. Jane Smith",
)
user_dob = st.date_input(
    "Date of birth",
    # FIX: st.date_input defaults min_value to (today - 10 years) and
    # max_value to today, which blocks any date before ~2016.
    # For KYC we need to support adult users born decades ago, so we
    # explicitly set min_value to 1 Jan 1900 and max_value to today.
    min_value=date(1900, 1, 1),
    max_value=date.today(),
    value=date(1990, 1, 1),          # sensible default avoids landing on today
    format="DD/MM/YYYY",             # familiar format for ID documents
    help="Must match the date of birth on your identification document.",
)

st.subheader("📄 Upload identification document")
uploaded_file = st.file_uploader(
    "Choose an image of your ID document",
    type=["jpg", "jpeg", "png","svg"],
    help="Supported formats: JPG, JPEG, PNG,SVG. Ensure the document is fully visible.",
)

# ── Step 5: Preview the uploaded document ────────────────────
# Show the image immediately after upload so the user can confirm
# they uploaded the right document before verification runs.
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded document", use_container_width=True)

# ── Step 6: Run verification ──────────────────────────────────
if st.button("🔍 Verify Document", use_container_width=True):
    # Guard: all three inputs must be present before invoking the chain.
    # Without this check, the chain would receive None or empty values
    # and the LLM would have nothing meaningful to verify against.
    if not uploaded_file:
        st.warning("Please upload an identification document.")
    elif not user_name.strip():
        st.warning("Please enter your full name.")
    else:
        with st.spinner("Analysing document..."):
            # Encode the uploaded file to base64 so the LLM can read it.
            # uploaded_file is a Streamlit UploadedFile — calling .read()
            # consumes the buffer, so encode_image must be called once
            # and the result stored in `image` for the invoke() call.
            # NOTE: st.image() above reads the file for display using a
            # separate internal buffer — it does NOT interfere with .read().
            image = encode_image(uploaded_file)

            # invoke() fills all four placeholders and calls the LLM.
            # user_dob is a datetime.date object — str() converts it to
            # "YYYY-MM-DD" which the LLM can read naturally.
            response = chain.invoke({
                "user_name": user_name,
                "user_dob":  str(user_dob),
                "image":     image,
            })

        st.markdown("### ✅ Verification Result")
        st.write(response.content)