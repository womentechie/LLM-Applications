import os
import streamlit as st
from faster_whisper import WhisperModel
from langchain_core.prompts import ChatPromptTemplate
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Import shared utilities ───────────────────────────────────
from utils import get_or_set_api_key, build_llm

# ── App Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Audio Q&A", layout="centered", page_icon="🎙️")
st.title("🎙️ Audio Transcription and Query Assistant")
st.markdown(
    "Upload an audio file and ask any question about it. "
    "Powered by an LLM."
)

# ── Step 1: Resolve chat model key ───────────────────────────
chat_key, chat_provider, chat_model = get_or_set_api_key()
llm = build_llm(chat_provider, chat_model, chat_key)

st.success(f"💬 Using: **{chat_provider}** · `{chat_model}`")
st.divider()

# ── Load Whisper model ONCE (important for performance) ──────
@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="int8")  # CPU-friendly

model = load_model()

# ── Transcription ────────────────────────────────────────────
def get_transcription(audio_path):
    segments, info = model.transcribe(audio_path)

    transcription = ""
    for segment in segments:
        transcription += segment.text + " "

    return transcription.strip()

# ── LLM Processing ───────────────────────────────────────────
def process_transcription(transcription, query):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You clearly explain and analyze audio transcriptions."),
            ("human", "{input}")
        ]
    )

    chain = prompt | llm

    response = chain.invoke({
        "input": f"Transcription:\n{transcription}\n\nQuery: {query}"
    })

    return response.content

# ── File Upload UI ───────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file is not None:

    audio_path = "uploaded_audio" + os.path.splitext(uploaded_file.name)[1]

    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    # Transcription with spinner
    with st.spinner("Transcribing audio..."):
        transcription = get_transcription(audio_path)

    st.subheader("📝 Transcription")
    st.write(transcription)

    query = st.text_input("Ask a question about the audio:")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Thinking..."):
                answer = process_transcription(transcription, query)

            st.subheader("💡 Answer")
            st.write(answer)
        else:
            st.warning("Please enter a query.")
else:
    st.info("Please upload an audio file.")