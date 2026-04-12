# ============================================================
# CONCEPT: LCEL Pipeline + Text-to-Speech with gTTS
# ============================================================
# This file extends File 4 (LCEL chains) with a new concept:
# processing the LLM's text output FURTHER using a non-LLM tool.
#
# PIPELINE RECAP (from File 4):
#   topic → [title_prompt | llm | StrOutputParser | lambda] → title
#   title → [speech_prompt | llm | StrOutputParser]         → speech text
#
# NEW CONCEPT — gTTS (Google Text-to-Speech):
#   gTTS converts a plain text string into audio (MP3 format).
#   This shows a key pattern: LLM output is just text — you can
#   pipe it into ANY downstream tool (TTS, email, PDF, database...).
#
# IN-MEMORY AUDIO with io.BytesIO():
#   Instead of saving an MP3 file to disk (tts.save("file.mp3")),
#   we write audio bytes directly into RAM using io.BytesIO().
#   Benefits:
#     - No leftover files cluttering the server
#     - Works in cloud environments where disk writes are restricted
#     - st.audio() accepts BytesIO directly — no path needed
#
# KEY DIFFERENCE vs FILE 4:
#   File 4:  second_chain = speech_prompt | llm            (returns AIMessage)
#   File 6:  second_chain = speech_prompt | llm | StrOutputParser()  (returns str)
#   StrOutputParser is added here because gTTS needs a plain string,
#   not an AIMessage object. Always use StrOutputParser when the
#   output leaves LangChain and enters another tool or library.
# ============================================================

import io                       # Standard library: in-memory byte streams
from gtts import gTTS           # Google Text-to-Speech: text → MP3 bytes
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import get_or_set_api_key, build_llm

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="Text To Speech", layout="centered", page_icon="💬")

# ── LLM Setup ────────────────────────────────────────────────
api_key, provider, model = get_or_set_api_key()
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── Prompt 1: Generate a Title ────────────────────────────────
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech
    on the following topic: {topic}
    Answer exactly with one title.
    """
)

# ── Prompt 2: Generate a Speech ───────────────────────────────
speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words
    for the following title: {title}
    """
)

# ── Chain 1: topic → title ────────────────────────────────────
# Lambda taps the pipeline to display the title in the UI,
# then passes the title string on to second_chain.
first_chain = (
    title_prompt
    | llm
    | StrOutputParser()
    | (lambda title: (st.write(title), title)[1])
)

# ── Chain 2: title → speech text (plain string) ───────────────
# StrOutputParser() is essential here — gTTS requires a plain str,
# not a LangChain AIMessage object.
second_chain = speech_prompt | llm | StrOutputParser()

# ── Final Pipeline ────────────────────────────────────────────
# topic → title (shown in UI) → speech text (plain string)
final_chain = first_chain | second_chain

st.title("🎤 AI Speech Writer & Dictator")
st.markdown("Generate a speech and listen to it instantly.")

topic = st.text_input("Enter a topic:")

if topic:
    # ── Step 1: Generate speech text via LLM pipeline ────────
    with st.spinner("Writing your speech..."):
        speech_text = final_chain.invoke({"topic": topic})
        st.write(speech_text)  # Display the written speech

    # ── Step 2: Convert speech text to audio ─────────────────
    with st.spinner("Converting to audio..."):
        # gTTS takes a plain string and language code.
        # slow=False uses normal speaking speed.
        tts = gTTS(text=speech_text, lang='en', slow=False)

        # io.BytesIO() creates a virtual file in RAM.
        # Think of it as an MP3 file that only exists in memory.
        audio_buffer = io.BytesIO()

        # write_to_fp() writes MP3 bytes into the buffer
        # instead of saving to disk (avoids tts.save("file.mp3")).
        tts.write_to_fp(audio_buffer)

        # st.audio() renders a play/pause/download audio widget.
        # It accepts BytesIO directly — no file path needed.
        st.audio(audio_buffer, format='audio/mp3')
