# LangChain Expression Language (LCEL) being used to build a pipeline
import os
import io # NEW: For handling audio in-memory
from gtts import gTTS # NEW: Google Text-to-Speech

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from utils import get_or_set_openai_key # Import the shared function

#from langchain_core.globals import set_debug
#set_debug(True) #Enable Debug to see whats going behind the scene.
# 1. Call the function. It handles the logic, the UI prompt, and the blocking automatically!
api_key = get_or_set_openai_key()
# 2. If the code reaches this line, you are 100% guaranteed to have the key.
st.success("API Key is ready!")
st.write("Your secret key starts with:", api_key[:7])

llm=ChatOpenAI(model="gpt-4o-mini",api_key=api_key)

# --- App Setup ---
st.set_page_config(page_title="Text To Speech", layout="centered", page_icon="💬")
title_prompt= PromptTemplate(
    input_variables=["topic"],
    template ="""You are an experienced speech writer.
    You need to craft an impactful title for a speech
    on the following topic: {topic}
    Answer exactly with one title.
    """
    )

speech_prompt= PromptTemplate(
    input_variables=["title"],
    template ="""You need to write a powerful speech of 350 words
    for the following title: {title}
    """
    )
# The pipe operator (|) in LangChain takes the output of the component on the left and passes it as the input to the component on the right.
# If you were to invoke this chain by passing {"topic": "Machine Learning"}, here is how the data flows:
# title_prompt: Takes your dictionary {"topic": "Machine Learning"} and injects it into the template, outputting a formatted string:
# "You are an experienced speech writer... topic: Machine Learning..."
# llm: Takes that formatted string, runs it through your model (like the local Ollama instance), and outputs an AI Message object.
# StrOutputParser(): Takes the raw AI Message object and extracts just the plain text string (e.g., "The Future of Algorithms").
# lambda title: ...: This is where the output string lands.

# The last step—(lambda title: (st.write(title),title)[1])—is a very common pattern when marrying LangChain with Streamlit.
# LangChain chains are designed to pass data straight through to the end. But in Streamlit, you often want to display intermediate steps to the UI while the chain is running.
# Here is what that lambda is actually doing:
# It takes the parsed string (title).
# It creates a tuple containing two items: (Item 1, Item 2).
# Item 1: st.write(title) executes the Streamlit function to print the title to your web app UI. The return value of st.write() is None.
# Item 2: title is just the string itself.
# So, the tuple evaluates to (None, "The Future of Algorithms").
# Finally, the [1] at the end grabs the second item in that tuple (index 1), which is the title string, and passes it forward.

## first_chain generates the title, prints it to UI, and passes the title string forward
first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])

# Added the StrOutputParser here to output a clean string instead of an AIMessage object
second_chain = speech_prompt | llm | StrOutputParser()

final_chain = first_chain | second_chain

# --- 3. Streamlit UI ---
st.title("🎤 AI Speech Writer & Dictator")
st.markdown("Generate a speech and listen to it instantly.")

topic = st.text_input("Enter a topic:")
# How the New Code Works
# io.BytesIO(): This creates a temporary file-like object in your computer's RAM.
# Think of it as a virtual .mp3 file that only exists while the app is running. This keeps your local directory clean of hundreds of generated audio files.
#
# tts.write_to_fp(audio_buffer): Instead of using tts.save("speech.mp3"), which writes to your hard drive,
# this command tells gTTS to write the audio data directly into that in-memory buffer.
#
# st.audio(): This is Streamlit's native audio widget. It accepts file paths, URLs, or—in our case—raw bytes from memory.
# It automatically renders a sleek play/pause/download bar in your UI.
if topic:
    with st.spinner("Writing your speech..."):
        # The chain executes. The lambda function prints the title,
        # and the final string lands in 'speech_text'
        speech_text = final_chain.invoke({"topic": topic})
        st.write(speech_text)

    # --- NEW: Audio Generation Block ---
    with st.spinner("Converting to audio..."):
        # 1. Initialize gTTS with the text
        tts = gTTS(text=speech_text, lang='en', slow=False)

        # 2. Create an in-memory buffer
        audio_buffer = io.BytesIO()

        # 3. Write the audio directly to the buffer instead of a file
        tts.write_to_fp(audio_buffer)

        # 4. Render the Streamlit audio player
        st.audio(audio_buffer, format='audio/mp3')