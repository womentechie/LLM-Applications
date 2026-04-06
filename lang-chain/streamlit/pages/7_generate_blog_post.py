import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # CRITICAL IMPORT
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from utils import get_or_set_openai_key # Import the shared function

# 1. Call the function. It handles the logic, the UI prompt, and the blocking automatically!
api_key = get_or_set_openai_key()
# 2. If the code reaches this line, you are 100% guaranteed to have the key.
st.success("API Key is ready!")
st.write("Your secret key starts with:", api_key[:7])

llm=ChatOpenAI(model="gpt-4o-mini",api_key=api_key)
# --- App Setup ---
st.set_page_config(page_title="Generate Blog Post", layout="centered", page_icon="👩‍💻")
topic_prompt= PromptTemplate(
    input_variables=["topic"],
    template ="""You are a professional blogger.
    Create an outline for a blog post on the following topic: {topic}
    The outline should include:
        - Introduction
        - 3 main points with subpoints
        - Conclusion: {topic}
    """
    )

#  Added {tone} to inject emotion into the writing
blog_prompt = PromptTemplate(
    input_variables=["outline", "tone"],
    template="""You are a professional blogger.
    Write an engaging introduction paragraph based on the following outline:
    {outline}

    CRITICAL INSTRUCTION: The tone and emotion of this introduction MUST be highly {tone}. 
    Use vocabulary, pacing, and phrasing that strongly reflects this emotion.
    """
)
# --- Chains ---
# 1. first_chain outputs the raw outline string
first_chain = topic_prompt | llm | StrOutputParser() | (lambda title: (st.write(title), title)[1])

# 2. second_chain (Added StrOutputParser here so you don't have to use .content later!)
second_chain = blog_prompt | llm | StrOutputParser()

# 3. THE FIX: final_chain uses assign() to prevent the "tone" variable from being deleted
final_chain = RunnablePassthrough.assign(outline=first_chain) | second_chain

# --- UI ---
st.title("Blog Post Generator")
topic = st.text_input("Enter a topic ")

tone = st.selectbox("Select Emotion/Tone", [
    "Inspiring and Passionate",
    "Empathetic and Motivational",
    "Heartfelt and Uplifting",
    "Empowering and Sincere",
    "Hopeful and Enthusiastic"
])

if topic:
    # Because of assign(), both topic and tone survive the journey!
    response = final_chain.invoke({"topic": topic, "tone": tone})

    # Your updated UI code
    st.subheader(f"Generated Intro ({tone})")
    st.write(response)  # No need for .content because we added StrOutputParser to second_chain