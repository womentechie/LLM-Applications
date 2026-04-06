# LangChain Expression Language (LCEL) being used to build a pipeline
import os

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

st.set_page_config(page_title="SimpleChain LCEL", layout="centered", page_icon="⛓️‍💥")
llm=ChatOpenAI(model="gpt-4o-mini",api_key=api_key)

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
second_chain = speech_prompt | llm

#second_chain = speech_prompt | llm | StrOutputParser() # Without StrOutputParse,  response is an AIMessage object (and not a raw string), you extract the actual speech.
final_chain = first_chain | second_chain

st.title("Speech Generator")
topic = st.text_input("Enter a topic ")

if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content) # No need for .content anymore! as we added StrOutputParser against second chain
