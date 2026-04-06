import os

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from utils import get_or_set_openai_key # Import the shared function


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
#from langchain_core.globals import set_debug
#set_debug(True)

# 1. Call the function. It handles the logic, the UI prompt, and the blocking automatically!
api_key = get_or_set_openai_key()
# 2. If the code reaches this line, you are 100% guaranteed to have the key.
st.success("API Key is ready!")
st.write("Your secret key starts with:", api_key[:7])

st.set_page_config(page_title="Prompt Template With No Variable", layout="centered", page_icon="🎈")
llm=ChatOpenAI(model="gpt-4o",api_key=api_key)
prompt_template = PromptTemplate(
    input_variables=["country"],
    template="""You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    """
)

st.title("Cuisine App")

country=st.text_input("Enter the country : ")
if country:
    response=llm.invoke(prompt_template.format(country=country))
    st.write(response.content)
