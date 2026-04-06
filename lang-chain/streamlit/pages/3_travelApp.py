import os

import streamlit as st
from langchain_openai import ChatOpenAI
#from langchain_core.globals import set_debug
from langchain_core.prompts import PromptTemplate
from utils import get_or_set_openai_key # Import the shared function


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# 1. Call the function. It handles the logic, the UI prompt, and the blocking automatically!
api_key = get_or_set_openai_key()
# 2. If the code reaches this line, you are 100% guaranteed to have the key.
st.success("API Key is ready!")
st.write("Your secret key starts with:", api_key[:7])

st.set_page_config(page_title="Travel App", layout="centered", page_icon="🚢")
llm=ChatOpenAI(model="gpt-4o",api_key=api_key)

prompt_template = PromptTemplate(
    input_variables=["city","month","language","budget"],
    template="""Welcome to the {city} travel guide!
            If you're visiting in {month}, here's what you can do:
            1. Must-visit attractions.
            2. Local cuisine you must try.
            3. Useful phrases in {language}.
            4. Tips for traveling on a {budget} budget.
            Enjoy your trip!
    """
)

st.title("Travel App")

city=st.text_input("Enter the city : ")
month=st.text_input("Enter the month : ")
language=st.text_input("Enter the language : ")
budget=st.selectbox("Travel Budget", ["Low","Medium","High"])
if city and month and language and budget:
    response=llm.invoke(prompt_template.format(city=city,
                                               month=month,
                                               language=language,
                                               budget=budget
                                               ))
    st.write(response.content)