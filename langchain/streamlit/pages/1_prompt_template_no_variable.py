# ============================================================
# CONCEPT: PromptTemplate with a Single Variable
# ============================================================
# PromptTemplate is LangChain's way of creating reusable,
# structured instructions for an LLM.
#
# KEY IDEA: Instead of hardcoding a question each time, you
# define a template with {placeholders} and fill them in at
# runtime. This separates your instructions from your data.
#
# HOW IT WORKS:
#   1. Define the template string with {variable} placeholders
#   2. Declare which variables the template expects via input_variables
#   3. Call .format(variable=value) to produce the final prompt string
#   4. Pass that string to llm.invoke() to get a response
#
# THIS FILE demonstrates the simplest case: one variable {country}
# ============================================================

import streamlit as st
from langchain_core.prompts import PromptTemplate
from utils import get_or_set_api_key, build_llm  # Shared key + LLM factory

from dotenv import load_dotenv, find_dotenv
# load_dotenv() reads your .env file and injects variables like
# OPENAI_API_KEY into os.environ so they're available app-wide.
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="Prompt Template With No Variable", layout="centered", page_icon="🎈")

# ── LLM Setup ────────────────────────────────────────────────
# get_or_set_api_key() checks (in order):
#   1. Streamlit session state (already entered this session)
#   2. Environment variables (.env file)
#   3. Streamlit UI prompt (asks the user to paste a key)
# build_llm() uses the provider + model info to return the
# correct LangChain chat model (OpenAI, Anthropic, Groq, etc.)
api_key, provider, model = get_or_set_api_key()
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── PromptTemplate Definition ─────────────────────────────────
# input_variables: declares all {placeholders} used in the template.
#   LangChain validates that you supply all of them at .format() time.
# template: the instruction string sent to the LLM.
#   {country} will be replaced with the actual value at runtime.
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

# ── UI + Invocation ───────────────────────────────────────────
country = st.text_input("Enter the country:")

if country:
    # .format(country=country) fills the {country} placeholder,
    # producing a plain string that the LLM can understand.
    # llm.invoke() sends that string to the model and returns
    # an AIMessage object. We extract the text via .content
    response = llm.invoke(prompt_template.format(country=country))
    st.write(response.content)
