# ============================================================
# CONCEPT: PromptTemplate with Multiple Variables
# ============================================================
# Building on File 1, this file shows how to inject MULTIPLE
# dynamic values into a single template.
#
# WHY MULTIPLE VARIABLES?
#   Real-world prompts rarely have just one input. A travel
#   guide needs a city AND a month. A recipe app needs
#   ingredients AND dietary restrictions. Multiple variables
#   let you build rich, personalised prompts from simple inputs.
#
# HOW IT WORKS:
#   - List every {placeholder} name in input_variables
#   - Call .format(var1=val1, var2=val2, ...) to fill them all
#   - The order you list them doesn't matter — they're matched by name
#
# NEW vs FILE 1:
#   File 1 had one variable  → {country}
#   This file has three      → {country}, {no_of_paras}, {language}
# ============================================================

import streamlit as st
from langchain_core.prompts import PromptTemplate
from utils import get_or_set_api_key, build_llm

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="Prompt Template With Variable", layout="centered", page_icon="✨")

# ── LLM Setup ────────────────────────────────────────────────
# Same pattern as File 1 — provider/model/key resolved once,
# then reused for every invocation in this session.
api_key, provider, model = get_or_set_api_key()
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── Multi-variable PromptTemplate ────────────────────────────
# Three placeholders in one template:
#   {country}     — which cuisine to describe
#   {no_of_paras} — controls response length
#   {language}    — controls response language
# This makes one template serve many different use-cases
# without rewriting the core instructions each time.
prompt_template = PromptTemplate(
    input_variables=["country", "no_of_paras", "language"],
    template="""You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {no_of_paras} short paras in {language}.
    """
)

st.title("Cuisine App")

# ── UI Inputs ─────────────────────────────────────────────────
# Each widget maps directly to one template variable.
# st.number_input() returns a number — it gets injected into
# the template as-is; Python converts it to a string automatically.
country = st.text_input("Enter the country:")
no_of_paras = st.number_input("Enter the number of paragraphs:", min_value=1, max_value=5)
language = st.text_input("Enter the language:")

if country:
    # All three variables are filled at once via keyword arguments.
    # If any declared variable were missing here, LangChain would
    # raise a KeyError — helpful for catching bugs early.
    response = llm.invoke(prompt_template.format(
        country=country,
        no_of_paras=no_of_paras,
        language=language
    ))
    st.write(response.content)
