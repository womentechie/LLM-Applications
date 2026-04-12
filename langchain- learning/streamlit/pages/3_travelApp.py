# ============================================================
# CONCEPT: Real-world Multi-variable PromptTemplate (Travel App)
# ============================================================
# This file is a practical extension of File 2.
# It shows how a PromptTemplate with four variables can power
# a complete, user-facing application.
#
# NEW CONCEPT — st.selectbox():
#   Some variables shouldn't be free-text. Constraining "budget"
#   to ["Low", "Medium", "High"] means the LLM always receives
#   a predictable, well-defined value — leading to better outputs.
#   Mix free-text inputs (st.text_input) with constrained ones
#   (st.selectbox) based on what makes sense for each variable.
#
# PROMPT ENGINEERING TIP:
#   Notice the template uses a numbered list format. Structuring
#   the prompt this way guides the LLM to return a structured,
#   scannable response rather than a wall of prose.
# ============================================================

import streamlit as st
from langchain_core.prompts import PromptTemplate
from utils import get_or_set_api_key, build_llm

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="Travel App", layout="centered", page_icon="🚢")

# ── LLM Setup ────────────────────────────────────────────────
api_key, provider, model = get_or_set_api_key()
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── Four-variable PromptTemplate ──────────────────────────────
# {city}     — destination
# {month}    — time of year (affects weather/events advice)
# {language} — local language for useful phrases section
# {budget}   — Low / Medium / High (constrained by selectbox)
prompt_template = PromptTemplate(
    input_variables=["city", "month", "language", "budget"],
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

# ── UI Inputs ─────────────────────────────────────────────────
# st.text_input  → open-ended values the user types freely
# st.selectbox   → constrained values from a predefined list
#   Using a selectbox for budget ensures the LLM always receives
#   "Low", "Medium", or "High" — never unexpected values like
#   "cheap" or "broke" that could confuse the model.
city = st.text_input("Enter the city:")
month = st.text_input("Enter the month:")
language = st.text_input("Enter the language:")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])

# Guard: only invoke if ALL required fields are filled.
# Without this check, .format() would inject empty strings
# into the template, producing poor or confusing LLM responses.
if city and month and language and budget:
    response = llm.invoke(prompt_template.format(
        city=city,
        month=month,
        language=language,
        budget=budget
    ))
    st.write(response.content)
