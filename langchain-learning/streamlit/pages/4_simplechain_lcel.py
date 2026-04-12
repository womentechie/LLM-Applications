# ============================================================
# CONCEPT: LangChain Expression Language (LCEL) — Simple Chain
# ============================================================
# LCEL is LangChain's pipeline syntax. It uses the pipe operator
# (|) to connect components so the output of one flows directly
# into the input of the next — similar to Unix pipes.
#
# PIPELINE ANATOMY:
#   component_A | component_B | component_C
#   └─ output of A becomes input of B
#                └─ output of B becomes input of C
#
# COMPONENTS USED HERE:
#   PromptTemplate  → formats your template into a prompt string
#   llm             → sends the prompt to the model, returns AIMessage
#   StrOutputParser → extracts the plain text string from AIMessage
#   lambda          → custom Python function inserted mid-pipeline
#
# WHY TWO CHAINED PROMPTS?
#   Step 1 (first_chain):  topic  → generate a TITLE
#   Step 2 (second_chain): title  → generate a full SPEECH
#   This is called "prompt chaining" — breaking a complex task
#   into smaller, focused steps produces better results than one
#   giant prompt trying to do everything at once.
#
# THE LAMBDA TRICK:
#   LangChain chains pass data forward automatically. But sometimes
#   you want to display an intermediate result in the UI without
#   breaking the pipeline. The lambda does this:
#       lambda title: (st.write(title), title)[1]
#   - st.write(title) → displays the title in Streamlit (side effect)
#   - [1] extracts `title` from the tuple and passes it forward
#   This lets you "tap" the pipeline mid-flow to show progress.
# ============================================================

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import get_or_set_api_key, build_llm

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="SimpleChain LCEL", layout="centered", page_icon="⛓️")

# ── LLM Setup ────────────────────────────────────────────────
api_key, provider, model = get_or_set_api_key()
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── Prompt 1: Generate a Title ────────────────────────────────
# Narrow, focused prompt → ask for exactly ONE thing (a title).
# Constraining the output ("Answer exactly with one title") makes
# the LLM's response easier to use as input for the next step.
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech
    on the following topic: {topic}
    Answer exactly with one title.
    """
)

# ── Prompt 2: Generate a Speech from the Title ────────────────
# This prompt receives the OUTPUT of prompt 1 as its input.
# The {title} placeholder will be filled by the pipeline automatically.
speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words
    for the following title: {title}
    """
)

# ── Chain 1: topic → title (with UI display side-effect) ──────
# Data flow:
#   title_prompt   → formats {topic} into a full prompt string
#   | llm          → sends to model, returns AIMessage object
#   | StrOutputParser() → extracts plain text string from AIMessage
#   | lambda       → displays title in UI AND passes the string forward
first_chain = (
    title_prompt
    | llm
    | StrOutputParser()
    | (lambda title: (st.write(title), title)[1])  # tap to display, then pass on
)

# ── Chain 2: title → speech ───────────────────────────────────
# Note: NO StrOutputParser here — response is an AIMessage object.
# We call .content at the end (see st.write(response.content) below).
second_chain = speech_prompt | llm

# ── Final Chain: combines both steps end-to-end ───────────────
# When final_chain.invoke({"topic": topic}) is called:
#   1. first_chain runs:  topic  → title string (also shown in UI)
#   2. second_chain runs: title string → AIMessage with full speech
final_chain = first_chain | second_chain

st.title("Speech Generator")
topic = st.text_input("Enter a topic:")

if topic:
    # invoke() triggers the entire pipeline from left to right.
    # The dict {"topic": topic} provides the initial input.
    response = final_chain.invoke({"topic": topic})
    st.write(response.content)  # .content extracts text from AIMessage
