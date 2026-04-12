# ============================================================
# CONCEPT: RunnablePassthrough — Keeping Variables Alive in Chains
# ============================================================
# This file solves a critical problem in multi-step LCEL chains:
# VARIABLE LOSS.
#
# THE PROBLEM:
#   In a simple pipeline:  A | B | C
#   Each step receives ONLY the output of the previous step.
#   So if step A produces "outline text", step B gets "outline text"
#   — but any OTHER variables (like "tone") are silently discarded.
#
#   Example of the bug:
#     final_chain = first_chain | second_chain
#     final_chain.invoke({"topic": "AI", "tone": "Inspiring"})
#     → first_chain outputs the outline string
#     → second_chain receives ONLY the outline string
#     → {tone} is gone! KeyError when blog_prompt tries to use it.
#
# THE FIX — RunnablePassthrough.assign():
#   RunnablePassthrough passes the ENTIRE current input dict forward,
#   and .assign(key=chain) ADDS a new key to that dict using the
#   output of the given chain.
#
#   So instead of:
#     input_dict → first_chain → outline_string → second_chain
#   We get:
#     input_dict → assign(outline=first_chain) → {topic, tone, outline}
#                                                          ↓
#                                                    second_chain (has all 3!)
#
# MENTAL MODEL:
#   RunnablePassthrough is like a "carry-all" bag.
#   Instead of handing just the new output to the next step,
#   it hands the whole bag PLUS the new output together.
# ============================================================

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough  # The key import!
from utils import get_or_set_api_key, build_llm

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="Generate Blog Post", layout="centered", page_icon="👩‍💻")

# ── LLM Setup ────────────────────────────────────────────────
api_key, provider, model = get_or_set_api_key()
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── Prompt 1: Generate an Outline ────────────────────────────
# Receives {topic} from the initial invoke() call.
# Outputs a structured blog outline as plain text.
topic_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are a professional blogger.
    Create an outline for a blog post on the following topic: {topic}
    The outline should include:
        - Introduction
        - 3 main points with subpoints
        - Conclusion: {topic}
    """
)

# ── Prompt 2: Write the Introduction ─────────────────────────
# Needs BOTH {outline} (from step 1) AND {tone} (from the original
# invoke() call). Without RunnablePassthrough, {tone} would be lost.
blog_prompt = PromptTemplate(
    input_variables=["outline", "tone"],
    template="""You are a professional blogger.
    Write an engaging introduction paragraph based on the following outline:
    {outline}

    CRITICAL INSTRUCTION: The tone and emotion of this introduction MUST be highly {tone}.
    Use vocabulary, pacing, and phrasing that strongly reflects this emotion.
    """
)

# ── Chain 1: topic → outline (displayed in UI, passed forward) ─
# StrOutputParser() converts AIMessage → plain string for display
# and for use as the {outline} variable in blog_prompt.
first_chain = (
    topic_prompt
    | llm
    | StrOutputParser()
    | (lambda outline: (st.write(outline), outline)[1])  # Show outline in UI
)

# ── Chain 2: {outline, tone} → introduction paragraph ─────────
second_chain = blog_prompt | llm | StrOutputParser()

# ── Final Chain: RunnablePassthrough.assign() ─────────────────
# Step-by-step of what happens when invoke({"topic":..., "tone":...}) runs:
#
#   1. RunnablePassthrough receives: {"topic": "AI", "tone": "Inspiring"}
#   2. .assign(outline=first_chain) runs first_chain with that dict,
#      producing the outline string, then ADDS it to the dict:
#      → {"topic": "AI", "tone": "Inspiring", "outline": "...outline text..."}
#   3. second_chain (blog_prompt) receives all three keys — no KeyError!
final_chain = RunnablePassthrough.assign(outline=first_chain) | second_chain

# ── UI ────────────────────────────────────────────────────────
st.title("Blog Post Generator")
topic = st.text_input("Enter a topic:")

# st.selectbox constrains tone to known values, giving the LLM
# clear, consistent instructions about the emotional register.
tone = st.selectbox("Select Emotion/Tone", [
    "Inspiring and Passionate",
    "Empathetic and Motivational",
    "Heartfelt and Uplifting",
    "Empowering and Sincere",
    "Hopeful and Enthusiastic"
])

if topic:
    # Both "topic" AND "tone" are passed in.
    # Thanks to RunnablePassthrough, "tone" survives all the way
    # to blog_prompt even though first_chain doesn't use it.
    response = final_chain.invoke({"topic": topic, "tone": tone})
    st.subheader(f"Generated Intro ({tone})")
    st.write(response)
