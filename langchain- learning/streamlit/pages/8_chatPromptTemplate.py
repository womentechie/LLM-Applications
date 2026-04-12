# ============================================================
# CONCEPT: ChatPromptTemplate — Role-based Conversation Prompts
# ============================================================
# ChatPromptTemplate is different from PromptTemplate (Files 1–3).
#
# PromptTemplate      → produces a single plain text string
# ChatPromptTemplate  → produces a LIST of role-tagged messages
#
# WHY ROLES MATTER:
#   Modern LLMs (GPT, Claude, Gemini, etc.) are trained on
#   conversations with distinct speakers. By labelling each
#   message with a role, you give the model crucial context:
#
#   "system"  → Background instructions / persona for the AI.
#               Set here once; applies to the whole conversation.
#               Think of it as the AI's job description.
#
#   "human"   → A message FROM the user TO the AI.
#               {input} will be replaced with the actual question.
#
#   "ai"      → A message FROM the AI (a pre-written response).
#               Used here as a fixed greeting/sign-off that primes
#               the model to always start from a helpful, warm tone.
#               Also called a "few-shot example" — showing the model
#               an example of the kind of response you want.
#
# HOW from_messages() WORKS:
#   Takes a list of (role, content) tuples and builds a structured
#   prompt that gets sent to the model as a conversation history.
#   The LLM sees these as if they were a real prior exchange,
#   which shapes its response style and scope.
#
# GUARDRAIL PATTERN:
#   The system prompt here includes "Don't answer any question
#   outside of the Agility Coach context!" — this is a common
#   guardrail technique. It reduces (but doesn't guarantee) that
#   the model stays on-topic. Combine with output validation for
#   production apps.
# ============================================================

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate  # Role-aware prompt builder
from utils import get_or_set_api_key, build_llm

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="Agility Coach", layout="centered", page_icon="🏃")

# ── LLM Setup ────────────────────────────────────────────────
api_key, provider, model = get_or_set_api_key()
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── ChatPromptTemplate ────────────────────────────────────────
# from_messages() accepts a list of (role, content) tuples.
# Roles available: "system", "human", "ai" (or "assistant")
#
# Message 1 — "system":
#   Sets the AI's persona and hard constraints.
#   The model treats this as its core identity for the session.
#
# Message 2 — "human":
#   The actual user question. {input} is the only placeholder
#   here — it will be filled when chain.invoke({"input": ...}) runs.
#
# Message 3 — "ai":
#   A hard-coded AI response injected into the conversation history.
#   This primes the model: it "sees" that it already gave a warm
#   greeting, so it's more likely to continue in that register.
#   This technique is called a "few-shot" or "priming" message.
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an Agile coach. You can answer any questions related to Agile process. "
            "Don't answer any question outside of the Agility Coach context!"
        ),
        (
            "human",
            "{input}"  # Filled at invoke() time with the user's question
        ),
        (
            "ai",
            # Pre-written AI turn — primes the model's tone and persona.
            # The model will "continue" the conversation from this point.
            "Thank you so much for connecting with your Agility Coach!"
        ),
    ]
)

st.title("Agile Coach")
user_input = st.text_input("Enter your question:")

# ── LCEL Chain: prompt → llm ──────────────────────────────────
# ChatPromptTemplate formats the messages list.
# llm receives the formatted messages and returns an AIMessage.
# No StrOutputParser here — we use .content to extract text below.
chain = prompt_template | llm

if user_input:
    # invoke() substitutes {input} with the user's question,
    # builds the full message list, sends it to the LLM,
    # and returns an AIMessage object.
    response = chain.invoke({"input": user_input})
    st.write(response.content)  # .content extracts the reply text
