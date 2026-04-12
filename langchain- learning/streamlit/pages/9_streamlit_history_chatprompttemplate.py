# ============================================================
# CONCEPT: Conversational Memory with RunnableWithMessageHistory
# ============================================================
# All previous files had NO memory — each llm.invoke() was a
# fresh call with zero awareness of prior messages. That works
# for single-turn Q&A, but breaks for a real conversation.
#
# THIS FILE adds memory, turning a stateless chain into a
# proper back-and-forth chatbot.
#
# THREE NEW COMPONENTS:
#
# 1. MessagesPlaceholder
#    A special slot inside ChatPromptTemplate that gets filled
#    with the full conversation history at invocation time.
#    Without this, there's nowhere in the prompt to inject
#    past messages — the LLM would have amnesia every turn.
#
# 2. StreamlitChatMessageHistory
#    Stores the conversation history inside Streamlit's
#    session_state (browser tab memory). It is:
#      - Ephemeral: wiped when the user refreshes the page
#      - Isolated:  each browser tab gets its own history
#      - Automatic: LangChain reads from and writes to it
#                   without you managing the list manually
#
# 3. RunnableWithMessageHistory
#    A wrapper that intercepts every chain call and:
#      a) Fetches the stored history for the given session_id
#      b) Injects it into the MessagesPlaceholder slot
#      c) Runs the chain (prompt → llm)
#      d) Appends both the user message AND the AI reply
#         back into the history store automatically
#
# DATA FLOW PER TURN:
#   User types question
#       ↓
#   RunnableWithMessageHistory fetches history from session_state
#       ↓
#   Injects history into MessagesPlaceholder in the prompt
#       ↓
#   Full prompt (system + history + new question) → LLM
#       ↓
#   AI reply → shown in UI
#       ↓
#   Both user message + AI reply saved back to session_state
#       ↓
#   Next turn starts with updated history
#
# SESSION ID:
#   RunnableWithMessageHistory supports multiple concurrent users
#   by isolating history per session_id. Here we use the dummy
#   string "abc123" because StreamlitChatMessageHistory already
#   isolates by browser tab — so the ID is ignored in practice.
#   In a multi-user production app (e.g., with a database backend),
#   you would use a real unique ID per user.
# ============================================================

import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils import get_or_set_api_key, build_llm  # Shared key + LLM factory

from dotenv import load_dotenv, find_dotenv
# load_dotenv() reads .env so OPENAI_API_KEY / ANTHROPIC_API_KEY etc.
# are available without hardcoding secrets in source code.
_ = load_dotenv(find_dotenv())

# ── App Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Science Coach", layout="centered", page_icon="🔬")

# ── 1. Resolve provider, model, and API key ───────────────────
# get_or_set_api_key() checks session_state → .env → UI prompt.
# Returns a 3-tuple so the app knows which LLM class to instantiate.
api_key, provider, model = get_or_set_api_key()

# ── 2. Build the LLM ─────────────────────────────────────────
# build_llm() is the provider-agnostic factory in utils.py.
# Swap keys and it transparently switches between OpenAI,
# Anthropic, Groq, Gemini, etc. — no code changes needed here.
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── 3. Prompt Template with Memory Slot ──────────────────────
# ChatPromptTemplate.from_messages() builds a role-aware prompt.
#
# Message 1 — "system":
#   The AI's persona and guardrail. Applied once per conversation.
#   Restricts the coach to Science topics only.
#
# Message 2 — MessagesPlaceholder(variable_name="chat_history"):
#   This is the MEMORY SLOT. At invocation time, LangChain
#   automatically replaces this placeholder with the full list
#   of past HumanMessage and AIMessage objects from the history store.
#   The LLM sees all previous turns as if they were in the prompt.
#
# Message 3 — "human" with {input}:
#   The current user question, filled at invoke() time.
#   Always placed AFTER history so the LLM sees past context first.
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Science coach. Answer only Science-related questions. "
            "If a question is not related to Science, say: I don't know.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),  # ← history injected here
        ("human", "{input}"),                               # ← current question
    ]
)

# ── 4. Base Chain (no memory yet) ────────────────────────────
# A standard LCEL pipeline: prompt → llm.
# This chain is stateless on its own — memory is added in step 6.
chain = prompt_template | llm

# ── 5. History Store ─────────────────────────────────────────
# StreamlitChatMessageHistory stores messages in st.session_state.
# It behaves like a list of HumanMessage / AIMessage objects.
# Characteristics:
#   - Lives only for the browser session (ephemeral)
#   - Automatically namespaced per tab (no cross-tab leakage)
#   - Compatible with MessagesPlaceholder out of the box
history_for_chain = StreamlitChatMessageHistory()

# ── 6. Wrap Chain with Memory Manager ────────────────────────
# RunnableWithMessageHistory transforms the stateless `chain`
# into a stateful conversational chain.
#
# Arguments explained:
#   chain                      → the pipeline to wrap
#   lambda session_id: ...     → function that returns the history store
#                                for a given session. LangChain always
#                                calls this with a session_id, but
#                                StreamlitChatMessageHistory ignores it
#                                (tab isolation handles it instead).
#   input_messages_key="input" → tells LangChain which key in the
#                                invoke() dict holds the user's message
#   history_messages_key="chat_history" → must match the variable_name
#                                in MessagesPlaceholder above
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,  # returns history store for this session
    input_messages_key="input",            # maps to ("human", "{input}") in the prompt
    history_messages_key="chat_history",   # maps to MessagesPlaceholder above
)

# ── 7. UI ─────────────────────────────────────────────────────
st.title("🔬 Science Coach")

user_input = st.text_input("Enter your question")

if user_input:
    # invoke() triggers the full memory-aware pipeline:
    #   1. Fetches history from StreamlitChatMessageHistory
    #   2. Injects it into MessagesPlaceholder
    #   3. Appends {input} as the new human turn
    #   4. Sends everything to the LLM
    #   5. Saves the new exchange back to history automatically
    #
    # configurable session_id: required by LangChain's interface.
    # "abc123" is a dummy — StreamlitChatMessageHistory doesn't
    # use it, but the argument must be present.
    response = chain_with_history.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "abc123"}},
    )
    st.write(response.content)  # .content extracts text from AIMessage

# ── 8. Debug Panel: Raw History ───────────────────────────────
# Displays the raw StreamlitChatMessageHistory object so you can
# verify that turns are being recorded correctly.
# Each entry alternates: HumanMessage → AIMessage → HumanMessage ...
# Remove this block in production — it's a development aid only.
st.write("HISTORY")
st.write(history_for_chain)
