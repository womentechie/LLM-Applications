# utils.py
import os
import streamlit as st

def get_or_set_openai_key():
    """
    Checks OS environment for the API key.
    If missing, prompts the user via Streamlit UI.
    """
    if "openai_api_key" in st.session_state and st.session_state["openai_api_key"]:
        return st.session_state["openai_api_key"]

    os_key = os.getenv("OPENAI_API_KEY")

    if os_key:
        st.session_state["openai_api_key"] = os_key
        return os_key

    st.warning("OpenAI API Key not found.")
    user_input = st.text_input("Please enter your OpenAI API Key:", type="password")

    if user_input:
        st.session_state["openai_api_key"] = user_input
        os.environ["OPENAI_API_KEY"] = user_input
        st.rerun()

    else:
        st.stop()