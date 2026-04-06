import streamlit as st

st.set_page_config(page_title="App Setup", page_icon="⚙️")

st.title("Welcome to the AI Application")
st.markdown("Before you begin, please authenticate with your API provider.")

# Initialize the session state variable if it doesn't exist yet
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

# Create the password input field
api_key_input = st.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    value=st.session_state["openai_api_key"], # Keeps the field populated if they return to this page
    placeholder="sk-proj-...",
    help="Your key is stored locally in your browser's session state and is cleared when you close the tab."
)

# Update the session state when the user provides a key
if api_key_input:
    st.session_state["openai_api_key"] = api_key_input
    st.success("Key saved securely to session! You can now navigate to other pages using the sidebar.")
else:
    st.warning("Please provide an API key to unlock the app features.")
