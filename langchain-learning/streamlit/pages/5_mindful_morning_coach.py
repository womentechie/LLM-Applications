# ============================================================
# CONCEPT: Dynamic Prompts + Streamlit UI Customisation
# ============================================================
# This file combines two ideas:
#
# 1. CONDITIONAL PROMPT BUILDING
#    The PromptTemplate is defined INSIDE the button-click handler.
#    This is valid — you can build prompts anywhere in your code.
#    Here it means the prompt is only created when actually needed,
#    which is clean for apps where the template never changes.
#
# 2. STREAMLIT UI CUSTOMISATION
#    st.markdown() with unsafe_allow_html=True lets you inject raw
#    HTML/CSS into the Streamlit page. This is used here to:
#      - Apply a random gradient background to the whole app
#      - Render a mood-coloured decorative bar based on user selection
#    Use this power carefully — injecting unsanitised user input as
#    HTML is a security risk (XSS). Safe here because the values
#    come from a fixed dict, not raw user text.
#
# 3. st.spinner()
#    Wrapping llm.invoke() in `with st.spinner("...")` shows a
#    loading indicator while the model is thinking. Essential UX
#    for any real app since LLM calls can take several seconds.
#
# 4. st.download_button()
#    Lets the user save the AI's response as a .txt file without
#    any extra server-side code — Streamlit handles it in-browser.
# ============================================================

import streamlit as st
from langchain_core.prompts import PromptTemplate
import random
from utils import get_or_set_api_key, build_llm

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="Mindful Morning Coach", layout="centered", page_icon="🧘")

# ── LLM Setup ────────────────────────────────────────────────
api_key, provider, model = get_or_set_api_key()
llm = build_llm(provider, model, api_key)
st.success(f"✅ Ready — **{provider}** · `{model}`")

# ── Mood → Colour Mapping ─────────────────────────────────────
# A Python dict maps each mood string to a hex colour.
# mood_colors.get(mood, "#FFFFFF") safely returns white as a
# fallback if the mood key isn't found.
mood_colors = {
    "Calm": "#B3E5FC",
    "Stressed": "#FFE0B2",
    "Grateful": "#C8E6C9",
    "Motivated": "#FFF59D",
    "Tired": "#E1BEE7",
    "Reflective": "#FFCCBC",
}

# ── Random Background ─────────────────────────────────────────
# random.choice() picks one gradient from the list each time
# the page loads (i.e., each Streamlit rerun). This gives the
# app a fresh feel without any database or state management.
backgrounds = [
    "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
    "linear-gradient(135deg, #c3cfe2 0%, #c3f0ca 100%)",
    "linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)",
    "linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%)",
]

bg_choice = random.choice(backgrounds)

# Inject CSS into the Streamlit page using a <style> block.
# .stApp is the root Streamlit container class.
# The double {{ }} are Python f-string escapes for literal braces.
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg_choice};
        color: #333333;
    }}
    </style>
    """,
    unsafe_allow_html=True  # Required to render raw HTML/CSS
)

st.title("🧘 Mindful Morning Coach")
st.subheader("Start your day with calm, focus, and intention 🌞")
st.write("Tell me how you feel and what you want to focus on today.")

# ── User Inputs ───────────────────────────────────────────────
# st.selectbox constrains mood to known values so the colour
# lookup and prompt always receive expected inputs.
mood = st.selectbox(
    "💭 How are you feeling right now?",
    ["Calm", "Stressed", "Grateful", "Motivated", "Tired", "Reflective"]
)

goals = st.text_input(
    "🎯 What's your focus or goal for today?",
    placeholder="e.g., stay positive, finish a key task, be present"
)
time_of_day = st.selectbox("⏰ Time of day", ["Morning", "Afternoon", "Evening"])

# ── Mood Colour Bar ───────────────────────────────────────────
# Renders a thin coloured bar matching the selected mood.
# This is pure cosmetic feedback — it updates immediately on
# selectbox change because Streamlit reruns on every interaction.
if mood:
    color = mood_colors.get(mood, "#FFFFFF")
    st.markdown(
        f"<div style='background-color:{color};padding:10px;border-radius:10px;'></div>",
        unsafe_allow_html=True
    )

# ── Button + LLM Call ─────────────────────────────────────────
# st.button() returns True only on the rerun triggered by a click.
# Validation happens before the expensive LLM call to avoid
# wasting API credits on incomplete input.
if st.button("🌸 Get My Mindful Note"):
    if not mood or not goals:
        st.warning("Please select your mood and enter your goal for the day.")
    else:
        with st.spinner("Breathing in calm energy... ✨"):  # Loading indicator

            # Prompt defined here (inside the handler) because it's
            # only needed when the button is clicked.
            prompt_template = PromptTemplate(
                input_variables=["mood", "goals", "time_of_day"],
                template="""
                You are a warm, empathetic mindfulness coach.
                Based on the user's current mood ({mood}), daily goal ({goals}), and time of day ({time_of_day}),
                write a short, inspiring reflection to start their day.

                Include:
                1. A personalized affirmation (1 line)
                2. A short mindfulness reflection (2–3 lines)
                3. A journaling prompt (1 question)

                Use a gentle, encouraging tone. Add fitting emojis.
                """
            )

            formatted_prompt = prompt_template.format(
                mood=mood, goals=goals, time_of_day=time_of_day
            )

            response = llm.invoke(formatted_prompt)

            st.markdown("### 🌤️ Your Mindful Moment")
            st.write(response.content)

            # st.download_button() creates a browser download from
            # an in-memory string — no file needs to be saved on disk.
            st.download_button(
                label="📥 Save Reflection",
                data=response.content,
                file_name=f"mindful_note_{time_of_day.lower()}.txt",
                mime="text/plain"
            )

st.markdown("---")
st.caption("🌼 Created with LangChain + Streamlit")
