# utils.py
# pip install langchain-openai langchain-anthropic langchain-google-genai langchain-cohere langchain-mistralai langchain-groq
import os
import streamlit as st

# ── Provider registry ──────────────────────────────────────────────────────
# Each entry defines:
#   key_prefix   : used to auto-detect provider from a pasted key (None = can't auto-detect)
#   env_var      : environment variable name to check
#   models       : ordered list of available models shown in the dropdown
#   default_model: pre-selected model in the dropdown

PROVIDERS = {
    "OpenAI": {
        "key_prefix":    "sk-",
        "env_var":       "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "default_model": "gpt-4o-mini",
    },
    "Anthropic": {
        "key_prefix":    "sk-ant-",
        "env_var":       "ANTHROPIC_API_KEY",
        "models": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ],
        "default_model": "claude-sonnet-4-5",
    },
    "Google Gemini": {
        "key_prefix":    "AIza",
        "env_var":       "GOOGLE_API_KEY",
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        "default_model": "gemini-2.0-flash",
    },
    "Cohere": {
        "key_prefix":    None,   # No consistent prefix
        "env_var":       "COHERE_API_KEY",
        "models": [
            "command-r-plus",
            "command-r",
            "command",
        ],
        "default_model": "command-r",
    },
    "Mistral": {
        "key_prefix":    None,   # No consistent prefix
        "env_var":       "MISTRAL_API_KEY",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
        ],
        "default_model": "mistral-small-latest",
    },
    "Groq": {
        "key_prefix":    "gsk_",
        "env_var":       "GROQ_API_KEY",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "default_model": "llama-3.3-70b-versatile",
    },
}


def _auto_detect_provider(api_key: str) -> str | None:
    """
    Return provider name if the key prefix matches a known provider, else None.

    IMPORTANT: Sort by prefix length descending before checking.
    'sk-ant-' (Anthropic) and 'sk-' (OpenAI) both match an Anthropic key
    if checked in the wrong order — longest prefix must win.
    e.g. 'sk-ant-abc' → startswith('sk-ant-') ✅  startswith('sk-') ✅
    Checking longest first ensures 'sk-ant-' is matched correctly.
    """
    # Build list of (name, prefix) sorted by prefix length, longest first
    sorted_providers = sorted(
        [(name, cfg["key_prefix"]) for name, cfg in PROVIDERS.items() if cfg["key_prefix"]],
        key=lambda x: len(x[1]),
        reverse=True  # longest prefix first → 'sk-ant-' before 'sk-'
    )
    for name, prefix in sorted_providers:
        if api_key.startswith(prefix):
            return name
    return None


def build_llm(provider: str, model: str, api_key: str):
    """
    Instantiate and return the correct LangChain chat model object.
    Shows a helpful pip install error if the required package is missing.
    """
    pkg_map = {
        "OpenAI":        "langchain-openai",
        "Anthropic":     "langchain-anthropic",
        "Google Gemini": "langchain-google-genai",
        "Cohere":        "langchain-cohere",
        "Mistral":       "langchain-mistralai",
        "Groq":          "langchain-groq",
    }

    try:
        if provider == "OpenAI":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, api_key=api_key)

        elif provider == "Anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, api_key=api_key)

        elif provider == "Google Gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

        elif provider == "Cohere":
            from langchain_cohere import ChatCohere
            return ChatCohere(model=model, cohere_api_key=api_key)

        elif provider == "Mistral":
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(model=model, api_key=api_key)

        elif provider == "Groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    except ImportError:
        pkg = pkg_map.get(provider, "the relevant langchain package")
        st.error(f"Missing package for **{provider}**. Install it with:\n```\npip install {pkg}\n```")
        st.stop()


def get_or_set_api_key_old() -> tuple[str, str, str]:
    """
    Resolves provider, model, and API key via (in priority order):
      1. Streamlit session state  — already validated this session
      2. Environment variables    — checked against PROVIDERS registry
      3. Streamlit UI             — provider dropdown → model dropdown → key input

    Returns:
        (api_key, provider, model)
    """

    # ── 1. Already resolved this session ────────────────────────────────────
    if all(k in st.session_state for k in ("api_key", "provider", "model")):
        return (
            st.session_state["api_key"],
            st.session_state["provider"],
            st.session_state["model"],
        )

    # ── 2. Check environment variables ──────────────────────────────────────
    for provider_name, cfg in PROVIDERS.items():
        env_key = os.getenv(cfg["env_var"])
        if env_key:
            st.session_state["api_key"]  = env_key
            st.session_state["provider"] = provider_name
            st.session_state["model"]    = cfg["default_model"]
            return env_key, provider_name, cfg["default_model"]

    # ── 3. Streamlit UI ──────────────────────────────────────────────────────
    st.title("🤖 LLM Configuration")
    st.markdown("Choose a provider and model, then enter your API key to continue.")

    col1, col2 = st.columns(2)

    with col1:
        chosen_provider = st.selectbox("LLM Provider", list(PROVIDERS.keys()))

    cfg = PROVIDERS[chosen_provider]

    with col2:
        chosen_model = st.selectbox(
            "Model",
            cfg["models"],
            index=cfg["models"].index(cfg["default_model"]),
        )

    placeholder = f"{cfg['key_prefix']}..." if cfg["key_prefix"] else "Paste your API key here"
    user_key = st.text_input(
        f"{chosen_provider} API Key",
        type="password",
        placeholder=placeholder,
        help=f"Set `{cfg['env_var']}` in your `.env` file to skip this step.",
    )

    st.caption(f"💡 Tip: Add `{cfg['env_var']}=your_key` to your `.env` file to skip this screen.")

    if st.button("✅ Confirm & Continue", use_container_width=True):
        if not user_key.strip():
            st.error("API key cannot be empty.")
        else:
            # Warn if the key prefix doesn't match the selected provider
            detected = _auto_detect_provider(user_key.strip())
            if detected and detected != chosen_provider:
                st.warning(
                    f"⚠️ This key looks like a **{detected}** key, "
                    f"but you selected **{chosen_provider}**. Please double-check."
                )
            else:
                st.session_state["api_key"]  = user_key.strip()
                st.session_state["provider"] = chosen_provider
                st.session_state["model"]    = chosen_model
                os.environ[cfg["env_var"]]   = user_key.strip()
                st.rerun()

    st.stop()  # Block the rest of the app until confirmed


def get_or_set_api_key() -> tuple[str, str, str]:
    """
    Resolves provider, model, and API key — always per user, per session.

    !! DEPLOYMENT SECURITY — WHY os.environ IS NOT USED !!
    -------------------------------------------------------
    os.environ is PROCESS-LEVEL memory shared across every concurrent
    user on the same server. Using it here would cause two critical bugs:

      1. KEY LEAKAGE — reading os.environ after one user sets it would
         hand that user's private API key to every subsequent user who
         opens the app, even in a different browser tab.

      2. KEY POLLUTION — writing a user's key to os.environ would persist
         it for the lifetime of the server process, not just the session.

    st.session_state is the correct storage — it is isolated per browser
    tab and automatically cleared when the tab is closed.

    Priority order:
      1. st.session_state — already entered this session (fast path)
      2. Streamlit UI     — user enters key fresh via the auth screen

    Returns:
        (api_key, provider, model)
    """

    # ── 1. Already resolved this session (fast path) ────────────────────────
    # session_state is scoped to a single browser tab — reading from it
    # here is safe and never touches another user's data.
    if all(k in st.session_state for k in ("api_key", "provider", "model")):
        return (
            st.session_state["api_key"],
            st.session_state["provider"],
            st.session_state["model"],
        )

    # ── 2. Show auth UI — every user must enter their own key ────────────────
    st.title("🤖 LLM Configuration")
    st.markdown("Choose a provider and model, then enter your API key to continue.")
    st.info(
        "🔒 Your API key is stored only in your browser session and is never "
        "shared with other users. It clears automatically when you close the tab."
    )

    col1, col2 = st.columns(2)

    with col1:
        chosen_provider = st.selectbox("LLM Provider", list(PROVIDERS.keys()))

    cfg = PROVIDERS[chosen_provider]

    with col2:
        chosen_model = st.selectbox(
            "Model",
            cfg["models"],
            index=cfg["models"].index(cfg["default_model"]),
        )

    placeholder = f"{cfg['key_prefix']}..." if cfg["key_prefix"] else "Paste your API key here"
    user_key = st.text_input(
        f"{chosen_provider} API Key",
        type="password",
        placeholder=placeholder,
        help="Your key is stored only in your session and cleared when you close the tab.",
    )

    if st.button("✅ Confirm & Continue", use_container_width=True):
        if not user_key.strip():
            st.error("API key cannot be empty.")
        else:
            detected = _auto_detect_provider(user_key.strip())
            if detected and detected != chosen_provider:
                st.warning(
                    f"⚠️ This key looks like a **{detected}** key, "
                    f"but you selected **{chosen_provider}**. Please double-check."
                )
            else:
                # ✅ Store ONLY in session_state — isolated to this user's tab.
                # ❌ Never write to os.environ — that would leak to all users.
                st.session_state["api_key"] = user_key.strip()
                st.session_state["provider"] = chosen_provider
                st.session_state["model"] = chosen_model
                st.rerun()

    st.stop()  # Block the rest of the app until the user confirms their key

# Any file still calling get_or_set_openai_key() will continue to work.
def get_or_set_openai_key() -> str:
    api_key, _, _ = get_or_set_api_key()
    return api_key