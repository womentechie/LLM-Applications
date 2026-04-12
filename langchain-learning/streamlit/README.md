# 🤖 LangChain Demo Hub

A collection of LangChain + Streamlit demos covering prompt templates, LCEL chains, chat memory, and more — all wired to a **provider-agnostic API key system** that supports OpenAI, Anthropic, Google Gemini, Cohere, Mistral, and Groq out of the box.

---

## 📁 Project Structure

```
app.py                                        # Entry point — run this
utils.py                                      # Shared key auth + LLM factory
requirements.txt                              # All dependencies
pages/
  home.py                                     # Landing page
  1_prompt_template_no_variable.py            # Cuisine App (single variable)
  2_prompt_template_with_variable.py          # Cuisine App (multiple variables)
  3_travelApp.py                              # Travel Guide
  4_simplechain_lcel.py                       # Speech Generator (LCEL chain)
  5_mindful_morning_coach.py                  # Mindful Morning Coach
  6_TextToSpeech.py                           # AI Speech Writer + Audio
  7_generate_blog_post.py                     # Blog Post Generator
  8_chatPromptTemplate.py                     # Agile Coach (chat roles)
  9_streamlit_history_chatprompttemplate.py   # Science Coach (with memory)
```

---

## 🚀 Getting Started

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd langchain-demo-hub
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Install only the providers you need, or install all — the app handles missing packages gracefully with a helpful error message.

### 3. Set your API key (optional but recommended)

Add your key to a `.env` file in the project root:

```env
# Use whichever provider you have a key for — only one is needed
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
COHERE_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...
```

> If no `.env` key is found, the app will prompt you to paste one in the UI on first launch.

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🔑 API Key System

The key system in `utils.py` works as follows (in priority order):

| Priority | Source | Details |
|----------|--------|---------|
| 1st | `st.session_state` | Already entered this session — no re-prompt |
| 2nd | `.env` file | Auto-detected from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. |
| 3rd | Streamlit UI | Provider + model dropdown + key input shown on first launch |

Once resolved, the key is shared across **all pages** via `session_state` — you authenticate once and all demos just work.

---

## 🌐 Supported Providers

| Provider | Key Prefix | Models (default) |
|----------|-----------|-----------------|
| OpenAI | `sk-` | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | `sk-ant-` | claude-opus-4-5, claude-sonnet-4-5, claude-3-5-sonnet, claude-3-haiku |
| Google Gemini | `AIza` | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |
| Cohere | _(none)_ | command-r-plus, command-r, command |
| Mistral | _(none)_ | mistral-large, mistral-small, open-mistral-7b |
| Groq | `gsk_` | llama-3.3-70b, llama-3.1-8b, mixtral-8x7b, gemma2-9b |

---

## 📚 Demo Pages & Concepts

### 📝 Prompt Templates

| Page | Concept |
|------|---------|
| 🎈 Single Variable | `PromptTemplate` with one `{placeholder}`, `.format()`, `llm.invoke()` |
| ✨ Multiple Variables | Multiple `{placeholders}`, mixed input types (`text_input`, `number_input`) |
| 🚢 Travel App | Real-world template, `st.selectbox` for constrained inputs, input guards |

### ⛓️ LCEL Chains

| Page | Concept |
|------|---------|
| ⛓️ Simple Chain | LCEL `\|` pipe operator, two chained prompts, `StrOutputParser`, lambda tap |
| 👩‍💻 Blog Post Generator | `RunnablePassthrough.assign()` to keep variables alive across chain steps |

### 🤖 Chat & Memory

| Page | Concept |
|------|---------|
| 🏃 Agile Coach | `ChatPromptTemplate`, system/human/ai roles, priming messages, guardrails |
| 🔬 Science Coach | `MessagesPlaceholder`, `StreamlitChatMessageHistory`, `RunnableWithMessageHistory` |

### 🛠️ Advanced

| Page | Concept |
|------|---------|
| 🧘 Mindful Morning Coach | Conditional prompt building, `st.spinner`, `st.download_button`, CSS injection |
| 💬 Text To Speech | LLM output → gTTS, `io.BytesIO` in-memory audio, `st.audio` widget |

---

## ➕ Adding a New Provider

Edit the `PROVIDERS` dict in `utils.py` — no other file needs to change:

```python
"MyProvider": {
    "key_prefix":    "mp-",           # Key prefix for auto-detection (or None)
    "env_var":       "MYPROVIDER_API_KEY",
    "models":        ["model-a", "model-b"],
    "default_model": "model-a",
},
```

Then add the LangChain instantiation inside `build_llm()` in `utils.py`.

---

## ➕ Adding a New Demo Page

1. Create `pages/my_new_app.py`
2. Start the file with:
    ```python
    import streamlit as st
    from utils import get_or_set_api_key, build_llm

    api_key, provider, model = get_or_set_api_key()
    llm = build_llm(provider, model, api_key)
    ```
3. Register it in `app.py` under the appropriate section:
    ```python
    st.Page("pages/my_new_app.py", title="My New App", icon="🆕")
    ```

---

## 📦 Dependencies

```
streamlit
python-dotenv
langchain
langchain-core
langchain-community
langchain-openai
langchain-anthropic
langchain-google-genai
langchain-cohere
langchain-mistralai
langchain-groq
gtts                  # Required for Text To Speech demo only
```

---

## 🗒️ Notes

- **gTTS** (`pip install gtts`) is only required for the Text To Speech demo.
- Each demo page has detailed inline comments explaining the LangChain concepts it demonstrates — they're designed to be read as learning material.
- The `HISTORY` debug panel at the bottom of the Science Coach page is intentional — remove it in production.
- `session_state` memory is ephemeral: it clears on browser refresh. For persistent memory, swap `StreamlitChatMessageHistory` for a database-backed store like `RedisChatMessageHistory`.
