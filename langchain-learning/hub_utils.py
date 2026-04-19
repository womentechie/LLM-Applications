# hub_utils.py
# ============================================================
# Root-level credential resolver for the LangChain Learning Hub
# ============================================================
# This is the ONLY file at the repo root that handles imports
# from module-level utils. It uses explicit file paths via
# importlib so there is zero ambiguity about which utils.py
# or embeddings_utils.py is being loaded — regardless of
# sys.path ordering or Streamlit Cloud's page execution context.
#
# WHY NOT USE ROOT-LEVEL utils.py / embeddings_utils.py STUBS?
#   Having files at the root with the same name as files inside
#   the modules creates confusion — which one gets imported?
#   It also means two files to keep in sync whenever the real
#   implementation changes. This single file with a unique name
#   has no naming conflict and is the only place to maintain.
#
# WHY IMPORTLIB INSTEAD OF sys.path?
#   sys.path injection in main-app.py persists within a single
#   Streamlit page run, but Streamlit Cloud may execute pages in
#   fresh contexts where that injected path is gone. importlib
#   loads from an absolute file path — it always works regardless
#   of sys.path state.
#
# USAGE (in main-app.py only):
#   from hub_utils import get_embedding_key, get_chat_key
#   embed_key, embed_provider, embed_model = get_embedding_key()
#   chat_key,  chat_provider,  chat_model  = get_chat_key()
#
# Individual pages do NOT import from here — they import from
# their own module's utils.py (resolved via sys.path injection
# which is reliable within a single page execution).
# ============================================================

import os
import importlib.util

# Absolute path to the repo root (the folder this file lives in)
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_module(relative_path: str, module_name: str):
    """
    Load a Python module from an explicit file path.

    This bypasses sys.path entirely — the module is found by its
    absolute path on disk, not by searching any import path list.
    Safe to call from any execution context.

    Args:
        relative_path: path to the .py file relative to repo root
                       e.g. "embeddings/embeddings_utils.py"
        module_name:   unique name to register in sys.modules
                       e.g. "hub.embeddings_utils"

    Returns:
        The loaded module object.
    """
    abs_path = os.path.join(_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Load the real implementations once at import time ─────────
# embeddings/embeddings_utils.py → embedding key + model factory
# embeddings/utils.py            → chat key + LLM factory
#
# Registered under unique names ("hub.*") so they never collide
# with the module-level imports that pages do independently.
_emb_utils  = _load_module("embeddings/embeddings_utils.py", "hub.embeddings_utils")
_chat_utils = _load_module("embeddings/utils.py",            "hub.chat_utils")


# ── Public API ────────────────────────────────────────────────
# main-app.py calls these two functions and nothing else.
# They delegate to the real get_or_set_*() functions which handle
# session_state caching, auth UI, and provider selection.

def get_embedding_key() -> tuple[str | None, str, str]:
    """
    Resolve the embedding model credentials.

    Delegates to embeddings/embeddings_utils.py::get_or_set_embedding_key().
    Returns (api_key, provider, model) — same signature as the original.
    Credentials are cached in st.session_state after the first call.
    """
    return _emb_utils.get_or_set_embedding_key()


def get_chat_key() -> tuple[str, str, str]:
    """
    Resolve the chat/vision model credentials.

    Delegates to embeddings/utils.py::get_or_set_api_key().
    Returns (api_key, provider, model) — same signature as the original.
    Credentials are cached in st.session_state after the first call.
    """
    return _chat_utils.get_or_set_api_key()