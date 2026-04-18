# ============================================================
# CONCEPT: Semantic Job Search using Embeddings + ChromaDB
# ============================================================
# This app demonstrates a real-world use case for embeddings:
# searching a list of job postings using MEANING rather than
# exact keyword matching.
#
# WHAT CHANGED FROM THE ORIGINAL:
#   Before → Hardcoded OpenAI only, raw os.environ key, input()
#   After  → Any provider via embeddings_utils, session-safe key,
#             Streamlit UI, pipeline_utils handles loading/chunking
#
# HOW SEMANTIC SEARCH WORKS HERE:
#   1. Load job_listings.txt from data/text/
#   2. Split it into small chunks (one job per chunk ideally)
#   3. Embed every chunk → store vectors in ChromaDB
#   4. User types a query (e.g. "python developer remote")
#   5. Query is embedded → compared against stored vectors
#   6. ChromaDB returns the top-k most semantically similar chunks
#
# WHY THIS IS BETTER THAN KEYWORD SEARCH:
#   Keyword: "ML engineer" won't match "machine learning role"
#   Semantic: both map to similar vectors → both returned ✅
#
# FILE LOCATION:
#   Place this file at:  embeddings/pages/08_chroma_job_search.py
#   Job listings file:   embeddings/data/text/job_listings.txt
# ============================================================

# FIX: hashlib added to generate a unique, filesystem-safe Chroma
# collection_name from the cache key (see Step 3 below).
import hashlib
import streamlit as st
from langchain_chroma import Chroma

# ── Import shared utilities ───────────────────────────────────
# embeddings_utils → resolves provider/model/key, builds embedding object
# pipeline_utils   → loads job_listings.txt from data/text/ and chunks it
from embeddings_utils import get_or_set_embedding_key, build_embeddings, get_embedding_dimensions
from pipeline_utils import load_and_chunk_text, list_available_files, summarise_chunks

# ── App Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Job Search Helper", layout="centered", page_icon="💼")
st.title("💼 Semantic Job Search")
st.markdown(
    "Search job listings by **meaning**, not just keywords. "
    "Ask naturally — _'remote Python role'_, _'entry level data job'_, "
    "_'leadership position in tech'_."
)

# ── Step 1: Resolve embedding model ───────────────────────────
# get_or_set_embedding_key() checks st.session_state first (fast path),
# then shows the provider/model/key UI if not yet entered.
# Keys are stored ONLY in session_state — never in os.environ.
api_key, embed_provider, model = get_or_set_embedding_key()

# CONDITIONAL OVERRIDE: If the user selected OpenAI, force the 3072-dimension
# model so it doesn't crash against the existing ChromaDB collection.
if embed_provider == "OpenAI":
    embed_model = "text-embedding-3-large"

# build_embeddings() returns the correct LangChain embeddings object:
#   OpenAI   → OpenAIEmbeddings(model=model, api_key=api_key)
#   Cohere   → CohereEmbeddings(...)
#   HuggingFace → HuggingFaceEmbeddings(model_name=model)  ← free, local
#   etc.
embeddings = build_embeddings(embed_provider, model, api_key)

dims = get_embedding_dimensions(embed_provider, model)
st.success(f"✅ Ready — **{embed_provider}** · `{model}`")
if dims:
    st.caption(f"📐 Vector dimensions: {dims}")

st.divider()

# ── Step 2: File selector ──────────────────────────────────────
# list_available_files() scans data/text/ and data/pdfs/ and
# returns {"text": [...filenames...], "pdfs": [...filenames...]}
# This lets the user pick which file to search without hardcoding.
available = list_available_files()

if not available["text"]:
    st.error(
        "No `.txt` files found in `data/text/`. "
        "Add a `job_listings.txt` file there to get started."
    )
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    # Let user pick which text file to search
    # Defaults to job_listings.txt if it exists, otherwise first file
    default_file = (
        "job_listings.txt"
        if "job_listings.txt" in available["text"]
        else available["text"][0]
    )

    # FIX: Track the previously selected file so we can detect when the user
    # switches documents. Without this, stale vectors from the old file remain
    # in session_state and contaminate searches against the new file.
    prev_file = st.session_state.get("job_prev_file")
    selected_file = st.selectbox(
        "📄 Job listings file",
        available["text"],
        index=available["text"].index(default_file),
    )
    # FIX: When the document changes, purge only this page's cached vectorstores
    # (keys prefixed "job_vs_") so the new file gets a clean index.
    # Auth keys ("embed_api_key" etc.) are intentionally left intact.
    if selected_file != prev_file:
        keys_to_clear = [k for k in st.session_state if k.startswith("job_vs_")]
        for k in keys_to_clear:
            del st.session_state[k]
        st.session_state["job_prev_file"] = selected_file

with col2:
    # chunk_size controls how much text each vector represents.
    # For job listings, 200 chars ≈ one job posting — keep it small
    # so each chunk is one distinct role, not a mix of several.
    chunk_size = st.number_input(
        "Chunk size (chars)",
        min_value=100,
        max_value=2000,
        value=200,
        step=50,
        help="200 chars ≈ one job posting. Smaller = more precise search.",
    )

# ── Step 3: Load, chunk, and index ────────────────────────────
# We cache the vector store in session_state so it's only built
# once per session — rebuilding it on every Streamlit rerun would
# be slow and waste API credits.
#
# The cache key includes the filename, chunk size, and provider+model
# so it rebuilds automatically when any of those change.
#
# FIX: Cache key now uses the "job_vs_" prefix so it is scoped to this
# page only and never collides with cache keys from other RAG pages that
# share the same session_state (e.g. 10_rag_demo, 12_pdf_rag_demo).
cache_key = f"job_vs_{selected_file}_{chunk_size}_{embed_provider}_{model}"

if cache_key not in st.session_state:
    with st.spinner(f"Loading and indexing `{selected_file}`..."):

        # ── load_and_chunk_text() is a one-liner from pipeline_utils:
        #   1. TextLoader reads data/text/{selected_file}
        #   2. RecursiveCharacterTextSplitter splits into chunks
        #   3. Returns list[Document], each with .page_content + .metadata
        #
        # chunk_overlap=10 means adjacent chunks share 10 characters —
        # prevents a job title being cut off at a chunk boundary.
        chunks = load_and_chunk_text(
            selected_file,
            chunk_size=chunk_size,
            chunk_overlap=10,
            strategy="recursive",
        )

        if not chunks:
            st.error(f"`{selected_file}` is empty or could not be loaded.")
            st.stop()

        # ── Chroma.from_documents() does two things in one call:
        #   a) Calls embeddings.embed_documents() on all chunks
        #      → converts every chunk's text into a vector
        #   b) Stores those vectors in an in-memory ChromaDB collection
        #
        # In-memory means the index is lost when the app restarts.
        # For persistence across restarts, add:
        #   persist_directory="./chroma_db"
        # to the Chroma.from_documents() call.
        #
        # FIX: collection_name is now passed explicitly.
        # Without it, every call to Chroma.from_documents() writes into
        # Chroma's single default collection, so switching files merges
        # all their vectors together — queries return chunks from every
        # file ever indexed in this session.
        # The MD5 of cache_key produces a short, unique, filesystem-safe
        # name that is deterministic for a given (file + settings + provider).
        collection_name = "job-" + hashlib.md5(cache_key.encode()).hexdigest()[:16]
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,  # FIX: isolates this file's vectors
        )

        # Cache both the store and chunk stats in session_state
        st.session_state[cache_key]          = vectorstore
        st.session_state[cache_key + "_stats"] = summarise_chunks(chunks)

# Retrieve from cache
vectorstore = st.session_state[cache_key]
stats       = st.session_state[cache_key + "_stats"]

# Show indexing stats
st.info(
    f"📊 Indexed **{stats['total_chunks']} chunks** from `{selected_file}` — "
    f"avg {stats['avg_chars']} chars each "
    f"(min {stats['min_chars']}, max {stats['max_chars']})"
)

st.divider()

# ── Step 4: Search ────────────────────────────────────────────
st.subheader("🔍 Search jobs")

col_q, col_k = st.columns([3, 1])

with col_q:
    # Use st.text_input instead of input() — input() is a terminal
    # function that blocks Streamlit's event loop completely.
    query = st.text_input(
        "Describe the role you're looking for:",
        placeholder="e.g. remote Python developer, entry level data analyst...",
    )

with col_k:
    # k = number of results to return from the vector store.
    # db.as_retriever() defaults to k=4 — we expose it as a control.
    k = st.number_input("Results (k)", min_value=1, max_value=10, value=4)

if st.button("🔎 Search", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching..."):

            # ── as_retriever() wraps the vector store as a LangChain
            # Retriever object. search_kwargs={"k": k} sets how many
            # top results to return.
            #
            # Under the hood, retriever.invoke(query) does:
            #   1. embeddings.embed_query(query) → query vector
            #   2. ChromaDB cosine similarity search against stored vectors
            #   3. Returns top-k Document objects sorted by similarity
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            results   = retriever.invoke(query)

        if not results:
            st.warning("No matching jobs found. Try a different query.")
        else:
            st.success(f"Found {len(results)} matching job(s):")

            for i, doc in enumerate(results, 1):
                # Each result is a LangChain Document with:
                #   doc.page_content → the chunk text (job listing snippet)
                #   doc.metadata     → {"source": "...", "chunk": N, ...}
                with st.expander(f"Result {i}", expanded=(i == 1)):
                    st.write(doc.page_content)

                    # Show metadata so users can see which file/chunk
                    # each result came from — essential for RAG trust
                    source = doc.metadata.get("source", "unknown")
                    chunk  = doc.metadata.get("chunk", "?")
                    st.caption(
                        f"📁 Source: `{source.split('/')[-1]}`  "
                        f"· Chunk: {chunk}"
                    )

# ── Debug: show raw chunks (collapsible) ──────────────────────
with st.expander("🔬 Show all indexed chunks"):
    st.caption(
        "These are all the chunks stored in ChromaDB. "
        "Each one has its own embedding vector."
    )
    # Peek at all documents in the Chroma collection
    all_chunks = vectorstore.get()
    for i, text in enumerate(all_chunks["documents"]):
        st.markdown(f"**Chunk {i}:** {text}")
        st.divider()