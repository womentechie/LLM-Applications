# ============================================================
# CONCEPT: RAG — Retrieval-Augmented Generation
# ============================================================
# RAG solves a fundamental LLM limitation: LLMs only know what
# they were trained on. They can't answer questions about YOUR
# documents (product data, internal docs, research papers, etc.)
#
# RAG fixes this by:
#   1. Splitting your documents into small chunks
#   2. Embedding each chunk → storing vectors in ChromaDB
#   3. When a question arrives, embedding it too
#   4. Finding the most relevant chunks via similarity search
#   5. Passing those chunks as context to the LLM
#   6. LLM generates an answer GROUNDED in your actual documents
#
# TWO MODELS ARE NEEDED — THIS IS WHY app.py RESOLVES BOTH:
#   Embedding model → converts text to vectors (retrieval step)
#   Chat model      → generates the answer (generation step)
#   You can mix providers: OpenAI embeddings + Claude for answers ✅
#
# WHAT CHANGED FROM THE ORIGINAL:
#   Before → Hardcoded OpenAI only, os.environ key, input() terminal
#   After  → Any provider via utils/embeddings_utils, session-safe keys,
#             pipeline_utils for loading, full Streamlit UI,
#             source display so users see WHERE the answer came from
#
# CHAINS USED:
#   create_stuff_documents_chain (langchain_classic) → takes retrieved chunks
#     + question, formats them into a prompt, sends to LLM, returns answer
#   create_retrieval_chain (langchain_classic) → wraps the above with the
#     retriever: automatically retrieves chunks for each question then calls
#     the stuff chain
#
# NOTE ON IMPORTS:
#   LangChain recently restructured — chains moved to langchain-classic.
#   Use: from langchain_classic.chains.retrieval import create_retrieval_chain
#   NOT: from langchain.chains.retrieval import create_retrieval_chain
#
# FILE LOCATION:
#   embeddings/pages/rag_demo.py
#   Place your document at: embeddings/data/text/product-data.txt
# ============================================================

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ── Import shared utilities ───────────────────────────────────
# embeddings_utils → embedding model (for vectorising docs + queries)
# utils            → chat model     (for generating answers)
# pipeline_utils   → document loading + chunking
from embeddings_utils import (
    get_or_set_embedding_key,
    build_embeddings,
    get_embedding_dimensions,
)
from utils import get_or_set_api_key, build_llm
from pipeline_utils import (
    load_and_chunk_text,
    load_and_chunk_all,
    list_available_files,
    summarise_chunks,
)

# ── App Setup ──────────────────────────────────────────────────
st.set_page_config(page_title="RAG Demo", layout="centered", page_icon="🧠")
st.title("🧠 RAG — Chat with Your Documents")
st.markdown(
    "Ask questions about your own documents. "
    "The app retrieves the most relevant sections and uses an LLM to "
    "generate a grounded answer with source references."
)

# ── Step 1: Resolve both model keys ───────────────────────────
# Both keys were resolved in app.py before this page loaded.
# These calls hit the session_state fast path — no UI shown.
#
# EMBEDDING key — used to vectorise chunks + embed user queries
# Stored under: "embed_api_key", "embed_provider", "embed_model"
embed_key, embed_provider, embed_model = get_or_set_embedding_key()

# CHAT key — used to generate the final answer from retrieved context
# Stored under: "api_key", "provider", "model"
chat_key, chat_provider, chat_model = get_or_set_api_key()

# ── Step 2: Build both model objects ──────────────────────────
# build_embeddings() → correct LangChain Embeddings class for provider
# build_llm()        → correct LangChain Chat class for provider
embeddings = build_embeddings(embed_provider, embed_model, embed_key)
llm        = build_llm(chat_provider, chat_model, chat_key)

# Show active configuration
col1, col2 = st.columns(2)
with col1:
    dims = get_embedding_dimensions(embed_provider, embed_model)
    st.success(f"🔢 Embedding: **{embed_provider}** · `{embed_model}`")
    if dims:
        st.caption(f"📐 {dims}-dimensional vectors")
with col2:
    st.success(f"💬 Chat LLM: **{chat_provider}** · `{chat_model}`")
    st.caption("Generates answers from retrieved context")

st.divider()

# ── Step 3: Document selection ────────────────────────────────
# list_available_files() scans data/text/ and data/pdfs/
# Returns {"text": [...], "pdfs": [...]}
st.subheader("📄 Document source")

available = list_available_files()

# Combine text and pdf options for the selector
all_files = (
    [f"text/{f}" for f in available["text"]] +
    [f"pdfs/{f}" for f in available["pdfs"]]
)

if not all_files:
    st.error(
        "No files found in `data/text/` or `data/pdfs/`. "
        "Add `product_data.txt` to `data/text/` to get started."
    )
    st.stop()

col_file, col_size, col_overlap = st.columns([3, 1, 1])

with col_file:
    # Default to product_data.txt if it exists
    default = next(
        (f for f in all_files if "product_data" in f),
        all_files[0]
    )
    selected = st.selectbox(
        "Choose a document",
        all_files,
        index=all_files.index(default),
    )

with col_size:
    # chunk_size controls how much text each vector represents.
    # 1000 chars ≈ 200 tokens — good for product descriptions.
    chunk_size = st.number_input(
        "Chunk size",
        min_value=100,
        max_value=3000,
        value=1000,
        step=100,
        help="Characters per chunk. Larger = more context per result.",
    )

with col_overlap:
    # chunk_overlap prevents a sentence being cut at a chunk boundary.
    # 200 chars on 1000 char chunks = 20% overlap.
    chunk_overlap = st.number_input(
        "Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Shared chars between adjacent chunks.",
    )

# ── Step 4: Load, chunk, and index ────────────────────────────
# Cache the vector store in session_state — rebuilding it on every
# Streamlit rerun wastes time and API credits.
# Cache key includes file + chunk settings + both providers/models
# so it rebuilds automatically when any of those change.
cache_key = (
    f"rag_vs_{selected}_{chunk_size}_{chunk_overlap}"
    f"_{embed_provider}_{embed_model}"
)

if cache_key not in st.session_state:
    with st.spinner(f"Loading and indexing `{selected}`..."):

        # pipeline_utils handles loading (TextLoader or PyPDFLoader)
        # and chunking (RecursiveCharacterTextSplitter) in one call.
        # Returns list[Document] with .page_content and .metadata
        filename = selected.split("/", 1)[1]   # strip "text/" or "pdfs/"
        if selected.startswith("text/"):
            chunks = load_and_chunk_text(
                filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy="recursive",
            )
        else:
            from pipeline_utils import load_and_chunk_pdf
            chunks = load_and_chunk_pdf(
                filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy="recursive",
            )

        if not chunks:
            st.error(f"`{selected}` is empty or could not be loaded.")
            st.stop()

        # Chroma.from_documents() embeds every chunk and stores vectors.
        # In-memory — index lost on restart. For persistence add:
        #   persist_directory="./chroma_db"
        vector_store = Chroma.from_documents(chunks, embeddings)

        # Cache the store and chunk stats
        st.session_state[cache_key]           = vector_store
        st.session_state[cache_key + "_stats"] = summarise_chunks(chunks)

# Retrieve from cache
vector_store = st.session_state[cache_key]
stats        = st.session_state[cache_key + "_stats"]

st.info(
    f"📊 Indexed **{stats['total_chunks']} chunks** — "
    f"avg {stats['avg_chars']} chars each · "
    f"sources: {', '.join(stats['sources'])}"
)

st.divider()

# ── Step 5: Build the RAG chain ───────────────────────────────
# retriever.invoke(query) does:
#   1. embed_query(query) → query vector
#   2. ChromaDB cosine similarity search → top-k chunks
#   3. Returns list[Document]
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}   # retrieve top 3 most relevant chunks
)

# CONCEPT: ChatPromptTemplate with {context} and {input}
# {context} → filled automatically by create_retrieval_chain with
#             the retrieved document chunks formatted as plain text
# {input}   → the user's question
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for answering questions about documents.
Use ONLY the provided context to respond.
If the answer isn't in the context, say you don't know — do not make things up.
Keep your response to three concise sentences maximum.

Context:
{context}""",
        ),
        ("human", "{input}"),
    ]
)

# CONCEPT: create_stuff_documents_chain
# "Stuff" means all retrieved chunks are stuffed into one prompt.
# It formats the list[Document] chunks into a single {context} string,
# builds the full prompt, sends it to the LLM, and returns the answer.
# Good for small-to-medium documents. For very large document sets,
# use map_reduce or refine chains instead.
qa_chain = create_stuff_documents_chain(llm, prompt_template)

# CONCEPT: create_retrieval_chain
# Wraps the retriever and qa_chain into a single end-to-end pipeline.
# When invoked with {"input": question}:
#   1. Retriever finds relevant chunks
#   2. qa_chain formats them + generates the answer
# Returns: {"input": ..., "context": [Documents], "answer": "..."}
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ── Step 6: Question UI ───────────────────────────────────────
st.subheader("💬 Ask a question")

col_q, col_k = st.columns([4, 1])
with col_q:
    # st.text_input instead of input() — input() blocks Streamlit's
    # event loop and freezes the entire app.
    question = st.text_input(
        "Your question:",
        placeholder="e.g. What are the features of the XYZ smartphone?",
    )
with col_k:
    k = st.number_input(
        "Sources (k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of document chunks to retrieve.",
    )
    # Update retriever k dynamically
    retriever.search_kwargs["k"] = k

if st.button("🔍 Ask", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            # invoke() runs the full RAG pipeline end-to-end.
            # Returns a dict:
            #   response["answer"]  → the LLM's generated answer string
            #   response["context"] → list[Document] chunks used
            #   response["input"]   → the original question
            response = rag_chain.invoke({"input": question})

        # ── Display answer ─────────────────────────────────────
        st.markdown("### 💡 Answer")
        st.write(response["answer"])

        # ── Display source chunks ──────────────────────────────
        # Showing sources is critical for RAG trust — users should
        # always be able to verify WHERE the answer came from.
        st.markdown("### 📄 Sources used")
        context_docs = response.get("context", [])

        if context_docs:
            for i, doc in enumerate(context_docs, 1):
                source = doc.metadata.get("source", "unknown")
                chunk  = doc.metadata.get("chunk", "?")
                page   = doc.metadata.get("page", None)

                # Build a readable label for the expander
                label_parts = [f"Source {i} — `{source.split('/')[-1]}`"]
                if page is not None:
                    label_parts.append(f"page {page + 1}")
                label_parts.append(f"chunk {chunk}")

                with st.expander(" · ".join(label_parts), expanded=(i == 1)):
                    st.write(doc.page_content)
        else:
            st.caption("No source chunks returned.")

# ── Debug: show all indexed chunks ────────────────────────────
with st.expander("🔬 Show all indexed chunks"):
    all_chunks = vector_store.get()
    for i, text in enumerate(all_chunks["documents"]):
        st.markdown(f"**Chunk {i}:** {text[:300]}{'...' if len(text) > 300 else ''}")
        st.divider()