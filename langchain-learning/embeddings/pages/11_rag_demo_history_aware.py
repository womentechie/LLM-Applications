# ==============================================================================
# CONVERSATIONAL RAG APPLICATION (Retrieval-Augmented Generation)
# ==============================================================================
# This script builds a web-based AI assistant that can read your documents
# and answer questions about them while remembering the context of the chat.
# ==============================================================================

# FIX: hashlib added to generate a unique, filesystem-safe Chroma
# collection_name from the cache key (see Section 4 below).
import hashlib
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever

# -- MEMORY MANAGERS --
# These two imports allow LangChain to automatically read/write to Streamlit's memory.
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# -- CUSTOM UTILITIES --
# These are local helper files (embeddings_utils.py, utils.py, pipeline_utils.py)
# that hide the messy logic of loading files and switching between OpenAI/Google/etc.
from embeddings_utils import get_or_set_embedding_key, build_embeddings, get_embedding_dimensions
from utils import get_or_set_api_key, build_llm
from pipeline_utils import load_and_chunk_text, load_and_chunk_all, list_available_files, summarise_chunks

# ==============================================================================
# 1. APP SETUP & UI INITIALIZATION
# ==============================================================================

# Configure the Streamlit browser tab
st.set_page_config(page_title="RAG Demo", layout="centered", page_icon="🧠")
st.title("🧠 Conversational RAG")
st.markdown(
    "Ask questions about your documents. The AI remembers the conversation, "
    "powered natively by LangChain's `RunnableWithMessageHistory`."
)

# ==============================================================================
# 2. LOAD MODELS (Embeddings & LLM)
# ==============================================================================

# Fetch the API keys and requested models from Streamlit's background state.
# (This assumes a previous setup page already asked the user for their keys).
embed_key, embed_provider, embed_model = get_or_set_embedding_key()
chat_key, chat_provider, chat_model = get_or_set_api_key()

# FORCED OVERRIDE: Hardcode the embedding model to the 3072-dimension version
# CONDITIONAL OVERRIDE: If the user selected OpenAI, force the 3072-dimension
# model so it doesn't crash against the existing ChromaDB collection.
if embed_provider == "OpenAI":
    embed_model = "text-embedding-3-large"

# Build the actual LangChain model objects.
# 'embeddings' translates text into math vectors.
# 'llm' generates the conversational text responses.
embeddings = build_embeddings(embed_provider, embed_model, embed_key)
llm = build_llm(chat_provider, chat_model, chat_key)

# Display the active models to the user in two neat columns
col1, col2 = st.columns(2)
with col1:
    dims = get_embedding_dimensions(embed_provider, embed_model)
    st.success(f"🔢 Embedding: **{embed_provider}** · `{embed_model}`")
with col2:
    st.success(f"💬 Chat LLM: **{chat_provider}** · `{chat_model}`")

st.divider()

# ==============================================================================
# 3. DOCUMENT SELECTION UI
# ==============================================================================
st.subheader("📄 Document source")

# Scan the local /text/ and /pdfs/ folders for files the user can chat with
available = list_available_files()
all_files = (
        [f"text/{f}" for f in available["text"]] +
        [f"pdfs/{f}" for f in available["pdfs"]]
)

# Stop the app completely if no files are found
if not all_files:
    st.error("No files found. Add documents to data folders to get started.")
    st.stop()

# Create UI columns for file selection and chunking settings
col_file, col_size, col_overlap = st.columns([3, 1, 1])

with col_file:
    # Try to select "product_data" by default, otherwise pick the first file in the list
    default = next((f for f in all_files if "product_data" in f), all_files[0])

    # FIX: Track the previously selected file so we can detect when the user
    # switches documents. Without this, stale vectors from the old file remain
    # in session_state and contaminate answers about the new file, and the old
    # chat history would still be fed into the history-aware retriever causing
    # the AI to reason about the wrong document's context.
    prev_selected = st.session_state.get("hist_prev_selected")
    selected = st.selectbox("Choose a document", all_files, index=all_files.index(default))

    # FIX: When the document changes, purge only this page's cached vectorstores
    # (keys prefixed "hist_vs_") so the new file gets a clean index.
    # Also clear chat_history — old conversation turns reference the previous
    # document's content and would mislead the history-aware retriever when
    # reformulating questions about the newly selected file.
    # Auth keys ("api_key", "embed_api_key" etc.) are intentionally left intact.
    if selected != prev_selected:
        keys_to_clear = [k for k in st.session_state if k.startswith("hist_vs_")]
        for k in keys_to_clear:
            del st.session_state[k]
        st.session_state["chat_history"] = []   # FIX: prevents stale history leaking into new doc Q&A
        st.session_state["hist_prev_selected"] = selected

with col_size:
    # How large each piece of text should be (in characters)
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=3000, value=1000, step=100)

with col_overlap:
    # How much text should overlap between chunks to prevent cutting a sentence in half
    chunk_overlap = st.number_input("Overlap", min_value=0, max_value=500, value=200, step=50)

# ==============================================================================
# 4. DATA PIPELINE: LOAD, CHUNK, AND VECTORIZE (WITH CACHING)
# ==============================================================================

# CACHING EXPLANATION:
# Vectorizing documents costs money and time. We create a unique "cache_key" based on the file
# and settings. If these haven't changed, we pull the database from memory instead of rebuilding it.
#
# FIX: Cache key now uses the "hist_vs_" prefix so it is scoped to this
# page only and never collides with cache keys from other RAG pages that
# share the same session_state (e.g. 10_rag_demo, 12_pdf_rag_demo).
cache_key = f"hist_vs_{selected}_{chunk_size}_{chunk_overlap}_{embed_provider}_{embed_model}"

if cache_key not in st.session_state:
    with st.spinner(f"Loading and indexing `{selected}`..."):

        # Strip the folder name (e.g., "text/") to get just the filename
        filename = selected.split("/", 1)[1]

        # Load and chunk the file based on its extension
        if selected.startswith("text/"):
            chunks = load_and_chunk_text(filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                         strategy="recursive")
        else:
            from pipeline_utils import load_and_chunk_pdf

            chunks = load_and_chunk_pdf(filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                        strategy="recursive")

        if not chunks:
            st.error(f"`{selected}` is empty or could not be loaded.")
            st.stop()

        # Chroma takes the text chunks, runs them through the Embedding model,
        # and saves the resulting math vectors into a temporary database.
        #
        # FIX: collection_name is now passed explicitly.
        # Without it, every call to Chroma.from_documents() writes into
        # Chroma's single default collection, so switching files merges
        # all their vectors together — queries return chunks from every
        # file ever indexed in this session.
        # The MD5 of cache_key produces a short, unique, filesystem-safe
        # name that is deterministic for a given (file + settings + provider).
        collection_name = "hist-" + hashlib.md5(cache_key.encode()).hexdigest()[:16]
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=collection_name,  # FIX: isolates this file's vectors
        )

        # Save the database and stats into Streamlit's session memory
        st.session_state[cache_key] = vector_store
        st.session_state[cache_key + "_stats"] = summarise_chunks(chunks)

# Retrieve the cached database so we can search it
vector_store = st.session_state[cache_key]
stats = st.session_state[cache_key + "_stats"]

st.info(f"📊 Indexed **{stats['total_chunks']} chunks**")
st.divider()

# ==============================================================================
# 5. BUILD THE AI "BRAIN" (THE CONVERSATIONAL RAG CHAIN)
# ==============================================================================
st.subheader("💬 Chat")

# We ask for the 'k' value BEFORE we build the AI brain,
# ensuring the retriever gets locked in with the correct number from the start.
col_blank, col_k = st.columns([4, 1])
with col_k:
    k = st.number_input("Sources (k)", min_value=1, max_value=10, value=3)

# ==============================================================================
# 6. BUILD THE AI "BRAIN" (THE CONVERSATIONAL RAG CHAIN)
# ==============================================================================

# Now the retriever uses your selected 'k' right at instantiation
# FIX: changed hardcoded k=3 to the variable `k` read from the number_input
# widget above. Previously the widget existed in the UI but had no effect
# because the literal 3 was passed here instead of the widget's value.
retriever = vector_store.as_retriever(search_kwargs={"k": k})  # FIX: was {"k": 3}

# -- CHAIN PART 1: The "History-Aware" Retriever --
# If a user says "How much does IT cost?", the AI doesn't know what "IT" is.
# This prompt tells the AI to look at the chat history and rewrite the question
# (e.g., "How much does the iPhone cost?") BEFORE searching the database.
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),  # Injects past conversation
    ("human", "{input}"),  # Injects the new question
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# -- CHAIN PART 2: The Document Reader (QA Chain) --
# This prompt takes the actual document chunks found by the retriever and
# asks the AI to answer the user's question using ONLY that text.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for answering questions. Use the provided context to respond. If the answer isn't clear, acknowledge that you don't know. Limit your response to three concise sentences.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine Part 1 and Part 2 into a single cohesive RAG pipeline
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# ==============================================================================
# 7. ATTACH MEMORY TO THE CHAIN
# ==============================================================================

# Tell Streamlit to create a dedicated memory array under the key "chat_history"
history_for_chain = StreamlitChatMessageHistory(key="chat_history")

# Wrap our RAG pipeline in LangChain's automatic memory manager.
# When triggered, this manager will:
#   1. Fetch history from Streamlit
#   2. Feed it into the `rag_chain`
#   3. Save the new Question and Answer back into Streamlit automatically.
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,  # Lambda required by LangChain syntax
    input_messages_key="input",  # Where to put the user's prompt
    history_messages_key="chat_history",  # Where to put the memory array
    output_messages_key="answer"  # CRITICAL: Only save the AI's final text answer, NOT the raw PDF chunks
)

# ==============================================================================
# 8. THE CHAT INTERFACE & EXECUTION
# ==============================================================================
# Render any existing chat history immediately so it stays on screen after a refresh
for msg in history_for_chain.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# st.chat_input is now freely placed at the bottom, avoiding Streamlit column errors
question = st.chat_input("e.g. What are the features? -> How much is it?")

if question:
    # 1. Print the user's new question to the UI immediately
    with st.chat_message("user"):
        st.write(question)

    # 2. Trigger the chain.
    # Because we used `RunnableWithMessageHistory`, we just pass the input.
    # The manager handles injecting the `chat_history` automatically.
    with st.spinner("Retrieving context and generating answer..."):
        response = chain_with_history.invoke(
            {"input": question},
            {"configurable": {"session_id": "abc123"}}  # Dummy ID required by LangChain
        )

    # 3. Print the AI's final answer to the screen
    with st.chat_message("assistant"):
        st.write(response["answer"])

        # 4. Display the source chunks used to generate the answer
        # This unpacks the list of Documents returned by the retriever step
        context_docs = response.get("context", [])
        if context_docs:
            # Hide the sources inside a dropdown expander to keep UI clean
            with st.expander("📄 Sources used"):
                for i, doc in enumerate(context_docs, 1):
                    # Safely extract metadata (like file name or chunk number)
                    source = doc.metadata.get("source", "unknown")
                    chunk = doc.metadata.get("chunk", "?")

                    st.markdown(f"**Source {i}** (`{source.split('/')[-1]}`, chunk {chunk})")
                    st.caption(doc.page_content)