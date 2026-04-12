# pipeline_utils.py
# ============================================================
# CONCEPT: Shared Document Loading + Chunking Pipeline
# ============================================================
# This utility is the shared plumbing used by:
#   - Vector DB scripts  (08_chroma_basic.py, 10_faiss_basic.py ...)
#   - RAG scripts        (12_rag_basic.py, 13_rag_memory.py ...)
#
# WHY A SEPARATE FILE?
#   Loading and chunking is boilerplate — every vector DB and RAG
#   demo needs it. Instead of copy-pasting the same 10 lines into
#   every script, we define it once here and import it everywhere.
#   Chunking scripts (04-07) stay standalone — their job is to
#   TEACH splitting. Everything else USES this util.
#
# FOLDER STRUCTURE EXPECTED:
#   embeddings/
#   ├── pipeline_utils.py       ← this file
#   ├── embeddings_utils.py
#   └── data/
#       ├── text/               ← .txt files loaded by TextLoader
#       │   ├── sample.txt
#       │   ├── science.txt
#       │   └── history.txt
#       └── pdfs/               ← .pdf files loaded by PyPDFLoader
#           ├── report.pdf
#           └── manual.pdf
#
# USAGE EXAMPLES:
#   from pipeline_utils import load_text, load_pdf, load_all_texts,
#                               load_all_pdfs, load_all, chunk_documents
#
#   # Load and chunk a single text file
#   chunks = chunk_documents(load_text("science.txt"))
#
#   # Load and chunk a single PDF
#   chunks = chunk_documents(load_pdf("report.pdf"))
#
#   # Load and chunk ALL text files at once
#   chunks = chunk_documents(load_all_texts())
#
#   # Load and chunk ALL files (text + PDFs combined)
#   chunks = chunk_documents(load_all())
#
#   # Custom chunk size and overlap
#   chunks = chunk_documents(load_all_texts(), chunk_size=300, overlap=30)
# ============================================================

from pathlib import Path
from typing import Optional
import streamlit as st

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# ── Folder paths ──────────────────────────────────────────────
# Path(__file__) is the absolute path of this file (pipeline_utils.py).
# .parent gives the embeddings/ folder it lives in.
# We then resolve data/text/ and data/pdfs/ relative to that.
# This works correctly regardless of where `streamlit run app.py`
# is executed from — no hardcoded absolute paths needed.

BASE_DIR = Path(__file__).parent          # embeddings/
DATA_DIR = BASE_DIR / "data"             # embeddings/data/
TEXT_DIR = DATA_DIR / "text"             # embeddings/data/text/
PDF_DIR  = DATA_DIR / "pdfs"            # embeddings/data/pdfs/


# ── Loader helpers ─────────────────────────────────────────────────────────

def load_text(filename: str) -> list[Document]:
    """
    Load a single .txt file from data/text/.

    CONCEPT: TextLoader
      Reads a plain text file and returns it as a list containing
      one LangChain Document object. The Document has two fields:
        .page_content : the raw text string
        .metadata     : dict with {"source": "/path/to/file"}

    Args:
        filename: name of the file, e.g. "science.txt"

    Returns:
        list[Document] — one Document per file

    Example:
        docs = load_text("science.txt")
        print(docs[0].page_content[:200])
        print(docs[0].metadata)   # {"source": ".../data/text/science.txt"}
    """
    path = TEXT_DIR / filename
    _check_exists(path)
    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def load_pdf(filename: str) -> list[Document]:
    """
    Load a single .pdf file from data/pdfs/.

    CONCEPT: PyPDFLoader
      Reads a PDF and returns one Document per PAGE. Each Document has:
        .page_content : text extracted from that page
        .metadata     : {"source": "/path/to/file", "page": 0}

      This is different from TextLoader which gives one Document total.
      A 10-page PDF → 10 Documents, one per page.

    Args:
        filename: name of the file, e.g. "report.pdf"

    Returns:
        list[Document] — one Document per PDF page

    Example:
        docs = load_pdf("report.pdf")
        print(f"Pages loaded: {len(docs)}")
        print(docs[0].metadata)   # {"source": "...", "page": 0}
    """
    path = PDF_DIR / filename
    _check_exists(path)
    loader = PyPDFLoader(str(path))
    return loader.load()


def load_all_texts() -> list[Document]:
    """
    Load ALL .txt files found in data/text/.

    Scans the text/ folder and loads every .txt file it finds.
    Files are sorted alphabetically for consistent ordering.
    Metadata on each Document includes the source filename so you
    can trace which chunk came from which file in RAG responses.

    Returns:
        list[Document] — combined Documents from all .txt files

    Example:
        docs = load_all_texts()
        print(f"Loaded {len(docs)} documents from {TEXT_DIR}")
        for doc in docs:
            print(doc.metadata["source"])
    """
    _check_dir_exists(TEXT_DIR)
    txt_files = sorted(TEXT_DIR.glob("*.txt"))

    if not txt_files:
        _warn_empty(TEXT_DIR, "*.txt")
        return []

    all_docs = []
    for path in txt_files:
        loader = TextLoader(str(path), encoding="utf-8")
        docs   = loader.load()
        all_docs.extend(docs)

    return all_docs


def load_all_pdfs() -> list[Document]:
    """
    Load ALL .pdf files found in data/pdfs/.

    Scans the pdfs/ folder and loads every .pdf file it finds.
    Each page of each PDF becomes a separate Document.

    Returns:
        list[Document] — combined Documents from all PDF pages

    Example:
        docs = load_all_pdfs()
        print(f"Loaded {len(docs)} pages from {len(list(PDF_DIR.glob('*.pdf')))} PDFs")
    """
    _check_dir_exists(PDF_DIR)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        _warn_empty(PDF_DIR, "*.pdf")
        return []

    all_docs = []
    for path in pdf_files:
        loader = PyPDFLoader(str(path))
        docs   = loader.load()
        all_docs.extend(docs)

    return all_docs


def load_all() -> list[Document]:
    """
    Load ALL files from both data/text/ and data/pdfs/ combined.

    This is the most common entry point for RAG demos where you
    want to index your entire knowledge base in one call.

    Returns:
        list[Document] — all text documents + all PDF pages combined

    Example:
        docs = load_all()
        print(f"Total documents loaded: {len(docs)}")
    """
    text_docs = load_all_texts()
    pdf_docs  = load_all_pdfs()
    return text_docs + pdf_docs


# ── Chunking strategies ────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks for embedding.

    CONCEPT: Why chunking is necessary
      Embedding models have a token limit (e.g. 8191 for OpenAI small).
      More importantly, embedding a 10-page document as one vector
      loses precision — the vector tries to represent everything and
      ends up representing nothing well. Chunking first ensures each
      vector captures one focused idea, making retrieval accurate.

    CONCEPT: chunk_size vs chunk_overlap
      chunk_size    : max characters per chunk (not tokens — characters).
                      500 chars ≈ 100-125 tokens — a safe starting point.
      chunk_overlap : how many characters the end of one chunk shares
                      with the start of the next. Overlap prevents a
                      sentence from being split mid-thought at a boundary.
                      50 chars overlap on 500 char chunks = 10% overlap.

    CONCEPT: Two splitting strategies
      "recursive" (default, recommended):
          RecursiveCharacterTextSplitter tries to split on paragraph
          breaks first (\n\n), then sentence breaks (\n), then spaces,
          then characters as a last resort. This preserves natural
          language boundaries as much as possible.

      "character":
          CharacterTextSplitter splits purely on a single separator
          (default \n\n). Simpler but can create very uneven chunks
          if the text has irregular paragraph spacing.

    Args:
        docs         : list of Documents from any load_*() function
        chunk_size   : max characters per chunk (default 500)
        chunk_overlap: overlap between adjacent chunks (default 50)
        strategy     : "recursive" (default) or "character"

    Returns:
        list[Document] — chunked Documents, each with metadata
                         including {"source": ..., "chunk": N}

    Example:
        docs   = load_all_texts()
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=30)
        print(f"Split {len(docs)} docs into {len(chunks)} chunks")
        print(f"First chunk ({len(chunks[0].page_content)} chars):")
        print(chunks[0].page_content)
    """
    if not docs:
        return []

    # ── Select splitter ──────────────────────────────────────
    if strategy == "recursive":
        # Tries \n\n → \n → " " → "" in order
        # Best general-purpose choice for prose text and PDFs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,          # count characters, not tokens
            add_start_index=True,         # adds {"start_index": N} to metadata
        )
    elif strategy == "character":
        # Splits only on separator (default \n\n)
        # Good for showing students what fixed splitting looks like
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose 'recursive' or 'character'."
        )

    # ── Split ────────────────────────────────────────────────
    chunks = splitter.split_documents(docs)

    # ── Add chunk index to metadata ──────────────────────────
    # Useful in RAG to show users which chunk an answer came from
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i

    return chunks


# ── Convenience one-liners ─────────────────────────────────────────────────

def load_and_chunk_text(
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> list[Document]:
    """
    One-liner: load a single .txt file and chunk it.

    Example:
        chunks = load_and_chunk_text("science.txt")
        chunks = load_and_chunk_text("history.txt", chunk_size=300)
    """
    return chunk_documents(load_text(filename), chunk_size, chunk_overlap, strategy)


def load_and_chunk_pdf(
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> list[Document]:
    """
    One-liner: load a single .pdf file and chunk it.

    Example:
        chunks = load_and_chunk_pdf("report.pdf")
        chunks = load_and_chunk_pdf("manual.pdf", chunk_size=800, chunk_overlap=100)
    """
    return chunk_documents(load_pdf(filename), chunk_size, chunk_overlap, strategy)


def load_and_chunk_all(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> list[Document]:
    """
    One-liner: load ALL files (text + PDFs) and chunk everything.
    The most common entry point for RAG scripts.

    Example:
        chunks = load_and_chunk_all()
        chunks = load_and_chunk_all(chunk_size=1000, chunk_overlap=100)
    """
    return chunk_documents(load_all(), chunk_size, chunk_overlap, strategy)


# ── Inspection helpers ─────────────────────────────────────────────────────

def list_available_files() -> dict[str, list[str]]:
    """
    Returns a dict of all files available in data/text/ and data/pdfs/.
    Useful for building a file picker in Streamlit.

    Returns:
        {
            "text": ["science.txt", "history.txt", ...],
            "pdfs": ["report.pdf", "manual.pdf", ...],
        }

    Example:
        files = list_available_files()
        chosen = st.selectbox("Choose a text file", files["text"])
    """
    text_files = sorted(f.name for f in TEXT_DIR.glob("*.txt")) if TEXT_DIR.exists() else []
    pdf_files  = sorted(f.name for f in PDF_DIR.glob("*.pdf"))  if PDF_DIR.exists()  else []
    return {"text": text_files, "pdfs": pdf_files}


def summarise_chunks(chunks: list[Document]) -> dict:
    """
    Returns a summary dict about a list of chunks.
    Useful for displaying stats in Streamlit after chunking.

    Returns:
        {
            "total_chunks"  : int,
            "avg_chars"     : int,
            "min_chars"     : int,
            "max_chars"     : int,
            "sources"       : list[str],  # unique source files
        }

    Example:
        chunks = load_and_chunk_all()
        stats  = summarise_chunks(chunks)
        st.metric("Total chunks", stats["total_chunks"])
        st.metric("Avg chunk size", f"{stats['avg_chars']} chars")
    """
    if not chunks:
        return {"total_chunks": 0, "avg_chars": 0, "min_chars": 0,
                "max_chars": 0, "sources": []}

    lengths = [len(c.page_content) for c in chunks]
    sources = sorted(set(
        Path(c.metadata.get("source", "unknown")).name for c in chunks
    ))

    return {
        "total_chunks": len(chunks),
        "avg_chars":    int(sum(lengths) / len(lengths)),
        "min_chars":    min(lengths),
        "max_chars":    max(lengths),
        "sources":      sources,
    }


# ── Internal helpers ───────────────────────────────────────────────────────

def _check_exists(path: Path) -> None:
    """Raise a clear error if a specific file doesn't exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Place your file in: {path.parent}"
        )


def _check_dir_exists(directory: Path) -> None:
    """Raise a clear error if the data subfolder doesn't exist."""
    if not directory.exists():
        raise FileNotFoundError(
            f"Data folder not found: {directory}\n"
            f"Create it and add your files:\n  mkdir -p {directory}"
        )


def _warn_empty(directory: Path, pattern: str) -> None:
    """Show a Streamlit warning (or print) if a folder has no matching files."""
    msg = f"No {pattern} files found in {directory}"
    try:
        st.warning(msg)
    except Exception:
        print(f"WARNING: {msg}")