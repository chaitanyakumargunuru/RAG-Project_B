"""
rag/splitter.py — Text Splitting (Member C)

Splits LangChain Document objects into overlapping chunks.
Default settings (from config.py):
  • chunk_size    = 1000 chars
  • chunk_overlap = 200 chars

Usage:
    from rag.splitter import split_documents
    chunks = split_documents(documents)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split a list of LangChain Document objects into smaller overlapping chunks.

    Args:
        documents    : Output from any loader (list of Document objects).
        chunk_size   : Maximum characters per chunk (default from config).
        chunk_overlap: Characters shared between adjacent chunks (default from config).

    Returns:
        list[Document] — Each Document has .page_content and .metadata preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer natural breaks
    )

    chunks = splitter.split_documents(documents)

    print(f"[Splitter] {len(documents)} document(s) -> {len(chunks)} chunk(s) "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    return chunks
