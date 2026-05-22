"""
rag/loader.py — Document Loading (Member C)

Supports:
  • PDF  (.pdf)  — via PyPDFLoader
  • Word (.docx) — via Docx2txtLoader
  • Text (.txt)  — via TextLoader

Usage:
    from rag.loader import load_document
    docs = load_document("path/to/file.pdf")
"""

import os
from typing import Union, Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
# PyMuPDF4LLMLoader lives in its own high-performance integration package
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

# ── Individual loaders ────────────────────────────────────────────────────────

def load_pdf(pdf_path: str, lazy: bool = False) -> Union[list[Document], Iterator[Document]]:
    """
    Load a PDF file and extract text/tables as clean structured Markdown blocks.
    """
    loader = PyMuPDF4LLMLoader(pdf_path)
    return loader.lazy_load() if lazy else loader.load()


def load_docx(docx_path: str, lazy: bool = False) -> Union[list[Document], Iterator[Document]]:
    """
    Load a Word (.docx) file and extract its content.
    """
    loader = Docx2txtLoader(docx_path)
    return loader.lazy_load() if lazy else loader.load()


def load_txt(txt_path: str, lazy: bool = False) -> Union[list[Document], Iterator[Document]]:
    """
    Load a plain-text file with utf-8 decoding fallback.
    """
    loader = TextLoader(txt_path, encoding="utf-8")
    return loader.lazy_load() if lazy else loader.load()


# ── Dispatcher ────────────────────────────────────────────────────────────────

def load_document(file_path: str, lazy: bool = False) -> Union[list[Document], Iterator[Document]]:
    """
    Auto-detect file type by extension and delegate to the correct loader.

    Args:
        file_path: Absolute or relative string path to the target document.
        lazy: If True, returns a lazy generator (Iterator) streaming pages one by one.
              If False, returns a static pre-loaded list of Document elements.

    Supported extensions: .pdf, .docx, .txt
    Returns: list[Document] or Iterator[Document]
    Raises:  ValueError for unsupported file types.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        return load_pdf(file_path, lazy=lazy)
    elif ext == ".docx":
        return load_docx(file_path, lazy=lazy)
    elif ext == ".txt":
        return load_txt(file_path, lazy=lazy)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported: .pdf, .docx, .txt"
        )