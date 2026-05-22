"""
rag/embedder.py — Embedding Pipeline (Member C)

Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2):
  • 100% local — no API key
  • 384-dimensional vectors
  • Fast & accurate for semantic search

Singleton pattern: model is loaded ONCE per process to save memory & time.

Usage:
    from rag.embedder import get_embeddings
    embeddings = get_embeddings()   # pass to ChromaDB
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

# ── Singleton ──────────────────────────────────────────────────────────────────
_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.
    Downloads the model on first call (~90 MB), then reuses it.
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        print(f"[Embedder] Loading embedding model: {EMBEDDING_MODEL} ...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},       # swap to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
        )
        print("[Embedder] Embedding model loaded [OK]")

    return _embeddings_instance
