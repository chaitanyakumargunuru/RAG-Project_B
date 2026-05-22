"""
rag/vectorstore.py — Vector Database Management (Member C)

Uses ChromaDB with local persistence (./chroma_db directory).
Each user gets their own isolated collection: "user_{user_id}"

Functions:
  add_documents(user_id, chunks)           — embed & store chunks
  similarity_search(user_id, query, k)     — retrieve top-k relevant chunks
  delete_collection(user_id)               — wipe a user's data
  list_collections()                       — admin: show all collections
  get_collection_info(user_id)             — doc count for a user

Usage:
    from rag.vectorstore import add_documents, similarity_search
    add_documents("user_42", chunks)
    results = similarity_search("user_42", "What is RAG?")
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from langchain_chroma import Chroma
from rag.embedder import get_embeddings
from config import CHROMA_DB_PATH, TOP_K_RESULTS


def _get_collection_name(user_id: str) -> str:
    """Generate a safe ChromaDB collection name for a given user."""
    # ChromaDB collection names must match [a-zA-Z0-9_-]
    safe_id = str(user_id).replace("@", "_at_").replace(".", "_").replace(" ", "_")
    return f"user_{safe_id}"


def add_documents(user_id: str, chunks: list) -> int:
    """
    Embed and store document chunks in the user's ChromaDB collection.

    Args:
        user_id : Unique identifier for the user (e.g., user ID from auth DB).
        chunks  : list[Document] from splitter.split_documents().

    Returns:
        int — Number of chunks stored.
    """
    collection_name = _get_collection_name(user_id)
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    vectorstore.add_documents(chunks)

    count = vectorstore._collection.count()
    print(f"[VectorStore] User '{user_id}' — {len(chunks)} chunk(s) added "
          f"(total in collection: {count})")
    return len(chunks)


def similarity_search(user_id: str, query: str, k: int = TOP_K_RESULTS) -> list:
    """
    Find the top-k most relevant document chunks for a given query.

    Args:
        user_id : User whose collection to search.
        query   : The user's natural-language question.
        k       : Number of results to return (default from config).

    Returns:
        list[Document] — Ordered by relevance (most relevant first).
    """
    collection_name = _get_collection_name(user_id)
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    results = vectorstore.similarity_search(query, k=k)
    print(f"[VectorStore] Query: '{query[:60]}...' → {len(results)} chunk(s) retrieved")
    return results


def delete_collection(user_id: str) -> bool:
    """
    Delete all documents in a user's collection.

    Returns: True if deleted, False if collection did not exist.
    """
    collection_name = _get_collection_name(user_id)

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection(name=collection_name)
        print(f"[VectorStore] Collection '{collection_name}' deleted.")
        return True
    except Exception as e:
        print(f"[VectorStore] Could not delete '{collection_name}': {e}")
        return False


def list_collections() -> list[str]:
    """Return names of all ChromaDB collections (admin utility)."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return [c.name for c in client.list_collections()]


def get_collection_info(user_id: str) -> dict:
    """
    Return metadata about a user's collection.

    Returns: {"user_id": ..., "collection": ..., "document_count": ...}
    """
    collection_name = _get_collection_name(user_id)

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        col = client.get_collection(name=collection_name)
        count = col.count()
    except Exception:
        count = 0

    return {
        "user_id": user_id,
        "collection": collection_name,
        "document_count": count,
    }
