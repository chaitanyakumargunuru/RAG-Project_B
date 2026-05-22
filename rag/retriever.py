"""
rag/retriever.py — RAG Logic (Member C)

Core RAG pipeline:
  1. Embed the user's question
  2. Retrieve top-k relevant chunks from ChromaDB
  3. Build a prompt: System context + retrieved chunks + question
  4. Send to Ollama (local LLM) for a grounded answer
  5. Return answer + source metadata

Requires:
  • Ollama running locally:  ollama serve
  • Model pulled:            ollama pull llama3.2

Usage:
    from rag.retriever import retrieve_and_answer
    result = retrieve_and_answer("user_42", "What is retrieval-augmented generation?")
    print(result["answer"])
    print(result["sources"])
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from rag.vectorstore import similarity_search
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K_RESULTS

# ── Prompt Template ───────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.

Use ONLY the information in the context below to answer the question.
If the context does not contain enough information to answer, say:
"I couldn't find relevant information in the uploaded documents."

Do NOT make up information.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

# ── LLM singleton ─────────────────────────────────────────────────────────────
_llm_instance = None


def _get_llm() -> OllamaLLM:
    """Return a cached OllamaLLM instance."""
    global _llm_instance
    if _llm_instance is None:
        print(f"[Retriever] Connecting to Ollama — model: {OLLAMA_MODEL}")
        _llm_instance = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,        # low temp -> factual, grounded answers
        )
    return _llm_instance


# ── Main RAG Function ─────────────────────────────────────────────────────────

def retrieve_and_answer(user_id: str, question: str, k: int = TOP_K_RESULTS) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Args:
        user_id  : User identifier (must have documents already uploaded).
        question : Natural-language question from the chat interface.
        k        : Number of chunks to retrieve (default from config).

    Returns:
        dict with keys:
          "answer"  : str  — LLM-generated answer grounded in the documents
          "sources" : list — Metadata of retrieved chunks (file, page number, etc.)
          "chunks"  : list — Raw text of retrieved chunks (for debugging)
    """
    # Step 1: Retrieve relevant chunks
    retrieved_docs = similarity_search(user_id, question, k=k)

    if not retrieved_docs:
        return {
            "answer": "No documents found. Please upload documents before asking questions.",
            "sources": [],
            "chunks": [],
        }

    # Step 2: Format context from chunks
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Chunk {i} | Source: {os.path.basename(source)} | Page: {page}]\n"
            f"{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # Step 3: Generate answer via Ollama
    llm = _get_llm()
    prompt_text = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    print(f"[Retriever] Sending query to Ollama ({OLLAMA_MODEL})...")
    answer = llm.invoke(prompt_text)
    print(f"[Retriever] Answer received [OK]")

    # Step 4: Build source metadata for the frontend
    sources = []
    for doc in retrieved_docs:
        meta = doc.metadata.copy()
        meta["source"] = os.path.basename(meta.get("source", "Unknown"))
        sources.append(meta)

    return {
        "answer": answer.strip(),
        "sources": sources,
        "chunks": [doc.page_content for doc in retrieved_docs],
    }


def check_ollama_connection() -> dict:
    """
    Health check: verify Ollama is running and the model is available.

    Returns:
        dict — {"status": "ok"|"error", "message": str, "model": str}
    """
    try:
        llm = _get_llm()
        # Minimal invoke to test connectivity
        response = llm.invoke("Say 'OK' in one word.")
        return {
            "status": "ok",
            "message": f"Ollama is running. Model '{OLLAMA_MODEL}' is available.",
            "model": OLLAMA_MODEL,
            "test_response": response.strip(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "model": OLLAMA_MODEL,
        }
