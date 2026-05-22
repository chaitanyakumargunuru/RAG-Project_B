"""
rag/api.py — FastAPI Endpoints for Member C's RAG Service

Runs on port 8001 (Member B's auth backend typically uses 8000).

Endpoints:
  POST   /process              — Upload a document, run the full RAG pipeline
  POST   /chat                 — Ask a question, get a grounded answer
  GET    /health               — Check service + Ollama status
  GET    /collection/{user_id} — Get info about a user's vector collection
  DELETE /collection/{user_id} — Delete a user's vector collection

Start server:
    uvicorn rag.api:app --host 0.0.0.0 --port 8001 --reload

Interactive docs:
    http://localhost:8001/docs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.loader import load_document
from rag.splitter import split_documents
from rag.vectorstore import (
    add_documents,
    similarity_search,
    delete_collection,
    get_collection_info,
    list_collections,
)
from rag.retriever import retrieve_and_answer, check_ollama_connection
from config import UPLOAD_DIR

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Pipeline API — Member C",
    description=(
        "AI & Vector Data Engineer endpoints.\n"
        "Handles document processing, embeddings, and RAG-powered Q&A."
    ),
    version="1.0.0",
)

# CORS — allow Member A's frontend (React/Angular) and Member B's backend to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    question: str
    k: int = 5          # number of chunks to retrieve

class ChatResponse(BaseModel):
    answer: str
    sources: list
    user_id: str

class ProcessResponse(BaseModel):
    message: str
    user_id: str
    filename: str
    chunks_stored: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """
    Check whether the RAG service and Ollama are running.
    Returns Ollama model status.
    """
    ollama_status = check_ollama_connection()
    return {
        "rag_service": "running",
        "ollama": ollama_status,
        "collections": list_collections(),
    }


@app.post("/process", response_model=ProcessResponse, tags=["Document Processing"])
async def process_document(
    user_id: str = Form(..., description="Unique user ID from the auth system"),
    file: UploadFile = File(..., description="PDF, DOCX, or TXT file to upload"),
):
    """
    **Full RAG ingestion pipeline:**
    1. Save the uploaded file temporarily
    2. Load and parse the document
    3. Split into overlapping chunks
    4. Generate embeddings
    5. Store in ChromaDB (user-specific collection)

    Called by Member B's backend after a user uploads a document.
    """
    # Validate file extension
    allowed_ext = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_ext)}",
        )

    # Save file to uploads directory
    safe_filename = f"{user_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Load → Split → Embed → Store
        documents = load_document(file_path)
        chunks    = split_documents(documents)
        stored    = add_documents(user_id, chunks)
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return ProcessResponse(
        message=f"Document '{file.filename}' processed and stored successfully.",
        user_id=user_id,
        filename=file.filename,
        chunks_stored=stored,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    **RAG-powered Q&A:**
    1. Embed the user's question
    2. Retrieve the most relevant document chunks from ChromaDB
    3. Send context + question to Ollama (llama3.2)
    4. Return the grounded answer and source metadata

    Called by Member A's chat interface after the user types a question.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = retrieve_and_answer(
            user_id=request.user_id,
            question=request.question,
            k=request.k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}. Make sure Ollama is running."
        )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        user_id=request.user_id,
    )


@app.get("/collection/{user_id}", tags=["Vector Store"])
def get_user_collection_info(user_id: str):
    """Get the number of stored document chunks for a user."""
    return get_collection_info(user_id)


@app.delete("/collection/{user_id}", tags=["Vector Store"])
def delete_user_collection(user_id: str):
    """
    Delete all stored documents for a user.
    Use when the user wants to upload a new set of documents from scratch.
    """
    deleted = delete_collection(user_id)
    if deleted:
        return {"message": f"Collection for user '{user_id}' deleted.", "success": True}
    return {"message": f"No collection found for user '{user_id}'.", "success": False}


@app.get("/collections", tags=["Vector Store"])
def list_all_collections():
    """Admin: list all ChromaDB collections."""
    return {"collections": list_collections()}
