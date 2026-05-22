import os

# Base directory of your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vector DB persistent storage folder
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Temporary folder where document files are saved
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Member C's Text splitting configurations
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Local embedding pipeline model identifier
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local LLM connectivity parameters (Ollama)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K_RESULTS = 5