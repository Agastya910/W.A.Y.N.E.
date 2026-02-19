import os
from dotenv import load_dotenv

load_dotenv()

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_0")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "wayne_index")
SPARSE_VECTOR_NAME = "sparse-text"

# Retrieval Configuration
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5

# Repo Scanner
IGNORE_DIRS = {
    "node_modules", ".git", "venv", "build", "dist", "__pycache__",
    "idea", ".vscode", ".repopilot_index", ".faiss_index", ".env"
}

MAX_FILE_SIZE = 1_000_000  # 1MB