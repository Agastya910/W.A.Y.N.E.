import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b-instruct-q4_0")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Repo Scanner
IGNORE_DIRS = {
    "node_modules", ".git", "venv", "build", "dist", "__pycache__",
    ".idea", ".vscode", ".repopilot_index", ".faiss_index", ".env"
}

MAX_FILE_SIZE = 1_000_000  # 1MB

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")