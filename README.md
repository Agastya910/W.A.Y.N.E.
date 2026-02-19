# WAYNE

Webless Autonomous Neural Engine

**A customizable, offline-first alternative for AI-assisted coding.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Agastya910/W.A.Y.N.E?style=social)](https://github.com/Agastya910/W.A.Y.N.E)

---

## Overview

W.A.Y.N.E. is an open-source AI agent system designed to navigate, understand, and edit codebases entirely on your local machine, and by local machine I do not mean a high end gaming pc or mac studio with 40 GB gpu vram, but your regular pc (having even an entry level GPU with 4GB vram would work wonders, but not necessary, also works on CPU. )

Unlike cloud-based solutions, this runs on your hardware (via Ollama), ensuring that your code never leaves your device, and you own the models so can use them offline for free 24/7.

It is built to be "hackable"—allowing developers to modify the agent's logic, tools, and prompts to fit their specific workflow.

It serves as a private, offline alternative to tools like GitHub Copilot, Cursor, or Gemini CLI, prioritizing data ownership and customizability.

## Philosophy

WAYNE operates under three core principles:

1. **Webless** — No cloud dependency. Fully local.
2. **Autonomous** — Agentic execution with structured reasoning.
3. **Neural Engine** — Powered by local LLMs via Ollama.

Privacy is not a feature. It is the foundation.

---

## ⚡ Core Technical Features

### 1. Hybrid RAG Pipeline
Retrieval combines semantic (dense) and keyword (sparse/BM25) vectors using
Reciprocal Rank Fusion, then reranks the top candidates with FlashRank
(a 30MB ONNX model that runs fast on CPU). This means both "explain the
authentication flow" and "find all calls to parse_token" work reliably.

### 2. Code + Document Intelligence
WAYNE indexes your codebase automatically. You can also run `index documents`
to ingest PDFs, Word documents, or plain text files into the same search index.
Ask questions across everything — your code, your docs, your notes.

### 3. 3-Layer Memory Architecture
System context, compressed history, and a sliding window of recent turns.
Lets you have extended conversations without losing context, even on models
with small context windows.

### 4. State-Managed Undo + Self-Healing
Every file edit is reversible with "undo". The `fix` command runs your
Python file, captures errors, asks the LLM to patch them with structured
JSON output, and retries — up to 5 cycles. Every auto-fix is also undoable.

### 5. Fully Offline
Runs on Ollama (localhost). Works on CPU-only machines or entry-level GPUs
(4GB VRAM). No code ever leaves your device.

---

## 📸 Screenshots

### CLI startup and local indexing

![WAYNE CLI startup](assets/cli-startup.jpg)

### Cloning & analyzing a GitHub repo

![WAYNE analyzing Go repo](assets/cli-analysis.jpg)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai) — install and run locally
- Docker Desktop — for the Qdrant vector database

### 1. Pull models
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct-q4_0
```

### 2. Start the vector database
```bash
docker-compose up -d
```

### 3. Clone and install
```bash
git clone https://github.com/Agastya910/W.A.Y.N.E.git
cd W.A.Y.N.E.
pip install -r requirements.txt
```

### 4. Run
```bash
python cli.py .
```

### Example commands once running
- "Explain the authentication flow in auth.py"
- "Edit utils.py — replace deprecated_fn with new_fn"
- "Undo"
- "Fix app.py" — self-healing loop
- "Index documents" — ingest a folder of PDFs/docs

---

## 💡 Usage Examples

### Code Navigation

> _"Explain the authentication flow in `auth_service.py` and draw a mermaid diagram of how user sessions are created."_

### Refactoring & Editing

> _"Find all calls to `deprecated_function` in `src/` and replace them with `new_function`, passing `context=None` as the second argument."_

### Error Correction

> _"That edit broke the tests. Undo the last change to `utils.py`."_

---

## 🏗️ Architecture

The system follows a modular agentic loop:

```
graph TD
    User[User Input] --> Router{Query Router}

    Router -->|Question| Planner
    Router -->|Edit| EditEngine
    Router -->|Undo| Executor

    Planner -->|Needs Context| RAG[RAG Pipeline]
    RAG -->|Retrieve| VectorDB[(FAISS)]
    RAG -->|Re-rank| CrossEncoder

    Planner -->|Plan| Executor
    Executor -->|Tool Calls| Tools[FileIO / Git / Search]

    EditEngine -->|Preview| DiffViewer
    DiffViewer -->|Apply| FileSystem
    FileSystem -->|Log| History[ChatHistory (3-Layer)]
```

**Tech Stack:**

- **Orchestration**: Custom Agent Loop (Planner -> Executor -> Verifier)
- **Memory**: 3-Layer System
- **Retrieval**: FAISS + HyDE + Cross-Encoder
- **LLM**: Ollama (Compatible with Llama 3, Mistral, Qwen, etc.)

---

## 🛠️ Development & Customization

WAYNE is designed to be modified. You can adjust the agent's prompts, add new tools, or swap out the retrieval logic to suit your needs.

```bash
# Run tests
pytest tests/

# Format code
black .
```

## 🤝 Contributing

Contributions are welcome. Please submit a Pull Request.

## FAQ

**Q: Can I use WAYNE with any local model?**
A: Yes, WAYNE supports any model compatible with Ollama, giving you the flexibility to choose models that fit your hardware and performance needs.

**Q: Is my code really safe?**
A: Absolutely. WAYNE runs entirely on your local machine. No code ever leaves your device or is sent to the cloud.

**Q: What does the Y stand for?**
A: It stands for **YOU**, as you have the full control over your stuff.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
