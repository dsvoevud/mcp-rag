# MCP RAG Server — Corrective RAG with LangGraph

An **MCP (Model Context Protocol) server** that exposes a **Corrective RAG** pipeline built with LangGraph, ChromaDB, and Ollama. Index local documents, then query them through any MCP-compatible client (VS Code Copilot agent mode, Claude Desktop, etc.).

---

## Features

- **5 MCP Tools** — `index_folder`, `ask_question`, `find_relevant_docs`, `summarize_document`, `index_status`
- **Corrective RAG pipeline** — query rewriting, chunk grading, hallucination checking via LangGraph
- **Local-first** — runs fully offline with Ollama LLMs (no OpenAI key required)
- **ChromaDB** vector store — persistent, fast, and embeddable
- **Docker support** — spin up the entire stack with one command

---

## Requirements

### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| **Python** | **3.11.x** | Required. Python 3.12+ may work; 3.13/3.14 are **not supported** due to missing pre-built wheels for dependencies |
| **Ollama** | Latest | For running local LLMs |
| **Docker & Docker Compose** | Latest | Optional, for containerised deployment |

> ⚠️ **Important:** This project requires **Python 3.11**. Python 3.13 and 3.14 (including the free-threaded `3.14t` variant) are not compatible because key dependencies (`chromadb`, `watchfiles`, `fastmcp`) do not yet provide pre-built wheels for those versions and require a Rust compiler to build from source.

### Python Packages

All dependencies are listed in [`requirements.txt`](requirements.txt):

```
fastmcp          # MCP server framework
langchain        # LLM orchestration
langchain-community
langchain-ollama # Ollama LLM integration
langgraph        # Corrective RAG graph engine
chromadb         # Vector store
unstructured     # Document loaders
pypdf            # PDF support
python-docx      # Word document support
python-dotenv    # Environment variable management
pyyaml           # YAML config support
pytest           # Testing
pytest-asyncio
pytest-mock
```

---

## Installation

### 1. Install Python 3.11

Download from [python.org](https://www.python.org/downloads/release/python-3119/) or via `winget`:

```powershell
winget install Python.Python.3.11
```

### 2. Clone the Repository

```bash
git clone <repo-url>
cd mcp-rag
```

### 3. Create a Virtual Environment

```powershell
# Windows
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install and Start Ollama

```bash
# Pull a supported model (choose one)
ollama pull phi3:mini       # Phi-3 Mini 3.8B (recommended)
ollama pull qwen2.5:3b      # Qwen 2.5 3B
```

---

## Usage

### Start the MCP Server

```bash
python src/server.py
```

### Connect to VS Code Copilot Agent Mode

Add the following to your VS Code `settings.json` or `.vscode/mcp.json`:

```json
{
  "mcp": {
    "servers": {
      "rag-server": {
        "type": "stdio",
        "command": "python",
        "args": ["src/server.py"]
      }
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|---|---|
| `index_folder` | Index all documents in a given folder path |
| `ask_question` | Ask a question — triggers the full Corrective RAG pipeline |
| `find_relevant_docs` | Retrieve top-k relevant chunks for a query |
| `summarize_document` | Summarise a specific indexed document |
| `index_status` | Show the current state of the vector index |

---

## Docker Deployment

```bash
docker-compose up --build
```

This starts the MCP server and Ollama together. The model is pulled automatically on first run.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
mcp-rag/
├── .github/
│   └── copilot-instructions.md
├── src/
│   ├── server.py        # FastMCP server & tool definitions
│   ├── config.py        # Configuration parameters
│   ├── indexer.py       # Document loading, chunking, ChromaDB
│   ├── graph.py         # LangGraph Corrective RAG pipeline
│   └── prompts.py       # Prompt templates
├── tests/
│   ├── test_indexer.py
│   ├── test_graph.py
│   └── test_mcp_tools.py
├── sample_docs/
├── .venv/               # Python 3.11 virtual environment (not committed)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
├── ARCHITECTURE.md
└── REPORT.md
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of the Corrective RAG graph, retrieval strategy, and MCP tool design.

---

## License

MIT
