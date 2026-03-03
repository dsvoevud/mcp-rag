# MCP RAG Server — Corrective RAG with LangGraph

An **MCP (Model Context Protocol) server** that exposes a **Corrective RAG** pipeline built with LangGraph, ChromaDB, and a local LLM via **LM Studio**. Index local documents, then query them through any MCP-compatible client (VS Code Copilot agent mode, Claude Desktop, etc.).

---

## Features

- **5 MCP Tools** — `index_folder`, `ask_question`, `find_relevant_docs`, `summarize_document`, `index_status`
- **Corrective RAG pipeline** — query rewriting, chunk grading, hallucination checking via LangGraph
- **Local-first** — runs fully offline with LM Studio (OpenAI-compatible API, no cloud key required)
- **ChromaDB** vector store — persistent, in-process, no separate server needed
- **Docker support** — spin up the entire stack with one command

---

## Requirements

### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| **Python** | **3.11.x** | Required. Python 3.12+ may work; 3.13/3.14 are **not supported** due to missing pre-built wheels for dependencies |
| **LM Studio** | Latest | For running local LLMs via OpenAI-compatible API on `localhost:1234` |
| **Docker & Docker Compose** | Latest | Optional, for containerised deployment |

> ⚠️ **Important:** This project requires **Python 3.11**. Python 3.13 and 3.14 (including the free-threaded `3.14t` variant) are not compatible because key dependencies (`chromadb`, `watchfiles`, `fastmcp`) do not yet provide pre-built wheels for those versions and require a Rust compiler to build from source.

### Python Packages

All dependencies are listed in [`requirements.txt`](requirements.txt):

```
fastmcp              # MCP server framework
langchain            # LLM orchestration
langchain-community
langchain-openai     # LM Studio / OpenAI-compatible LLM client
langchain-ollama     # Ollama LLM integration (optional / bonus embeddings)
langgraph            # Corrective RAG graph engine
chromadb             # Vector store (in-process, no server)
chardet              # File encoding auto-detection
unstructured         # Document loaders
pypdf                # PDF support
python-docx          # Word document support
python-dotenv        # Environment variable management
pyyaml               # YAML config support
pytest               # Testing
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

### 5. Configure LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model — this project uses **`deepseek/deepseek-r1-0528-qwen3-8b`** (8B reasoning model)
3. Start the local server on port `1234` (default): **Local Server tab → Start Server**
4. Verify `LLM_MODEL` in `src/config.py` matches the model identifier shown in LM Studio

> The server exposes an OpenAI-compatible API at `http://localhost:1234/v1`.
> No API key is required — `lm-studio` is used as a placeholder value.

> ⚠️ **DeepSeek R1 reasoning models** emit chain-of-thought inside `<think>...</think>` tags before the final answer. The pipeline automatically strips these tags (`LLM_STRIP_THINKING_TAGS = True` in `config.py`) so only the final answer is used in grading, hallucination checks, and responses.

### 6. (Optional) Configure Ollama Embeddings

For the bonus embedding task, install [Ollama](https://ollama.ai/) and pull:
```bash
ollama pull nomic-embed-text
```
Then set `EMBEDDING_PROVIDER = "ollama"` in `src/config.py`.

---

## Configuration

All parameters are centralised in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `LLM_MODEL` | `deepseek/deepseek-r1-0528-qwen3-8b` | Model name as shown in LM Studio |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output for RAG |
| `LLM_STRIP_THINKING_TAGS` | `True` | Strip `<think>` tags from R1 reasoning model output |
| `CHUNK_SIZE` | `600` | Tokens per chunk (tuned for Cyrillic + English prose) |
| `CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks (~13%) |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `TOP_K` | `5` | Chunks returned per retrieval query |
| `MAX_RETRIEVE_RETRIES` | `2` | Max query-broadening loops in the RAG graph |
| `MAX_GENERATE_RETRIES` | `1` | Max regeneration attempts on hallucination detection |
| `EMBEDDING_PROVIDER` | `default` | `default` (ChromaDB built-in) or `ollama` (bonus) |

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
