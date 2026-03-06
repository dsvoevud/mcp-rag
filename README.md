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

## Docker Deployment

### Prerequisites

- LM Studio running on the host with your model loaded and the local server started
  (**Local Server tab → Start Server**, default port `1234`)
- Docker Desktop installed and running

### Start

```bash
docker compose up --build
```

The MCP server starts on `http://localhost:8000/mcp` (streamable-http transport).
The ChromaDB index is persisted in the `chroma_data` Docker volume between restarts.

### Override the model

To use a different model without editing files, set `LLM_MODEL` before starting:

```bash
# Windows PowerShell
$env:LLM_MODEL="your-model-identifier"; docker compose up -d

# bash / macOS / Linux
LLM_MODEL=your-model-identifier docker compose up -d
```

### Index documents and query

Once the container is running, use the MCP tools from your client (e.g. VS Code with `.vscode/mcp.json` pointing to `http://localhost:8000/mcp`):

1. `index_folder` → `/app/sample_docs` (or any mounted path)
2. `ask_question` → ask anything about the indexed content

### Stop

```bash
docker compose down
```

To also wipe the index and start fresh:

```bash
docker compose down; docker volume rm mcp-rag_chroma_data
```

### Run tests inside the container

```bash
docker compose exec mcp-rag pytest tests/ -v
```
---


---

## Sample Queries

The following queries were used to validate the RAG pipeline against the included
`sample_docs/` corpus (The Hobbit + `Later_edits.txt` fictional amendments).

### Project-specific overrides (`Later_edits.txt` beats the base corpus)

These queries demonstrate the core RAG use-case: project-specific documentation
overriding common knowledge.

**Query:** `What colour is Smaug?`
```
Smaug is black with bright pink polka dots.
(source: /app/sample_docs/Later_edits.txt)
```

**Query:** `Tell me about the Eagles' Leader`
```
The Lord of the Eagles, named Kevin, leads the group. He possesses a commercial
pilot's license issued by the Misty Mountains Aviation Authority. As part of his
community service agreement, he is obligated to rescue at least one wizard per
fiscal year. On days off he drives a Toyota Camry and enjoys playing sudoku.
(source: /app/sample_docs/Later_edits.txt)
```

### Factual retrieval from the Hobbit text

**Query:** `What did Thorin say to Bilbo before he died?`
```
"Farewell, good thief. I go now to the halls of waiting to sit beside my
fathers, until the world is renewed..."
"No! There is more in you of good than you know, child of the kindly West.
Some courage and some wisdom, blended in measure. If more of us valued food
and cheer and song above hoarded gold, it would be a merrier world.
But sad or merry, I must leave it now. Farewell!"
(source: Chapter_18_THE_RETURN_JOURNEY.txt)
```

**Query:** `Where is the travel to goblins' caves described?`
```
Chapter 4 — "Over Hill and Under Hill" covers the company's journey through
the Misty Mountains, their capture during a storm, and their descent into
the goblin tunnels.
(source: Chapter_04_OVER_HILL_AND_UNDER_HILL.txt)
```

### Debugging retrieval with `find_relevant_docs`

When `ask_question` returns "insufficient information", use `find_relevant_docs`
with more descriptive terms to inspect what the vector store actually retrieves:

```
# Instead of the user's question phrasing:
find_relevant_docs("travel to goblins caves")  # poor recall

# Try content-rich terms:
find_relevant_docs("goblins caves mountains captured dwarves hobbit")  # hits Ch.4
```
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
2. Load a model — this project uses **`lmstudio-community/qwen2.5-7b-instruct-1m`** (7B, 1M token context window)
3. Start the local server on port `1234` (default): **Local Server tab → Start Server**
4. Verify `LLM_MODEL` in `src/config.py` matches the model identifier shown in LM Studio

> The server exposes an OpenAI-compatible API at `http://localhost:1234/v1`.
> No API key is required — `lm-studio` is used as a placeholder value.

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
| `LLM_MODEL` | `lmstudio-community/qwen2.5-7b-instruct-1m` | Model name as shown in LM Studio |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output for RAG |
| `LLM_STRIP_THINKING_TAGS` | `False` | Set `True` only for DeepSeek R1-style reasoning models that emit `<think>` tags |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks (~12%) |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `TOP_K` | `10` | Chunks returned per retrieval query |
| `MAX_RETRIEVE_RETRIES` | `2` | Max query-broadening loops in the RAG graph |
| `MAX_GENERATE_RETRIES` | `1` | Max regeneration attempts on hallucination detection |
| `EMBEDDING_PROVIDER` | `default` | `default` (ChromaDB built-in) or `ollama` (bonus) |

---

## Usage

### Start the MCP Server (stdio, local)

```bash
python -m src.server
```

### Connect to VS Code Copilot Agent Mode

**Option A — stdio (local venv, no Docker):**

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "mcp-rag": {
      "type": "stdio",
      "command": "${workspaceFolder}/.venv/Scripts/python.exe",
      "args": ["-m", "src.server"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

**Option B — HTTP (Docker container running):**

```json
{
  "servers": {
    "mcp-rag": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
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

## Running Tests

**Locally (recommended):**

```bash
pytest tests/ -v
```

**Inside the Docker container:**

```bash
docker compose exec mcp-rag pytest tests/ -v
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
│   └── Hobbit/          # Hobbit chapters split by chapter for better source attribution
├── results/
│   ├── ARCHITECTURE.md  # Detailed graph & design documentation
│   └── REPORT.md        # Development history & lessons learned
├── .vscode/
│   └── mcp.json         # VS Code MCP client configuration
├── .venv/               # Python 3.11 virtual environment (not committed)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of the Corrective RAG graph, retrieval strategy, and MCP tool design.

---

## License

MIT
