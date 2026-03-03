# Architecture — MCP RAG Server

## Overview

The project is a **Corrective RAG MCP server** built on FastMCP + LangGraph.
It exposes five tools to any MCP-compatible client (Claude Desktop, VS Code
Copilot, MCP Inspector, etc.) and runs entirely locally — no paid APIs required.

```
MCP Client (IDE / Claude Desktop)
        │  stdio or streamable-http
        ▼
  ┌─────────────────────┐
  │   FastMCP server    │   src/server.py
  │   (5 tools)         │
  └────────┬────────────┘
           │
    ┌──────┴───────┐
    │              │
    ▼              ▼
Indexer       LangGraph RAG graph
(ChromaDB)    (src/graph.py)
    │              │
    ▼              ▼
 ChromaDB     LM Studio (LLM)
(in-process)  localhost:1234/v1
              (OpenAI-compatible)
```

---

## Corrective RAG Graph

The graph is defined in `src/graph.py` using LangGraph's `StateGraph`.
All decisions are made by local LLM calls; no heuristics or hard-coded rules.

```
[User Query]
      │
      ▼
rewrite_query          — keyword-rich query for better vector recall
      │
      ▼
  retrieve             — top-k cosine similarity search in ChromaDB
      │
      ▼
grade_chunks           — per-chunk LLM relevance scoring (yes / no)
      │
 ┌────┴──────────────────────────────────────┐
 │ ≥1 relevant chunk                         │ 0 relevant chunks
 ▼                                           ▼
generate               ◄──────   rewrite_query_retry  (max 2 loops)
      │
      ▼
hallucination_check    — LLM verifies answer is grounded in context
      │
 ┌────┴──────────────┐
 │ grounded           │ not grounded
 ▼                    ▼
[END — return answer]  generate_retry  (max 1 retry)
```

### State shape (`RAGState`)

| Field | Type | Description |
|---|---|---|
| `question` | `str` | Original user question |
| `rewritten_query` | `str` | Improved query for vector search |
| `documents` | `list[dict]` | Raw retrieved chunks |
| `graded_documents` | `list[dict]` | Chunks that passed the relevance grade |
| `context` | `str` | Formatted graded chunks as a single string |
| `generation` | `str` | LLM answer |
| `sources` | `list[str]` | De-duplicated source file paths |
| `retrieve_retry_count` | `int` | How many query-broadening loops occurred |
| `generate_retry_count` | `int` | How many re-generation attempts occurred |
| `is_grounded` | `bool` | Result of final hallucination check |

### Conditional edges

| After node | Condition | Next node |
|---|---|---|
| `grade_chunks` | ≥1 relevant chunk | `generate` |
| `grade_chunks` | 0 relevant + retries left | `rewrite_query_retry` |
| `grade_chunks` | 0 relevant + no retries left | `generate` (falls through) |
| `hallucination_check` | grounded | `END` |
| `hallucination_check` | not grounded + retries left | `generate_retry` |
| `hallucination_check` | not grounded + no retries left | `END` |

---

## MCP Tools

All tools are defined in `src/server.py` and registered with `@mcp.tool()`.

| Tool | Input | Output | Uses LLM? |
|---|---|---|---|
| `index_folder(folder_path, glob_pattern)` | path, glob | `{files_indexed, chunks_added, skipped_files, errors}` | No (embeddings only) |
| `ask_question(question)` | str | `{answer, sources, is_grounded, retrieve_retries, generate_retries}` | Yes — full RAG graph |
| `find_relevant_docs(query, top_k)` | str, int | `{results: [{text, source, chunk_index, distance}], count}` | No |
| `summarize_document(file_path)` | path | `{summary, filename}` | Yes — single LLM call |
| `index_status()` | — | `{total_chunks, files_count, indexed_files, last_indexed_at, collection_name, chroma_db_path}` | No |

Transport is selected via the `TRANSPORT` environment variable:
- `stdio` — default; used by Claude Desktop and MCP clients
- `streamable-http` — HTTP on `MCP_HOST:MCP_PORT`; used inside Docker

---

## Indexer (`src/indexer.py`)

### Supported formats

| Extension | Splitter |
|---|---|
| `.md .txt .rst .json .yaml .yml` | `RecursiveCharacterTextSplitter` (paragraph → line → sentence) |
| `.py` | `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` |
| `.js` | `RecursiveCharacterTextSplitter.from_language(Language.JS)` |
| `.ts` | `RecursiveCharacterTextSplitter.from_language(Language.TS)` |

### Chunk ID

Each chunk is assigned a deterministic MD5-based ID:

```
chunk_id = MD5("{source_path}::{chunk_index}")
```

This allows safe re-indexing (upsert) without accumulating duplicate chunks.

### Encoding detection

Files are read with `chardet` auto-detection, falling back to UTF-8 with error
replacement. This handles mixed-encoding corpora (e.g. legacy Cyrillic documents).

---

## Vector Store (`ChromaDB`)

- **Client**: `chromadb.PersistentClient` (in-process, no separate server)
- **Collection**: single collection named `rag_collection` (configurable)
- **Distance metric**: cosine similarity (`hnsw:space = cosine`)
- **Embedding model**: ChromaDB default (`all-MiniLM-L6-v2`, runs in-process)
- **Persistence path**: `./chroma_db` (configurable via `CHROMA_DB_PATH`)

---

## Prompts (`src/prompts.py`)

| Prompt | Input vars | Output format |
|---|---|---|
| `QUERY_REWRITE_PROMPT` | `{question}` | plain string — rewritten query |
| `CHUNK_GRADING_PROMPT` | `{question}`, `{document}` | `{"relevant": "yes"\|"no"}` |
| `GENERATION_PROMPT` | `{question}`, `{context}` | free-form answer with `(source: <file>)` citations |
| `HALLUCINATION_PROMPT` | `{generation}`, `{context}` | `{"grounded": "yes"\|"no"}` |
| `SUMMARIZATION_PROMPT` | `{filename}`, `{document}` | structured markdown summary |

All prompts are language-agnostic — the model is instructed to respond in the
same language as the question.

`strip_thinking_tags(text)` strips `<think>…</think>` blocks emitted by
reasoning models (DeepSeek R1, Qwen3-thinking, etc.) before the output is
passed to downstream pipeline steps. Controlled by `LLM_STRIP_THINKING_TAGS`.

---

## Configuration (`src/config.py`)

Every constant reads from an environment variable with a hardcoded default,
making Docker / CI overrides trivial without rebuilding the image.

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio (or any OpenAI-compatible) endpoint |
| `LLM_MODEL` | `deepseek/deepseek-r1-0528-qwen3-8b` | Model identifier shown in LM Studio |
| `LLM_API_KEY` | `lm-studio` | Arbitrary; LM Studio ignores the value |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output for RAG |
| `LLM_STRIP_THINKING_TAGS` | `true` | Strip `<think>` blocks from R1 models |
| `CHUNK_SIZE` | `600` | Max chars per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `rag_collection` | ChromaDB collection name |
| `TOP_K` | `5` | Chunks returned per retrieval |
| `MAX_RETRIEVE_RETRIES` | `2` | Max query-broadening loops |
| `MAX_GENERATE_RETRIES` | `1` | Max regeneration retries on hallucination |
| `TRANSPORT` | `stdio` | MCP transport (`stdio` or `streamable-http`) |
| `MCP_HOST` | `127.0.0.1` | Bind address for HTTP transport |
| `MCP_PORT` | `8000` | Port for HTTP transport |

---

## Project Structure

```
mcp-rag/
├── src/
│   ├── config.py        # All tunable parameters (env-var backed)
│   ├── indexer.py       # Document loading, chunking, ChromaDB upsert/query
│   ├── prompts.py       # 5 ChatPromptTemplate objects + strip_thinking_tags()
│   ├── graph.py         # LangGraph Corrective RAG pipeline
│   ├── server.py        # FastMCP server — 5 @mcp.tool() handlers
│   └── __init__.py
├── tests/
│   ├── conftest.py      # Shared fixtures: FakeLLM, mock_indexer, tmp_docs_dir
│   ├── test_indexer.py  # 12 tests — Indexer class
│   ├── test_graph.py    # 10 tests — graph nodes (all LLM calls mocked)
│   └── test_mcp_tools.py# 11 tests — MCP tool handlers
├── sample_docs/
│   ├── Tolkien_The_Hobbit.txt
│   ├── Later_edits.txt  # Project-specific "amendments" that override base text
│   ├── example.md
│   └── example.txt
├── results/
│   ├── ARCHITECTURE.md  # This file
│   └── REPORT.md
├── Dockerfile           # Two-stage build; LM Studio on host via host.docker.internal
├── docker-compose.yml   # Single mcp-rag service; requires LM Studio running on host
├── requirements.txt
├── .env.example
└── README.md
```
