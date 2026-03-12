"""
server.py — FastMCP server entrypoint.

Exposes the following MCP tools:
    - index_folder        : Index all documents in a folder into ChromaDB
    - ask_question        : Run the full Corrective RAG pipeline for a question
    - find_relevant_docs  : Retrieve top-k relevant chunks without generation
    - summarize_document  : Summarise a single file via LLM
    - index_status        : Return current vector index statistics

Usage:
    python src/server.py

    Or via MCP client configuration:
    {
        "type": "stdio",
        "command": "python",
        "args": ["src/server.py"],
        "cwd": "<project root>"
    }
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastmcp import FastMCP
from langchain_openai import ChatOpenAI

import src.config as cfg
from src.indexer import Indexer
from src.graph import run_graph
from src.prompts import SUMMARIZATION_PROMPT, strip_thinking_tags


def _is_connection_error(exc: BaseException) -> bool:
    """Return True if *exc* (or any cause) is a network connectivity failure."""
    import httpx, httpcore
    for e in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout,
                          httpcore.ConnectError, httpcore.ConnectTimeout,
                          ConnectionRefusedError)):
            return True
    return False


def _llm_connection_error_message() -> str:
    return (
        f"Cannot connect to LM Studio at {cfg.LLM_BASE_URL}. "
        "Please make sure LM Studio is running on the host machine, "
        "a model is loaded, and the Local Server is started "
        "(LM Studio → Local Server tab → Start Server). "
        f"Expected model: '{cfg.LLM_MODEL}'."
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_indexer: Indexer | None = None
_llm: ChatOpenAI | None = None


def _get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=cfg.LLM_BASE_URL,
            api_key=cfg.LLM_API_KEY,
            model=cfg.LLM_MODEL,
            temperature=cfg.LLM_TEMPERATURE,
        )
    return _llm


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="mcp-rag-server",
    instructions=(
        "A RAG knowledge base server. "
        "ALWAYS call index_status first to check whether documents are already indexed. "
        "If total_chunks is 0, call index_folder with the path to the documents directory "
        "before answering any questions. "
        "If ask_question returns a connection error, call llm_status to diagnose whether "
        "LM Studio is running and the model is loaded. "
        "Use ask_question to query indexed documents through the full Corrective RAG pipeline. "
        "Use find_relevant_docs for raw chunk retrieval without generation. "
        "Use summarize_document to summarise a single file. "
        "Use index_status to check how many chunks are indexed and when indexing last ran. "
        "Use llm_status to check whether the LLM backend is reachable."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1 — index_folder
# ---------------------------------------------------------------------------

@mcp.tool()
def index_folder(folder_path: str, glob_pattern: str = "**/*") -> dict:
    """Index all supported documents in a folder into the vector store.

    Supported formats: .md, .txt, .rst, .py, .js, .ts, .json, .yaml

    Args:
        folder_path: Absolute or relative path to the folder to index.
        glob_pattern: Glob pattern to filter files (default: all files recursively).

    Returns:
        Summary dict with files_indexed, chunks_added, skipped_files, errors.
    """
    logger.info("index_folder called: path=%s glob=%s", folder_path, glob_pattern)
    try:
        result = _get_indexer().index_folder(folder_path, glob_pattern)
        logger.info(
            "Indexed %d files, %d chunks, %d errors",
            result["files_indexed"],
            result["chunks_added"],
            len(result["errors"]),
        )
        return result
    except FileNotFoundError as exc:
        return {"error": str(exc), "files_indexed": 0, "chunks_added": 0}
    except Exception as exc:
        logger.exception("Unexpected error in index_folder")
        return {"error": f"Unexpected error: {exc}", "files_indexed": 0, "chunks_added": 0}


# ---------------------------------------------------------------------------
# Tool 2 — ask_question
# ---------------------------------------------------------------------------

@mcp.tool()
def ask_question(question: str) -> dict:
    """Ask a question against the indexed document collection.

    Runs the full Corrective RAG pipeline:
    rewrite → retrieve → grade → generate → hallucination_check

    Args:
        question: Natural language question in any language.

    Returns:
        Dict with keys:
            - answer (str): The generated answer with source citations.
            - sources (list[str]): Source file paths referenced in the answer.
            - is_grounded (bool): Whether the answer passed hallucination check.
            - retrieve_retries (int): Number of retrieval retry loops performed.
            - generate_retries (int): Number of generation retries performed.
    """
    logger.info("ask_question called: %s", question[:120])
    if _get_indexer().get_status()["total_chunks"] == 0:
        path_hint = (
            f" Call index_folder with path '{cfg.AUTO_INDEX_PATH}'."
            if cfg.AUTO_INDEX_PATH
            else " Call index_folder with the path to your documents directory."
        )
        return {
            "answer": f"The index is empty.{path_hint}",
            "sources": [],
            "is_grounded": False,
            "retrieve_retries": 0,
            "generate_retries": 0,
        }
    try:
        result = run_graph(question, indexer=_get_indexer())
        return {
            "answer": result["generation"],
            "sources": result["sources"],
            "is_grounded": result["is_grounded"],
            "retrieve_retries": result["retrieve_retry_count"],
            "generate_retries": result["generate_retry_count"],
        }
    except Exception as exc:
        logger.exception("Error in ask_question")
        if _is_connection_error(exc):
            answer = _llm_connection_error_message()
        else:
            answer = f"Error running RAG pipeline: {exc}"
        return {
            "answer": answer,
            "sources": [],
            "is_grounded": False,
            "retrieve_retries": 0,
            "generate_retries": 0,
        }


# ---------------------------------------------------------------------------
# Tool 3 — find_relevant_docs
# ---------------------------------------------------------------------------

@mcp.tool()
def find_relevant_docs(query: str, top_k: int = 5) -> dict:
    """Retrieve the most relevant document chunks for a query without generating an answer.

    Useful for inspecting what the index contains or for debugging retrieval quality.

    Args:
        query: Natural language search query.
        top_k: Maximum number of chunks to return (default: 5).

    Returns:
        Dict with key 'results': list of chunks, each with text, source,
        chunk_index, and distance (lower = more similar).
    """
    logger.info("find_relevant_docs called: query=%s top_k=%d", query[:80], top_k)
    if _get_indexer().get_status()["total_chunks"] == 0:
        return {"results": [], "message": "Index is empty. Run index_folder first."}
    try:
        chunks = _get_indexer().retrieve(query, top_k=top_k)
        return {"results": chunks, "count": len(chunks)}
    except Exception as exc:
        logger.exception("Error in find_relevant_docs")
        return {"results": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 4 — summarize_document
# ---------------------------------------------------------------------------

@mcp.tool()
def summarize_document(file_path: str) -> dict:
    """Summarise a single document file using the local LLM.

    The file is read and fed directly to the summarisation prompt —
    it does not need to be indexed first.

    Args:
        file_path: Absolute or relative path to the file to summarise.

    Returns:
        Dict with keys: summary (str), filename (str).
    """
    logger.info("summarize_document called: %s", file_path)
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return {"summary": "", "filename": str(path), "error": f"File not found: {path}"}

    try:
        from src.indexer import _read_file_with_encoding  # reuse encoding-aware reader
        content = _read_file_with_encoding(path)

        if not content.strip():
            return {"summary": "File is empty.", "filename": path.name}

        # Truncate very large files to avoid context overflow
        max_chars = cfg.CHUNK_SIZE * 10
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[... document truncated for summarisation ...]"

        chain = SUMMARIZATION_PROMPT | _get_llm()
        result = chain.invoke({"filename": path.name, "document": content})
        raw = result.content if hasattr(result, "content") else str(result)
        if cfg.LLM_STRIP_THINKING_TAGS:
            raw = strip_thinking_tags(raw)

        return {"summary": raw.strip(), "filename": path.name}
    except Exception as exc:
        logger.exception("Error in summarize_document")
        if _is_connection_error(exc):
            return {"summary": "", "filename": str(path), "error": _llm_connection_error_message()}
        return {"summary": "", "filename": str(path), "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 5 — index_status
# ---------------------------------------------------------------------------

@mcp.tool()
def index_status() -> dict:
    """Return statistics about the current state of the vector index.

    Returns:
        Dict with keys:
            - total_chunks (int): Total number of indexed chunks.
            - files_count (int): Number of unique indexed files.
            - indexed_files (list[str]): Paths of all indexed files.
            - last_indexed_at (str | None): ISO timestamp of last indexing run.
            - collection_name (str): ChromaDB collection name.
            - chroma_db_path (str): Path to the ChromaDB persistence directory.
    """
    logger.info("index_status called")
    return _get_indexer().get_status()


# ---------------------------------------------------------------------------
# Tool 6 — llm_status
# ---------------------------------------------------------------------------

@mcp.tool()
def llm_status() -> dict:
    """Check whether the LLM backend (LM Studio) is reachable and responding.

    Sends a minimal test prompt to the configured LLM endpoint.
    Use this to diagnose connection problems before calling ask_question.

    Returns:
        Dict with keys:
            - ok (bool): True if the LLM responded successfully.
            - model (str): Configured model name.
            - base_url (str): Configured LLM endpoint.
            - error (str | None): Error message if the LLM is unreachable.
    """
    logger.info("llm_status called")
    try:
        llm = _get_llm()
        resp = llm.invoke("Reply with the single word: OK")
        return {"ok": True, "model": cfg.LLM_MODEL, "base_url": cfg.LLM_BASE_URL, "error": None}
    except Exception as exc:
        if _is_connection_error(exc):
            msg = _llm_connection_error_message()
        else:
            msg = str(exc)
        logger.warning("llm_status: LLM unreachable: %s", msg)
        return {"ok": False, "model": cfg.LLM_MODEL, "base_url": cfg.LLM_BASE_URL, "error": msg}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the MCP server.

    Transport is selected via the TRANSPORT environment variable:
      - ``stdio``             (default) — for MCP client integration (e.g. Claude Desktop)
      - ``streamable-http``  — HTTP server on MCP_HOST:MCP_PORT, used inside Docker
      - ``sse``              — legacy SSE transport

    Relevant env vars:
      TRANSPORT   stdio | streamable-http | sse  (default: stdio)
      MCP_HOST    bind address for HTTP transport  (default: 127.0.0.1)
      MCP_PORT    port for HTTP transport          (default: 8000)
    """
    import os

    transport = os.environ.get("TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8000"))

    logger.info(
        "Starting MCP RAG server (model: %s, transport: %s)",
        cfg.LLM_MODEL, transport,
    )

    # Auto-index sample docs on startup if the collection is empty.
    # Controlled by AUTO_INDEX_PATH env var (set in docker-compose.yml).
    if cfg.AUTO_INDEX_PATH:
        from pathlib import Path as _Path
        _auto_path = _Path(cfg.AUTO_INDEX_PATH)
        if _auto_path.exists():
            if _get_indexer().get_status()["total_chunks"] == 0:
                logger.info("Auto-indexing '%s' (collection is empty) ...", _auto_path)
                _result = _get_indexer().index_folder(str(_auto_path))
                logger.info(
                    "Auto-index complete: %d files, %d chunks, %d errors",
                    _result["files_indexed"], _result["chunks_added"], len(_result["errors"]),
                )
            else:
                logger.info(
                    "Auto-index skipped: collection already has %d chunks.",
                    _get_indexer().get_status()["total_chunks"],
                )
        else:
            logger.warning("AUTO_INDEX_PATH '%s' does not exist — skipping auto-index.", _auto_path)

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport in ("streamable-http", "sse"):
        mcp.run(transport=transport, host=host, port=port)
    else:
        raise ValueError(
            f"Unknown TRANSPORT={transport!r}. "
            "Use 'stdio', 'streamable-http', or 'sse'."
        )


if __name__ == "__main__":
    main()

