"""
indexer.py — Document loading, chunking, and ChromaDB persistence.

Responsibilities:
    - Load documents from a folder (.md, .txt, .rst, .py, .js, .ts, .json, .yaml)
    - Auto-detect file encoding (handles UTF-8 BOM, cp1251, etc.)
    - Split into chunks using RecursiveCharacterTextSplitter (text)
      and RecursiveCharacterTextSplitter.from_language() (code)
    - Upsert chunks with metadata into ChromaDB (in-process, no server)
    - Provide retrieval and status interfaces for the RAG graph and MCP tools
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import chardet
import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

import src.config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-type routing
# ---------------------------------------------------------------------------

# Plain-text formats — use generic RecursiveCharacterTextSplitter
_TEXT_EXTENSIONS: set[str] = {".md", ".txt", ".rst", ".json", ".yaml", ".yml"}

# Code formats — use language-aware splitter
_CODE_EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
}

_ALL_SUPPORTED: set[str] = _TEXT_EXTENSIONS | set(_CODE_EXTENSION_TO_LANGUAGE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file_with_encoding(path: Path) -> str:
    """Read a file, auto-detecting its encoding with chardet."""
    raw = path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        logger.warning("Failed to decode %s as %s, falling back to utf-8 with replacement", path, encoding)
        return raw.decode("utf-8", errors="replace")


def _chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic chunk ID based on source path and position."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _make_splitter(extension: str) -> RecursiveCharacterTextSplitter:
    """Return the appropriate text splitter for the given file extension."""
    lang = _CODE_EXTENSION_TO_LANGUAGE.get(extension)
    if lang is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )


# Pattern that matches numbered-section amendment files (e.g. "FACT 1 —").
_SECTION_HEADER_RE = re.compile(r"(?=\n{1,2}FACT \d+)", re.IGNORECASE)


def _pre_split_sections(content: str, source: str) -> list[Document]:
    """Split structured amendment files on FACT N section boundaries.

    Uses a zero-width lookahead so each chunk retains its own "FACT N —" header,
    giving the embedding a focused, single-topic vector per fact.
    Only called when the file contains at least one "FACT N" header.
    """
    parts = _SECTION_HEADER_RE.split(content)
    return [
        Document(page_content=part.strip(), metadata={"source": source})
        for part in parts
        if part.strip()
    ]


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class Indexer:
    """Manages document ingestion and vector store operations.

    Uses ChromaDB in-process with its default embedding function
    (all-MiniLM-L6-v2). No external embedding server required.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=cfg.CHROMA_DB_PATH)
        self._collection = self._client.get_or_create_collection(
            name=cfg.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._last_indexed_at: datetime | None = None
        self._indexed_files: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _walk_all_files(root: Path, glob_pattern: str) -> list[Path]:
        """Return all files under *root*, including hidden files and dirs.

        When glob_pattern is the default ``**/*``, ``os.walk`` is used so that
        hidden entries (names starting with ``.``) are never skipped — unlike
        Python's ``pathlib.glob`` on 3.12 + which omits them unless
        ``include_hidden=True`` is passed.  For any custom pattern the
        standard ``Path.glob`` is used as a fallback.
        """
        if glob_pattern == "**/*":
            result: list[Path] = []
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    result.append(Path(dirpath) / filename)
            return result
        return [p for p in root.glob(glob_pattern) if p.is_file()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_folder(self, folder_path: str, glob_pattern: str = "**/*") -> dict:
        """Load all supported documents from folder_path and index them.

        Args:
            folder_path: Absolute or relative path to the folder to index.
            glob_pattern: Glob pattern to filter files (default: all files).

        Returns:
            dict with keys: files_indexed, chunks_added, skipped_files, errors.
        """
        root = Path(folder_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Folder not found: {root}")

        all_files = self._walk_all_files(root, glob_pattern)
        files = [p for p in all_files if p.suffix in _ALL_SUPPORTED]

        chunks_added = 0
        skipped: list[str] = []
        errors: list[str] = []

        for file_path in files:
            try:
                new_chunks = self._index_file(file_path)
                chunks_added += new_chunks
                self._indexed_files.append(str(file_path))
            except Exception as exc:
                logger.exception("Failed to index %s", file_path)
                errors.append(f"{file_path}: {exc}")

        for p in all_files:
            if p.suffix not in _ALL_SUPPORTED:
                skipped.append(str(p))

        self._last_indexed_at = datetime.now(timezone.utc)

        return {
            "files_indexed": len(files) - len(errors),
            "chunks_added": chunks_added,
            "skipped_files": skipped,
            "errors": errors,
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return top_k most relevant chunks for the query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return (defaults to config TOP_K).

        Returns:
            List of dicts with keys: text, source, chunk_index, distance.
        """
        k = top_k if top_k is not None else cfg.TOP_K

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict] = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": dist,
            })
        return chunks

    def get_status(self) -> dict:
        """Return current index statistics.

        Returns:
            dict with keys: total_chunks, indexed_files, last_indexed_at,
            collection_name, chroma_db_path.
        """
        return {
            "total_chunks": self._collection.count(),
            "indexed_files": list(dict.fromkeys(self._indexed_files)),  # deduplicated
            "files_count": len(dict.fromkeys(self._indexed_files)),
            "last_indexed_at": self._last_indexed_at.isoformat() if self._last_indexed_at else None,
            "collection_name": cfg.COLLECTION_NAME,
            "chroma_db_path": cfg.CHROMA_DB_PATH,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_file(self, file_path: Path) -> int:
        """Load, chunk, and upsert a single file. Returns number of chunks added."""
        content = _read_file_with_encoding(file_path)
        if not content.strip():
            logger.debug("Skipping empty file: %s", file_path)
            return 0

        splitter = _make_splitter(file_path.suffix)

        # Pre-split structured files (e.g. Later_edits.txt with FACT N sections)
        # so that each numbered section becomes its own document before the
        # size-based splitter runs.  Without this step, facts of ~300 chars each
        # get merged into 1200-char multi-topic chunks and lose retrieval focus.
        if _SECTION_HEADER_RE.search(content):
            pre_docs = _pre_split_sections(content, str(file_path))
            logger.debug("Pre-split %s into %d section docs", file_path.name, len(pre_docs))
        else:
            pre_docs = [Document(page_content=content, metadata={"source": str(file_path)})]

        chunks: list[Document] = splitter.split_documents(pre_docs)

        if not chunks:
            return 0

        ids = [_chunk_id(str(file_path), i) for i in range(len(chunks))]
        texts = [c.page_content for c in chunks]
        metadatas = [
            {
                "source": str(file_path),
                "chunk_index": i,
                "extension": file_path.suffix,
            }
            for i in range(len(chunks))
        ]

        # Upsert — safe to re-index the same file
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        logger.debug("Indexed %d chunks from %s", len(chunks), file_path)
        return len(chunks)
