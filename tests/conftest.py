"""
conftest.py — Shared pytest fixtures for all test modules.

Provides:
    - mock_llm            : Deterministic fake LLM (no network calls)
    - mock_indexer        : In-memory Indexer stub with pre-loaded chunks
    - tmp_docs_dir        : Temporary directory with .txt, .md, .py, .json files
    - sample_chunks       : Reusable list of fake retrieved chunks
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

class FakeLLM:
    """Deterministic fake ChatOpenAI that never calls the network.

    Responses are keyed by substrings found in the prompt content.
    Falls back to a generic answer if no key matches.
    """

    _RESPONSES: dict[str, str] = {
        # query rewrite
        "search query optimiser": "Bilbo Baggins dragon treasure Lonely Mountain",
        # chunk grading — relevant by default
        "relevance grader": '{"relevant": "yes"}',
        # generation
        "helpful assistant": "Bilbo Baggins is the protagonist. (source: hobbit.txt)",
        # hallucination check — grounded by default
        "factual grounding verifier": '{"grounded": "yes"}',
        # summarisation
        "document summariser": "**File:** hobbit.txt\n**Summary:** A hobbit's adventure.\n**Key topics:**\n- Adventure\n- Dragons",
    }

    def invoke(self, messages) -> AIMessage:
        # Flatten all message content to scan for keywords
        text = " ".join(
            m.content if hasattr(m, "content") else str(m)
            for m in (messages if isinstance(messages, list) else [messages])
        ).lower()

        for keyword, response in self._RESPONSES.items():
            if keyword.lower() in text:
                return AIMessage(content=response)

        return AIMessage(content="Generic answer for testing purposes.")

    def __or__(self, other):
        """Support `prompt | llm` chain syntax used in prompts.py."""
        return _FakeChain(self)


class _FakeChain:
    """Wraps FakeLLM to support .invoke({...}) after a prompt template."""

    def __init__(self, llm: FakeLLM) -> None:
        self._llm = llm

    def invoke(self, input_dict: dict) -> AIMessage:
        # Build a fake message list from the dict values so FakeLLM can match
        combined = " ".join(str(v) for v in input_dict.values())
        return self._llm.invoke([AIMessage(content=combined)])


@pytest.fixture
def mock_llm() -> FakeLLM:
    """Return a deterministic fake LLM — no network, no API key needed."""
    return FakeLLM()


# ---------------------------------------------------------------------------
# Sample chunks
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "text": "Bilbo Baggins is a hobbit who lives in the Shire.",
            "source": "sample_docs/hobbit.txt",
            "chunk_index": 0,
            "distance": 0.1,
        },
        {
            "text": "The dragon Smaug guards treasure in the Lonely Mountain.",
            "source": "sample_docs/hobbit.txt",
            "chunk_index": 1,
            "distance": 0.2,
        },
        {
            "text": "Gandalf is a wizard who helps the company of dwarves.",
            "source": "sample_docs/hobbit.txt",
            "chunk_index": 2,
            "distance": 0.3,
        },
    ]


# ---------------------------------------------------------------------------
# Mock Indexer
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_indexer(sample_chunks) -> MagicMock:
    """Return a mock Indexer with pre-loaded sample chunks."""
    indexer = MagicMock()
    indexer.retrieve.return_value = sample_chunks
    indexer.index_folder.return_value = {
        "files_indexed": 2,
        "chunks_added": 10,
        "skipped_files": [],
        "errors": [],
    }
    indexer.get_status.return_value = {
        "total_chunks": 10,
        "files_count": 2,
        "indexed_files": ["sample_docs/hobbit.txt", "sample_docs/example.md"],
        "last_indexed_at": "2026-03-02T10:00:00+00:00",
        "collection_name": "rag_collection",
        "chroma_db_path": "./chroma_db",
    }
    return indexer


# ---------------------------------------------------------------------------
# Temporary documents directory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with varied supported file types."""
    (tmp_path / "hobbit.txt").write_text(
        "Bilbo Baggins is a hobbit of the Shire.\n\nHe went on an unexpected journey.",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text(
        "# Project Notes\n\nThis project uses LangGraph for Corrective RAG.\n",
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        '{"model": "deepseek-r1", "top_k": 5}',
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text(
        "def greet(name: str) -> str:\n    return f'Hello, {name}'\n",
        encoding="utf-8",
    )
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")  # unsupported — should be skipped
    return tmp_path

