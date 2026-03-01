"""
conftest.py — Shared pytest fixtures for all test modules.

Provides:
    - mock_llm        : A mock LLM that returns deterministic responses
    - mock_indexer    : A pre-populated in-memory Indexer stub
    - tmp_docs_dir    : A temporary directory with sample documents
"""

import pytest

# TODO: implement fixtures


@pytest.fixture
def mock_llm():
    """Return a mock LLM with deterministic responses."""
    # TODO: implement
    pass


@pytest.fixture
def mock_indexer():
    """Return a mock Indexer with pre-loaded sample chunks."""
    # TODO: implement
    pass


@pytest.fixture
def tmp_docs_dir(tmp_path):
    """Create a temporary directory with sample .txt and .md files."""
    # TODO: implement
    return tmp_path
