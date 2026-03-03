"""
test_graph.py — Unit tests for the LangGraph Corrective RAG graph nodes.

All tests patch the LLM and Indexer singletons so no network calls are made.

Tests cover:
    - rewrite_query node produces a non-empty string
    - grade_chunks marks chunks relevant / irrelevant based on LLM output
    - grade_chunks removes chunks graded irrelevant=yes
    - generate node produces a non-empty answer
    - hallucination_check marks grounded answers as is_grounded=True
    - hallucination_check marks ungrounded answers as is_grounded=False
    - run_graph() returns all expected output keys
    - run_graph() returns a non-empty generation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

import src.graph as graph_module
from src.graph import (
    RAGState,
    rewrite_query,
    retrieve,
    grade_chunks,
    generate,
    hallucination_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> RAGState:
    state: RAGState = {
        "question": "Who is Bilbo Baggins?",
        "rewritten_query": "",
        "documents": [],
        "graded_documents": [],
        "context": "",
        "generation": "",
        "sources": [],
        "retrieve_retry_count": 0,
        "generate_retry_count": 0,
        "is_grounded": False,
    }
    state.update(overrides)
    return state


def _fake_llm_for(response: str):
    """Return a callable that mimics (prompt | llm).invoke() returning *response*."""
    chain = MagicMock()
    chain.invoke.return_value = AIMessage(content=response)
    return chain


# ---------------------------------------------------------------------------
# rewrite_query node
# ---------------------------------------------------------------------------

class TestRewriteQueryNode:

    def test_rewritten_query_is_non_empty(self):
        state = _base_state()
        chain = _fake_llm_for("Bilbo Baggins hobbit Shire")

        with patch.object(graph_module, "_invoke_llm", return_value="Bilbo Baggins hobbit Shire"):
            result = rewrite_query(state)

        assert "rewritten_query" in result
        assert len(result["rewritten_query"]) > 0

    def test_rewritten_query_differs_from_original(self):
        state = _base_state(question="hobbit")
        with patch.object(graph_module, "_invoke_llm", return_value="Bilbo Baggins Shire Middle-earth"):
            result = rewrite_query(state)

        assert result["rewritten_query"] != state["question"]


# ---------------------------------------------------------------------------
# grade_chunks node
# ---------------------------------------------------------------------------

class TestGradeChunksNode:

    def test_relevant_chunks_are_kept(self, sample_chunks):
        state = _base_state(documents=sample_chunks, rewritten_query="hobbit")

        with patch.object(graph_module, "_invoke_llm", return_value='{"relevant": "yes"}'):
            result = grade_chunks(state)

        assert len(result["graded_documents"]) == len(sample_chunks)

    def test_irrelevant_chunks_are_removed(self, sample_chunks):
        state = _base_state(documents=sample_chunks, rewritten_query="unrelated topic")

        with patch.object(graph_module, "_invoke_llm", return_value='{"relevant": "no"}'):
            result = grade_chunks(state)

        assert result["graded_documents"] == []


# ---------------------------------------------------------------------------
# generate node
# ---------------------------------------------------------------------------

class TestGenerateNode:

    def test_generate_produces_non_empty_answer(self, sample_chunks):
        answer = "Bilbo Baggins is the protagonist of The Hobbit. (source: hobbit.txt)"
        state = _base_state(
            graded_documents=sample_chunks,
            rewritten_query="Who is Bilbo Baggins?",
        )

        with patch.object(graph_module, "_invoke_llm", return_value=answer):
            result = generate(state)

        assert result["generation"]
        assert len(result["generation"]) > 0

    def test_generate_populates_sources(self, sample_chunks):
        answer = "Bilbo Baggins. (source: hobbit.txt)"
        state = _base_state(
            graded_documents=sample_chunks,
            rewritten_query="hobbit?",
        )

        with patch.object(graph_module, "_invoke_llm", return_value=answer):
            result = generate(state)

        assert isinstance(result["sources"], list)


# ---------------------------------------------------------------------------
# hallucination_check node
# ---------------------------------------------------------------------------

class TestHallucinationCheckNode:

    def test_grounded_answer_sets_is_grounded_true(self, sample_chunks):
        state = _base_state(
            graded_documents=sample_chunks,
            generation="Bilbo Baggins lives in the Shire.",
        )

        with patch.object(graph_module, "_invoke_llm", return_value='{"grounded": "yes"}'):
            result = hallucination_check(state)

        assert result["is_grounded"] is True

    def test_hallucinated_answer_sets_is_grounded_false(self, sample_chunks):
        state = _base_state(
            graded_documents=sample_chunks,
            generation="Bilbo Baggins is a space explorer.",
        )

        with patch.object(graph_module, "_invoke_llm", return_value='{"grounded": "no"}'):
            result = hallucination_check(state)

        assert result["is_grounded"] is False


# ---------------------------------------------------------------------------
# run_graph integration (fully mocked)
# ---------------------------------------------------------------------------

class TestRunGraph:

    def test_run_graph_returns_required_keys(self, mock_indexer):
        with (
            patch.object(graph_module, "_get_indexer", return_value=mock_indexer),
            patch.object(graph_module, "_invoke_llm") as mock_invoke,
        ):
            # Wire responses in sequence: rewrite → grade → generate → hallucination
            mock_invoke.side_effect = [
                "Bilbo Baggins dragon",          # rewrite_query
                '{"relevant": "yes"}',           # grade chunk 1
                '{"relevant": "yes"}',           # grade chunk 2
                '{"relevant": "yes"}',           # grade chunk 3
                "Bilbo is a hobbit. (source: hobbit.txt)",  # generate
                '{"grounded": "yes"}',           # hallucination_check
            ]

            output = graph_module.run_graph("Who is Bilbo?", indexer=mock_indexer)

        for key in ("question", "generation", "sources", "is_grounded",
                    "retrieve_retry_count", "generate_retry_count"):
            assert key in output, f"Missing key: {key}"

    def test_run_graph_generation_is_non_empty(self, mock_indexer):
        with (
            patch.object(graph_module, "_get_indexer", return_value=mock_indexer),
            patch.object(graph_module, "_invoke_llm") as mock_invoke,
        ):
            mock_invoke.side_effect = [
                "search query",
                '{"relevant": "yes"}',
                '{"relevant": "yes"}',
                '{"relevant": "yes"}',
                "The answer is 42. (source: example.txt)",
                '{"grounded": "yes"}',
            ]

            output = graph_module.run_graph("Question?", indexer=mock_indexer)

        assert output["generation"]
    assert True
