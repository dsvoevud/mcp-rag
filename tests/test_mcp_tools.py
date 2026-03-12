"""
test_mcp_tools.py — Unit tests for MCP tool handler functions in src/server.py

All external dependencies (Indexer, graph, LLM) are patched so tests run
without a running LM Studio instance or persistent ChromaDB.

Tests cover:
    - index_folder returns a dict with files_indexed + chunks_added
    - index_folder propagates errors gracefully
    - ask_question returns required output keys
    - ask_question returns a helpful message when index is empty
    - ask_question returns a connection-error message on LLM connectivity failure
    - find_relevant_docs returns a list with expected chunk keys
    - find_relevant_docs with top_k respects the limit
    - summarize_document returns a non-empty string
    - summarize_document returns an error dict for missing file
    - index_status returns a dict containing total_chunks
    - index_status collection_name equals configured COLLECTION_NAME
    - llm_status returns ok=True when LLM responds
    - llm_status returns ok=False with error message on connection failure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import logging
import pytest

import src.server as server_module
from src import server as srv
from src.config import COLLECTION_NAME


# ---------------------------------------------------------------------------
# index_folder
# ---------------------------------------------------------------------------

class TestIndexFolderTool:

    def test_returns_success_dict(self, tmp_docs_dir, tmp_path):
        fake_result = {
            "files_indexed": 3,
            "chunks_added": 15,
            "skipped_files": [],
            "errors": [],
        }
        mock_idx = MagicMock()
        mock_idx.index_folder.return_value = fake_result

        with patch.object(server_module, "_get_indexer", return_value=mock_idx):
            result = srv.index_folder(str(tmp_docs_dir))

        assert result["files_indexed"] == 3
        assert result["chunks_added"] == 15
        assert result["errors"] == []

    def test_returns_error_dict_on_exception(self):
        mock_idx = MagicMock()
        mock_idx.index_folder.side_effect = RuntimeError("disk full")

        with patch.object(server_module, "_get_indexer", return_value=mock_idx):
            result = srv.index_folder("/non/existent")

        assert "error" in result
        assert "disk full" in result["error"]


# ---------------------------------------------------------------------------
# ask_question
# ---------------------------------------------------------------------------

class TestAskQuestionTool:

    def test_returns_required_keys(self, mock_indexer):
        graph_output = {
            "question": "Who is Bilbo?",
            "generation": "Bilbo is a hobbit.",
            "sources": ["hobbit.txt"],
            "is_grounded": True,
            "retrieve_retry_count": 0,
            "generate_retry_count": 0,
        }
        with (
            patch.object(server_module, "_get_indexer", return_value=mock_indexer),
            patch("src.server.run_graph", return_value=graph_output),
        ):
            result = srv.ask_question("Who is Bilbo?")

        for key in ("answer", "sources", "is_grounded",
                    "retrieve_retries", "generate_retries"):
            assert key in result, f"Missing key: {key}"

    def test_empty_index_returns_helpful_message(self):
        empty_mock = MagicMock()
        empty_mock.get_status.return_value = {"total_chunks": 0}

        with patch.object(server_module, "_get_indexer", return_value=empty_mock):
            result = srv.ask_question("Any question?")

        # Should not crash; should return a message indicating empty index
        assert "answer" in result or "error" in result


# ---------------------------------------------------------------------------
# find_relevant_docs
# ---------------------------------------------------------------------------

class TestFindRelevantDocsTool:

    def test_returns_dict_with_results_key(self, mock_indexer, sample_chunks):
        mock_indexer.retrieve.return_value = sample_chunks
        mock_indexer.get_status.return_value = {"total_chunks": len(sample_chunks)}

        with patch.object(server_module, "_get_indexer", return_value=mock_indexer):
            result = srv.find_relevant_docs("hobbit adventure")

        assert "results" in result
        assert len(result["results"]) == len(sample_chunks)

    def test_chunk_items_have_required_keys(self, mock_indexer, sample_chunks):
        mock_indexer.retrieve.return_value = sample_chunks
        mock_indexer.get_status.return_value = {"total_chunks": len(sample_chunks)}

        with patch.object(server_module, "_get_indexer", return_value=mock_indexer):
            result = srv.find_relevant_docs("hobbit")

        for item in result["results"]:
            for key in ("text", "source", "chunk_index", "distance"):
                assert key in item, f"Missing key: {key}"

    def test_top_k_is_forwarded(self, mock_indexer, sample_chunks):
        mock_indexer.retrieve.return_value = sample_chunks[:2]
        mock_indexer.get_status.return_value = {"total_chunks": len(sample_chunks)}

        with patch.object(server_module, "_get_indexer", return_value=mock_indexer):
            srv.find_relevant_docs("query", top_k=2)

        mock_indexer.retrieve.assert_called_once_with("query", top_k=2)


# ---------------------------------------------------------------------------
# summarize_document
# ---------------------------------------------------------------------------

class TestSummarizeDocumentTool:

    def test_returns_non_empty_summary(self, tmp_docs_dir):
        doc_path = str(tmp_docs_dir / "hobbit.txt")
        summary_text = "**File:** hobbit.txt\n**Summary:** A hobbit adventure story."

        from langchain_core.messages import AIMessage

        # Build a fake chain to replace SUMMARIZATION_PROMPT | llm
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = AIMessage(content=summary_text)
        fake_prompt = MagicMock()
        fake_prompt.__or__ = MagicMock(return_value=fake_chain)

        with (
            patch.object(server_module, "SUMMARIZATION_PROMPT", fake_prompt),
            patch.object(server_module, "_get_llm", return_value=MagicMock()),
        ):
            result = srv.summarize_document(doc_path)

        assert isinstance(result, dict)
        assert "summary" in result
        assert result["summary"]  # non-empty

    def test_missing_file_returns_error(self):
        result = srv.summarize_document("/no/such/file.txt")

        assert isinstance(result, dict)
        assert "error" in result


# ---------------------------------------------------------------------------
# index_status
# ---------------------------------------------------------------------------

class TestIndexStatusTool:

    def test_returns_dict_with_total_chunks(self, mock_indexer):
        with patch.object(server_module, "_get_indexer", return_value=mock_indexer):
            result = srv.index_status()

        assert "total_chunks" in result

    def test_collection_name_matches_config(self, mock_indexer):
        with patch.object(server_module, "_get_indexer", return_value=mock_indexer):
            result = srv.index_status()

        assert result["collection_name"] == COLLECTION_NAME


# ---------------------------------------------------------------------------
# llm_status
# ---------------------------------------------------------------------------

class TestLlmStatusTool:

    def test_returns_ok_true_when_llm_responds(self, caplog):
        """LM Studio is running and the model is loaded — llm_status must return ok=True."""
        from langchain_core.messages import AIMessage
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="OK")

        with caplog.at_level(logging.INFO, logger="__main__"):
            with patch.object(server_module, "_get_llm", return_value=mock_llm):
                result = srv.llm_status()

        assert result["ok"] is True
        assert result["error"] is None
        assert "model" in result
        assert "base_url" in result

    def test_returns_ok_false_on_connection_error(self, caplog):
        """LM Studio is NOT running — llm_status must return ok=False with a
        human-readable error that tells the user to start LM Studio."""
        import httpx
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = httpx.ConnectError("Connection refused")

        with caplog.at_level(logging.WARNING, logger="__main__"):
            with patch.object(server_module, "_get_llm", return_value=mock_llm):
                result = srv.llm_status()

        # Confirm the warning was logged
        assert any("LLM unreachable" in r.message or "unreachable" in r.message.lower()
                   for r in caplog.records), (
            "Expected a warning log indicating LM Studio is unreachable"
        )
        assert result["ok"] is False
        assert result["error"]
        assert "LM Studio" in result["error"], (
            "Error message should instruct the user to start LM Studio, "
            f"got: {result['error']!r}"
        )

    def test_ask_question_returns_connection_message_on_llm_error(self, mock_indexer, caplog):
        """When LM Studio is NOT running, ask_question must return a clear message
        instead of the raw 'Connection error.' exception text."""
        import httpx
        with caplog.at_level(logging.ERROR, logger="__main__"):
            with (
                patch.object(server_module, "_get_indexer", return_value=mock_indexer),
                patch("src.server.run_graph", side_effect=httpx.ConnectError("refused")),
            ):
                result = srv.ask_question("What is Bilbo's name?")

        # Confirm the error was logged
        assert any("Error in ask_question" in r.message for r in caplog.records), (
            "Expected an error log for the ask_question connection failure"
        )
        assert result["is_grounded"] is False
        assert "LM Studio" in result["answer"], (
            "Answer should instruct the user to start LM Studio, "
            f"got: {result['answer']!r}"
        )
