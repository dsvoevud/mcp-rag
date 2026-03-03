"""
test_indexer.py — Unit tests for src/indexer.py

Tests cover:
    - Indexing .txt and .md files produces non-empty chunks
    - Indexing .py files uses language-aware splitter (no crash)
    - Unsupported file types (.png) are silently skipped
    - Re-indexing the same file does not duplicate chunks (upsert semantics)
    - get_status() returns expected keys and chunk count
    - retrieve() returns results with required keys
    - index_folder() returns the expected summary structure
    - retrieve() on an empty collection returns an empty list
    - index_folder() with a too-narrow glob pattern indexes nothing
    - index_folder() result chunk count matches get_status() total_chunks
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import src.config as cfg
from src.indexer import Indexer


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def indexer(tmp_path):
    """Provide an Indexer that uses an isolated ChromaDB in tmp_path.

    cfg.CHROMA_DB_PATH and cfg.COLLECTION_NAME are patched for the full test
    scope so both __init__ and get_status() see the same values.
    """
    db_path = str(tmp_path / "chroma_db")
    with patch.object(cfg, "CHROMA_DB_PATH", db_path), \
         patch.object(cfg, "COLLECTION_NAME", "test_collection"):
        yield Indexer()


# ---------------------------------------------------------------------------
# index_folder tests
# ---------------------------------------------------------------------------

class TestIndexFolder:

    def test_txt_and_md_files_are_indexed(self, indexer, tmp_docs_dir):
        """At minimum hobbit.txt and notes.md should produce chunks."""
        result = indexer.index_folder(str(tmp_docs_dir))

        assert result["files_indexed"] >= 2
        assert result["chunks_added"] >= 1
        assert result["errors"] == []

    def test_unsupported_files_are_skipped(self, indexer, tmp_docs_dir):
        """image.png must appear in skipped_files, not errors."""
        result = indexer.index_folder(str(tmp_docs_dir))

        skipped = [str(p) for p in result["skipped_files"]]
        assert any("image.png" in s for s in skipped)
        assert result["errors"] == []

    def test_python_file_indexed_without_error(self, indexer, tmp_docs_dir):
        """Language-aware Python splitter must not crash."""
        result = indexer.index_folder(str(tmp_docs_dir), glob_pattern="**/*.py")

        assert result["errors"] == []
        assert result["files_indexed"] >= 1

    def test_reindexing_does_not_duplicate_chunks(self, indexer, tmp_docs_dir):
        """Upsert semantics: second index run must not increase chunk count."""
        indexer.index_folder(str(tmp_docs_dir))
        count_first = indexer.get_status()["total_chunks"]

        indexer.index_folder(str(tmp_docs_dir))
        count_second = indexer.get_status()["total_chunks"]

        assert count_second == count_first

    def test_result_contains_required_keys(self, indexer, tmp_docs_dir):
        result = indexer.index_folder(str(tmp_docs_dir))

        for key in ("files_indexed", "chunks_added", "skipped_files", "errors"):
            assert key in result, f"Missing key: {key}"

    def test_narrow_glob_matches_only_target_extension(self, indexer, tmp_docs_dir):
        """A *.txt glob should only index txt files."""
        result = indexer.index_folder(str(tmp_docs_dir), glob_pattern="**/*.txt")

        assert result["files_indexed"] >= 1
        for source in indexer.get_status()["indexed_files"]:
            assert source.endswith(".txt"), f"Non-txt file indexed: {source}"

    def test_chunk_count_matches_status(self, indexer, tmp_docs_dir):
        """chunks_added in result must equal total_chunks in get_status()."""
        result = indexer.index_folder(str(tmp_docs_dir))

        assert indexer.get_status()["total_chunks"] == result["chunks_added"]


# ---------------------------------------------------------------------------
# get_status tests
# ---------------------------------------------------------------------------

class TestGetStatus:

    def test_status_keys_present(self, indexer, tmp_docs_dir):
        indexer.index_folder(str(tmp_docs_dir))

        status = indexer.get_status()
        for key in ("total_chunks", "files_count", "indexed_files",
                    "collection_name", "chroma_db_path"):
            assert key in status, f"Missing key: {key}"

    def test_empty_index_has_zero_chunks(self, indexer):
        """A freshly created Indexer must report zero chunks."""
        assert indexer.get_status()["total_chunks"] == 0


# ---------------------------------------------------------------------------
# retrieve tests
# ---------------------------------------------------------------------------

class TestRetrieve:

    def test_retrieve_returns_list(self, indexer, tmp_docs_dir):
        indexer.index_folder(str(tmp_docs_dir))

        results = indexer.retrieve("hobbit adventure", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_results_have_required_keys(self, indexer, tmp_docs_dir):
        indexer.index_folder(str(tmp_docs_dir))

        results = indexer.retrieve("hobbit adventure", top_k=3)
        for item in results:
            for key in ("text", "source", "chunk_index", "distance"):
                assert key in item, f"Missing key: {key}"

    def test_retrieve_empty_collection_returns_empty_list(self, indexer):
        """retrieve() on an empty collection must return [] not raise."""
        assert indexer.retrieve("anything") == []
