"""
graph.py — LangGraph Corrective RAG pipeline.

Graph flow:
    User Query
        │
        ▼
    rewrite_query       — improve query for vector search
        │
        ▼
    retrieve            — fetch top-k chunks from ChromaDB
        │
        ▼
    grade_chunks        — filter irrelevant chunks
        │
    ┌───┴────────────────────────────────────────┐
    │ enough relevant                             │ too few relevant
    ▼                                             ▼
  generate              ←─────────────   rewrite_query (retry, max 2)
    │
    ▼
  hallucination_check
    │
  ┌─┴──────────────────┐
  │ grounded            │ not grounded (max 1 retry → generate)
  ▼                     ▼
  END                 generate (retry)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

import src.config as cfg
from src.indexer import Indexer
from src.prompts import (
    CHUNK_GRADING_PROMPT,
    GENERATION_PROMPT,
    HALLUCINATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    strip_thinking_tags,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised on first run_graph call)
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
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str                        # original user question
    rewritten_query: str                 # improved query for vector search
    documents: list[dict]                # raw retrieved chunks [{text, source, ...}]
    graded_documents: list[dict]         # only relevant chunks
    context: str                         # formatted context passed to generation
    generation: str                      # LLM-generated answer
    sources: list[str]                   # unique source file paths
    retrieve_retry_count: int            # number of retrieve loops so far
    generate_retry_count: int            # number of generation retries so far
    is_grounded: bool                    # result of hallucination check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke_llm(prompt_template, **kwargs) -> str:
    """Invoke an LLM chain and return the cleaned string output."""
    chain = prompt_template | _get_llm()
    result = chain.invoke(kwargs)
    raw = result.content if hasattr(result, "content") else str(result)
    if cfg.LLM_STRIP_THINKING_TAGS:
        raw = strip_thinking_tags(raw)
    return raw.strip()


def _parse_binary_json(text: str, key: str, fallback: str = "yes") -> str:
    """Extract a yes/no value from a JSON response.

    Tries strict JSON parsing first, then regex fallback, then returns the
    fallback value to ensure the pipeline does not crash on malformed output.
    """
    # Try strict JSON parse
    try:
        # Sometimes the model wraps JSON in ```json ... ``` fences
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(cleaned)
        return str(data.get(key, fallback)).lower()
    except (json.JSONDecodeError, AttributeError):
        pass

    # Regex fallback — look for the key followed by yes/no
    match = re.search(rf'"{key}"\s*:\s*"(yes|no)"', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Last resort: look for bare yes/no anywhere in the response
    lower = text.lower()
    if "yes" in lower:
        return "yes"
    if "no" in lower:
        return "no"

    logger.warning("Could not parse binary JSON for key '%s' from: %s — using fallback '%s'", key, text[:200], fallback)
    return fallback


def _format_context(chunks: list[dict]) -> tuple[str, list[str]]:
    """Format graded chunks into a context string and deduplicated sources list."""
    parts: list[str] = []
    sources: list[str] = []
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{chunk['text']}")
        if source not in sources:
            sources.append(source)
    return "\n\n---\n\n".join(parts), sources


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def rewrite_query(state: RAGState) -> RAGState:
    """Rewrite the question into a retrieval-optimised search query."""
    logger.debug("Node: rewrite_query | retry=%d", state.get("retrieve_retry_count", 0))
    rewritten = _invoke_llm(QUERY_REWRITE_PROMPT, question=state["question"])
    return {**state, "rewritten_query": rewritten}


def retrieve(state: RAGState) -> RAGState:
    """Fetch top-k chunks from ChromaDB using the rewritten query."""
    logger.debug("Node: retrieve | query=%s", state["rewritten_query"])
    docs = _get_indexer().retrieve(state["rewritten_query"])
    return {**state, "documents": docs}


def grade_chunks(state: RAGState) -> RAGState:
    """Score each retrieved chunk for relevance; keep only relevant ones."""
    logger.debug("Node: grade_chunks | chunks=%d", len(state["documents"]))
    relevant: list[dict] = []
    for chunk in state["documents"]:
        response = _invoke_llm(
            CHUNK_GRADING_PROMPT,
            question=state["question"],
            document=chunk["text"],
        )
        verdict = _parse_binary_json(response, key="relevant", fallback="yes")
        if verdict == "yes":
            relevant.append(chunk)

    context, sources = _format_context(relevant)
    return {**state, "graded_documents": relevant, "context": context, "sources": sources}


def generate(state: RAGState) -> RAGState:
    """Generate an answer from graded context chunks."""
    logger.debug("Node: generate | retry=%d", state.get("generate_retry_count", 0))
    answer = _invoke_llm(
        GENERATION_PROMPT,
        question=state["question"],
        context=state["context"],
    )
    return {**state, "generation": answer}


def hallucination_check(state: RAGState) -> RAGState:
    """Verify that the generated answer is grounded in the context."""
    logger.debug("Node: hallucination_check")
    response = _invoke_llm(
        HALLUCINATION_PROMPT,
        generation=state["generation"],
        context=state["context"],
    )
    verdict = _parse_binary_json(response, key="grounded", fallback="yes")
    return {**state, "is_grounded": verdict == "yes"}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _route_after_grading(state: RAGState) -> str:
    """Route after grade_chunks:
    - enough relevant docs → generate
    - too few + retries remaining → rewrite_query (broaden)
    - too few + retries exhausted → generate anyway (best-effort)
    """
    relevant_count = len(state.get("graded_documents", []))
    retry_count = state.get("retrieve_retry_count", 0)

    if relevant_count > 0:
        return "generate"
    if retry_count < cfg.MAX_RETRIEVE_RETRIES:
        logger.debug("No relevant chunks — retrying retrieval (%d/%d)", retry_count + 1, cfg.MAX_RETRIEVE_RETRIES)
        return "rewrite_query_retry"
    logger.warning("No relevant chunks after %d retries — generating best-effort answer", retry_count)
    return "generate"


def _route_after_hallucination_check(state: RAGState) -> str:
    """Route after hallucination_check:
    - grounded → END
    - not grounded + retry remaining → generate (regenerate)
    - not grounded + retries exhausted → END (return as-is with warning)
    """
    if state.get("is_grounded", True):
        return END
    retry_count = state.get("generate_retry_count", 0)
    if retry_count < cfg.MAX_GENERATE_RETRIES:
        logger.debug("Hallucination detected — regenerating (%d/%d)", retry_count + 1, cfg.MAX_GENERATE_RETRIES)
        return "generate_retry"
    logger.warning("Answer still not grounded after %d retries — returning best-effort answer", retry_count)
    return END


# ---------------------------------------------------------------------------
# Retry increment nodes (thin wrappers that bump counters before looping)
# ---------------------------------------------------------------------------

def _increment_retrieve_retry(state: RAGState) -> RAGState:
    return {**state, "retrieve_retry_count": state.get("retrieve_retry_count", 0) + 1}


def _increment_generate_retry(state: RAGState) -> RAGState:
    return {**state, "generate_retry_count": state.get("generate_retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    """Build and compile the Corrective RAG LangGraph graph.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    builder = StateGraph(RAGState)

    # Core nodes
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_chunks", grade_chunks)
    builder.add_node("generate", generate)
    builder.add_node("hallucination_check", hallucination_check)

    # Retry helper nodes
    builder.add_node("rewrite_query_retry", _increment_retrieve_retry)
    builder.add_node("generate_retry", _increment_generate_retry)

    # Entry point
    builder.set_entry_point("rewrite_query")

    # Linear edges
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "grade_chunks")
    builder.add_edge("generate", "hallucination_check")

    # Retry loop edges
    builder.add_edge("rewrite_query_retry", "rewrite_query")
    builder.add_edge("generate_retry", "generate")

    # Conditional: after grading
    builder.add_conditional_edges(
        "grade_chunks",
        _route_after_grading,
        {
            "generate": "generate",
            "rewrite_query_retry": "rewrite_query_retry",
        },
    )

    # Conditional: after hallucination check
    builder.add_conditional_edges(
        "hallucination_check",
        _route_after_hallucination_check,
        {
            END: END,
            "generate_retry": "generate_retry",
        },
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Compiled graph singleton
_graph = None


def run_graph(query: str, indexer: Indexer | None = None) -> dict:
    """Run the full Corrective RAG pipeline for a given query.

    Args:
        query: The user's natural language question.
        indexer: Optional Indexer instance (uses global singleton if not provided).

    Returns:
        dict with keys:
            - question (str)
            - generation (str)      — the final answer
            - sources (list[str])   — source file paths cited
            - is_grounded (bool)    — whether the answer passed hallucination check
            - retrieve_retry_count (int)
            - generate_retry_count (int)
    """
    global _graph, _indexer

    if indexer is not None:
        _indexer = indexer

    if _graph is None:
        _graph = build_graph()

    initial_state: RAGState = {
        "question": query,
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

    final_state = _graph.invoke(initial_state)

    return {
        "question": final_state["question"],
        "generation": final_state["generation"],
        "sources": final_state["sources"],
        "is_grounded": final_state.get("is_grounded", False),
        "retrieve_retry_count": final_state.get("retrieve_retry_count", 0),
        "generate_retry_count": final_state.get("generate_retry_count", 0),
    }

