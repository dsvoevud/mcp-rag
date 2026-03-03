"""
prompts.py — Prompt templates for the Corrective RAG pipeline.

Templates (all use LangChain ChatPromptTemplate):
    - QUERY_REWRITE_PROMPT    : Rewrite a user query for better vector retrieval
    - CHUNK_GRADING_PROMPT    : Score a retrieved chunk for relevance (yes/no)
    - GENERATION_PROMPT       : Generate a grounded answer from relevant context
    - HALLUCINATION_PROMPT    : Verify the answer is supported by context (yes/no)
    - SUMMARIZATION_PROMPT    : Summarise a single document or set of chunks

Notes:
    - Prompts are designed for instruction-following models (DeepSeek R1, Qwen3, etc.)
    - Grading and hallucination checks request structured binary output to avoid
      ambiguous free-form responses.
    - Thinking tags (<think>...</think>) from reasoning models are stripped in
      graph.py via the strip_thinking_tags() utility defined below.
    - Prompts are language-agnostic: the model is expected to respond in the same
      language as the user question.
"""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Utility: strip DeepSeek R1 / reasoning model thinking tags
# ---------------------------------------------------------------------------

def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. DeepSeek R1).

    Strips the entire block including surrounding whitespace so the caller
    receives only the final answer.

    Args:
        text: Raw LLM output, potentially containing <think> blocks.

    Returns:
        Cleaned text with all <think>...</think> sections removed.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# 1. Query Rewrite
# ---------------------------------------------------------------------------
# Input variables: {question}
# Expected output: a single rewritten query string, no preamble.

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a search query optimiser. Your task is to rewrite the user's "
            "question into a concise, keyword-rich search query that will find the most "
            "relevant passages in a vector database.\n\n"
            "Rules:\n"
            "- Output ONLY the rewritten query. No explanation, no preamble.\n"
            "- Preserve the original language of the question.\n"
            "- Remove conversational filler (please, can you, etc.).\n"
            "- Expand abbreviations if helpful.\n"
            "- Keep the query under 30 words."
        ),
    ),
    ("human", "Original question: {question}"),
])


# ---------------------------------------------------------------------------
# 2. Chunk Grading
# ---------------------------------------------------------------------------
# Input variables: {question}, {document}
# Expected output: JSON object {"relevant": "yes"} or {"relevant": "no"}

CHUNK_GRADING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a relevance grader. Assess whether the provided document chunk "
            "contains information that helps answer the user's question.\n\n"
            "Rules:\n"
            "- Output ONLY valid JSON in this exact format: {{\"relevant\": \"yes\"}} "
            "or {{\"relevant\": \"no\"}}.\n"
            "- Do not explain your reasoning.\n"
            "- Be generous: if the chunk partially addresses the question or provides "
            "useful background, mark it as relevant."
        ),
    ),
    (
        "human",
        "Question: {question}\n\nDocument chunk:\n{document}",
    ),
])


# ---------------------------------------------------------------------------
# 3. Answer Generation
# ---------------------------------------------------------------------------
# Input variables: {question}, {context}
# Expected output: a well-formed answer grounded in the provided context.

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful assistant that answers questions strictly based on the "
            "provided context excerpts from a document collection.\n\n"
            "Rules:\n"
            "- Answer ONLY using information present in the context.\n"
            "- If the context does not contain enough information to answer, say: "
            "'The available documents do not contain sufficient information to answer "
            "this question.'\n"
            "- Do NOT add information from your own knowledge.\n"
            "- Cite the source file for each key claim using the format: (source: <filename>).\n"
            "- Respond in the same language as the question.\n"
            "- Be concise and precise."
        ),
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}",
    ),
])


# ---------------------------------------------------------------------------
# 4. Hallucination Check
# ---------------------------------------------------------------------------
# Input variables: {generation}, {context}
# Expected output: JSON object {"grounded": "yes"} or {"grounded": "no"}

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a factual grounding verifier. Your task is to check whether "
            "every factual claim in the generated answer is directly supported by "
            "the provided context.\n\n"
            "Rules:\n"
            "- Output ONLY valid JSON in this exact format: {{\"grounded\": \"yes\"}} "
            "or {{\"grounded\": \"no\"}}.\n"
            "- 'yes' means ALL claims are traceable to the context.\n"
            "- 'no' means at least one claim is not supported by the context.\n"
            "- Do not explain your reasoning."
        ),
    ),
    (
        "human",
        "Context:\n{context}\n\nGenerated answer:\n{generation}",
    ),
])


# ---------------------------------------------------------------------------
# 5. Summarisation
# ---------------------------------------------------------------------------
# Input variables: {document}, {filename}
# Expected output: a structured summary of the document.

SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a document summariser. Produce a clear, structured summary of "
            "the provided document content.\n\n"
            "Format your summary as follows:\n"
            "**File:** <filename>\n"
            "**Summary:** 2–4 sentences describing the main content and purpose.\n"
            "**Key topics:** bullet list of 3–6 main topics or themes covered.\n\n"
            "Rules:\n"
            "- Base the summary only on the provided text.\n"
            "- Be concise. Do not repeat the same point twice.\n"
            "- Respond in the same language as the majority of the document text."
        ),
    ),
    (
        "human",
        "Filename: {filename}\n\nDocument content:\n{document}",
    ),
])

