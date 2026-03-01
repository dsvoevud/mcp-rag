# Architecture

> TODO: fill in once implementation is complete.

## Overview

<!-- Describe the high-level architecture here -->

## Corrective RAG Graph

<!-- Insert LangGraph diagram here -->

```
[User Query]
     │
     ▼
rewrite_query
     │
     ▼
  retrieve
     │
     ▼
grade_chunks ──(all irrelevant)──► rewrite_query (retry)
     │
  (relevant)
     │
     ▼
  generate
     │
     ▼
hallucination_check ──(fail)──► generate (retry)
     │
  (pass)
     │
     ▼
[Final Answer]
```

## MCP Tool Design

<!-- Describe each tool's input/output schema -->

## Vector Store Design

<!-- Describe ChromaDB collection structure, embedding strategy -->

## Configuration

<!-- Describe config.py parameters and how to tune them -->
