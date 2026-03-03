# Report — MCP RAG Server: Development History

## What Was Built

A fully local **Corrective RAG MCP server** that turns a folder of documents into
a searchable knowledge base accessible from any MCP-compatible IDE or client.

The server exposes five tools (`index_folder`, `ask_question`, `find_relevant_docs`,
`summarize_document`, `index_status`), orchestrates retrieval and generation through
a LangGraph Corrective RAG graph, and persists the vector index in ChromaDB.
No paid API keys are required — the LLM runs locally via LM Studio.

---

## Development Story

### Phase 1 — Configuration

The first design decision was to make **every tunable parameter environment-variable
backed** from the start. This turned out to be the right call: when the Docker
target changed (LM Studio instead of Ollama), only the `ENV` block in the
Dockerfile and the `environment:` section in `docker-compose.yml` needed updating —
zero code changes.

### Phase 2 — Indexer

The indexer presented two early problems:

**Encoding detection.** The corpus included Cyrillic files saved as UTF-8-SIG
(UTF-8 with a byte-order mark). Python's default `open()` would silently
misread the BOM as content. The fix was `chardet` auto-detection with a UTF-8
fallback — robust for any mixed-encoding corpus.

**Deprecated LangChain imports.** The original scaffolding used
`from langchain.schema import Document` and `from langchain.text_splitter import ...`.
Both are deprecated in LangChain ≥0.2 and were moved to
`langchain_core.documents` and `langchain_text_splitters`. Updating imports early
avoided a raft of deprecation warnings in tests.

**Upsert semantics.** To prevent duplicate chunks on re-indexing, each chunk
receives a deterministic MD5 ID based on its source path and position. ChromaDB's
`upsert` then replaces rather than appends — confirmed by the
`test_reindexing_does_not_duplicate_chunks` test.

### Phase 3 — Prompts

The most important prompt-engineering lesson was to **explicitly request structured
output** for binary classification nodes (grading and hallucination check).

Early prompt iterations asked the model to "answer yes or no". The model would
produce answers like `"Yes, this document is relevant because..."`, requiring
fragile substring parsing. Switching to `Output ONLY valid JSON: {"relevant": "yes"}`
and adding a robust three-stage parser (`strict JSON → regex fallback → bare-word
fallback → default`) made the pipeline resilient to any model's formatting habits.

**Language-agnostic instruction.** Adding `"Respond in the same language as the
question"` to every prompt was a late addition prompted by testing with Russian
queries. Without it, the LLM would answer Russian questions in English.

**Thinking tag stripping.** DeepSeek R1 models emit `<think>…</think>` blocks
before their final answer. Passing these raw to downstream nodes (e.g. treating
`<think>Let me check…</think>{"relevant":"yes"}` as a JSON string) broke the
grader. The `strip_thinking_tags()` utility and the `LLM_STRIP_THINKING_TAGS`
config flag were added to handle this cleanly.

### Phase 4 — LangGraph Graph

**Lazy singleton pattern.** The `_get_indexer()` and `_get_llm()` module-level
singletons are initialised on first call rather than at import time. This was
essential for testing — it allows tests to patch the singletons before the graph
ever touches them, without requiring dependency injection throughout every node
function.

**Retry routing.** A subtle issue arose in the `_route_after_grading` edge: if
`retrieve_retry_count` had hit its maximum, the graph should fall through to
`generate` with whatever (possibly empty) context it has, rather than looping
forever or aborting. The explicit `else → generate` branch handles this.

**`_parse_binary_json` helper.** Because different models format their JSON
output differently (sometimes with Markdown code fences, sometimes with extra
whitespace, sometimes as bare words), a dedicated three-stage parser was written
rather than relying on `json.loads()` alone. This made every grading and
hallucination-check node robust across model families.

### Phase 5 — FastMCP Server

**Transport abstraction.** The `main()` function reads a `TRANSPORT` environment
variable to select between `stdio` (for desktop MCP client integration) and
`streamable-http` (for Docker). This meant no code changes were needed when the
deployment target changed — only the environment variable value differs.

**Empty index guard.** `ask_question` checks `get_status()["total_chunks"] == 0`
before even invoking the graph. Without this guard the graph would run, retrieve
nothing, retry twice, and return an empty answer with no helpful message. The
early-exit returns a clear instruction to run `index_folder` first.

### Phase 6 — Tests

**Config patching strategy.** The `Indexer` class reads `cfg.CHROMA_DB_PATH` both
in `__init__` (to create the client) and in `get_status()` (to report the path).
Simply patching at construction time was not enough — the patch had to remain
active for the test's entire lifetime. The solution was a `yield`-based pytest
fixture that holds the `patch.object` context manager open for the full test scope.

**Mocking `_invoke_llm` vs mocking the LLM object.** Graph tests patch
`graph_module._invoke_llm` — the internal string-returning adapter — rather than
the `ChatOpenAI` object. This is cleaner because: (1) the return type is always
`str`, avoiding LangChain `AIMessage` wrapper concerns; (2) it's model-agnostic;
(3) the patch target is stable regardless of how the LLM backend changes.

**Summarize tool patch.** `summarize_document` builds its chain inline as
`SUMMARIZATION_PROMPT | _get_llm()`. Patching either side alone results in a
real `ChatPromptTemplate` trying to pipe into a `MagicMock`, raising a LangChain
type error. The fix is to patch `SUMMARIZATION_PROMPT` itself with a mock whose
`__or__` returns a pre-configured fake chain.

### Phase 7 — Docker

The initial Docker plan used Ollama as the LLM backend (since LM Studio is a
desktop application and cannot run inside a container). After reviewing the task
requirements, LM Studio was reinstated as the LLM provider. The solution:

- The container reaches LM Studio on the host via `host.docker.internal:1234`
- Docker Desktop (Windows/macOS) resolves `host.docker.internal` automatically
- `extra_hosts: ["host.docker.internal:host-gateway"]` in `docker-compose.yml`
  provides the same mapping on Linux hosts
- The Dockerfile `ENV` block sets `LLM_BASE_URL=http://host.docker.internal:1234/v1`
  as the baked-in default

This reduced the compose file from three services (ollama + ollama-init + mcp-rag)
to a single `mcp-rag` service, eliminating model pull orchestration entirely.

### Phase 8 — Sample Documents

The sample document set was designed to demonstrate the **"project-specific
overrides"** RAG use-case:

- `Tolkien_The_Hobbit.txt` — the base corpus (common knowledge)
- `Later_edits.txt` — 10 absurd fictional "amendments" (e.g. Smaug is black with
  pink polka dots; Thorin's axe is a Jedi lightaxe; Bilbo's full name is
  Thunderpants McBaggins III) that visibly override the base text in retrieval

When a user asks *"What colour is Smaug?"*, the `Later_edits.txt` chunk ranks
higher by cosine similarity than the Hobbit text because it directly contains
the query terms alongside the answer. The model then reports the absurd amended
fact rather than Tolkien's original — which is exactly the intended demonstration
of how project-specific documentation supersedes public knowledge in a RAG system.

War and Peace was initially included as a multilingual stress test but removed
before the final demo — it is too token-expensive per query and adds no value
beyond what the Hobbit corpus already provides.

---

## Key Problems and Solutions

| Problem | Root cause | Solution |
|---|---|---|
| Python 3.14t incompatible with native-extension wheels | Free-threaded CPython requires Rust toolchain for hnswlib/chromadb | Installed Python 3.11 via winget; created fresh venv |
| UTF-8-SIG BOM in Cyrillic files | Legacy encoding | `chardet` auto-detection + UTF-8 re-encode |
| Deprecated LangChain imports | LangChain ≥0.2 restructuring | Updated to `langchain_core.documents`, `langchain_text_splitters` |
| LLM returns free-text instead of JSON | Prompt not explicit enough | Added strict format instruction + three-stage JSON fallback parser |
| `<think>` tags breaking downstream JSON parsing | DeepSeek R1 reasoning blocks | `strip_thinking_tags()` + `LLM_STRIP_THINKING_TAGS` config flag |
| `cfg` read at call-time breaks test isolation | `get_status()` reads `cfg` live | `yield`-based fixture holds `patch.object` open for full test scope |
| LangChain type error when mocking `summarize_document` | `ChatPromptTemplate.__or__` rejects `MagicMock` | Patch `SUMMARIZATION_PROMPT` itself with a mock whose `__or__` returns a full fake chain |
| `host.docker.internal` not resolving on Linux | Docker Desktop auto-maps it, bare Docker does not | `extra_hosts: ["host.docker.internal:host-gateway"]` in compose |

---

## Prompt Engineering — Successes and Failures

### What worked well

- **Binary JSON output format** (`{"relevant": "yes"}`) for classification nodes —
  easy to parse, unambiguous, works across model families
- **"Respond in the same language as the question"** — essential for multilingual
  corpora; without it models default to English
- **Source citation instruction** (`(source: <filename>)`) in `GENERATION_PROMPT` —
  gives the user provenance at no extra LLM cost
- **Three-stage JSON parser** (`strict → regex → bare-word → fallback`) — handles
  every model's output formatting quirks without prompt over-engineering

### What was tried and discarded

- **Asking the grader to explain its reasoning** — produced verbose output that
  required more complex parsing and added latency with no downstream benefit;
  removed in favour of JSON-only output
- **Single mega-prompt** that combined grading + generation — LLMs would sometimes
  collapse both tasks into one step, skipping the grading; separating concerns
  into discrete nodes made the pipeline more reliable and easier to test
- **"Chain of thought" in hallucination check** — asking the model to justify its
  grounded/not-grounded verdict produced useful explanations but made the JSON
  extraction brittle; reverted to binary JSON output

---

## Limitations and Future Improvements

- **LM Studio dependency** — the server requires LM Studio to be running on the
  host. For fully self-contained Docker deployment, adding an Ollama service back
  to `docker-compose.yml` (behind a feature flag) would remove this constraint.
- **Single ChromaDB collection** — all indexed documents share one collection.
  Namespacing by project or user would allow multi-tenant usage.
- **No PDF support** — the indexer skips `.pdf` files. Adding `PyPDFLoader` is
  straightforward and was left as a bonus task.
- **Embedding model is fixed** — ChromaDB's default `all-MiniLM-L6-v2` is fast
  but not the strongest retriever. The `EMBEDDING_PROVIDER=ollama` config path
  exists but is not wired up; connecting it to `nomic-embed-text` would improve
  retrieval quality on longer documents.
- **No streaming** — `ask_question` returns the full answer in one shot. Adding
  streaming via FastMCP's `ctx.stream()` would improve perceived responsiveness
  for long answers.
