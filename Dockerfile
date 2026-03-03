# ---------------------------------------------------------------------------
# Stage 1 — dependency builder
# Install packages in a separate layer so rebuilds on code changes are fast.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Build tools needed by hnswlib (ChromaDB dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy pre-installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY sample_docs/ ./sample_docs/

# ---------------------------------------------------------------------------
# Runtime configuration
#
# LM Studio runs on the host machine. Inside Docker Desktop (Windows/macOS)
# the host is reachable via host.docker.internal. On Linux you must also
# add `extra_hosts: ["host.docker.internal:host-gateway"]` in compose
# (already done in docker-compose.yml).
#
# TRANSPORT   stdio            → MCP stdio (default; used by Claude Desktop)
#             streamable-http  → HTTP on MCP_HOST:MCP_PORT (used in Docker)
# LLM_BASE_URL                 → LM Studio on host (via host.docker.internal)
# LLM_MODEL                   → must match the model loaded in LM Studio
# CHROMA_DB_PATH              → mount a volume here for persistence
# ---------------------------------------------------------------------------
ENV TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    LLM_BASE_URL=http://host.docker.internal:1234/v1 \
    LLM_MODEL=deepseek/deepseek-r1-0528-qwen3-8b \
    LLM_API_KEY=lm-studio \
    LLM_TEMPERATURE=0.0 \
    LLM_STRIP_THINKING_TAGS=true \
    CHROMA_DB_PATH=/app/chroma_db \
    COLLECTION_NAME=rag_collection

# Persist ChromaDB data and allow external sample_docs injection
VOLUME ["/app/chroma_db", "/app/sample_docs"]

# Expose HTTP port (only used when TRANSPORT=streamable-http)
EXPOSE 8000

# Healthcheck — only meaningful in HTTP mode
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["python", "-m", "src.server"]
