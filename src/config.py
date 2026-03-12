"""
config.py — Central configuration for the MCP RAG server.

All tunable parameters live here: LLM provider settings, chunk sizes,
vector store paths, retrieval top-k, and retry limits.

Every value can be overridden by an environment variable of the same name,
which makes it trivial to reconfigure the server inside Docker Compose
without rebuilding the image.

LLM backend: LM Studio (OpenAI-compatible API) running at localhost:1234.
               Inside Docker, set LLM_BASE_URL=http://ollama:11434/v1.
Embedding backend: ChromaDB default (all-MiniLM-L6-v2, runs in-process).
"""

import os

def _bool(name: str, default: bool) -> bool:
    """Read a boolean from an env var; accept 'true'/'1'/'yes' (case-insensitive)."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# LLM — LM Studio exposes an OpenAI-compatible REST API
# Set LLM_MODEL to the model name shown in LM Studio's "Model" tab.
# In Docker, override LLM_BASE_URL to point at the Ollama container.
# ---------------------------------------------------------------------------
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "lm-studio")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "lmstudio-community/qwen2.5-7b-instruct-1m")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

# DeepSeek R1 models emit chain-of-thought inside <think>...</think> tags
# before the final answer. Qwen2.5 does not emit thinking tags, so this
# defaults to False. Set to True only for DeepSeek R1 / QwQ models.
LLM_STRIP_THINKING_TAGS: bool = _bool("LLM_STRIP_THINKING_TAGS", False)

# ---------------------------------------------------------------------------
# Embeddings — ChromaDB in-process default model (no extra server needed)
# Swap to "nomic-embed-text" via Ollama for the bonus task.
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER: str = os.environ.get("EMBEDDING_PROVIDER", "default")
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL: str = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Chunking
# Tuned for long literary / mixed-language prose (e.g. Hobbit).
# Cyrillic tokenizes at ~1.5–2× tokens per word vs Latin, so keep
# chunk_size on the higher side to preserve paragraph-level context.
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "150"))

# ---------------------------------------------------------------------------
# Vector store — ChromaDB in-process (no separate server)
# ---------------------------------------------------------------------------
CHROMA_DB_PATH: str = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "rag_collection")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = int(os.environ.get("TOP_K", "10"))

# ---------------------------------------------------------------------------
# Auto-indexing
# If set, the server will automatically index this path on startup when the
# collection is empty. Useful for Docker deployments where the sample_docs
# folder is always mounted at a known location.
# ---------------------------------------------------------------------------
AUTO_INDEX_PATH: str = os.environ.get("AUTO_INDEX_PATH", "")

# ---------------------------------------------------------------------------
# Corrective RAG graph
# ---------------------------------------------------------------------------
MAX_RETRIEVE_RETRIES: int = int(os.environ.get("MAX_RETRIEVE_RETRIES", "2"))
MAX_GENERATE_RETRIES: int = int(os.environ.get("MAX_GENERATE_RETRIES", "1"))
