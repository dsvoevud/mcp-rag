"""
config.py — Central configuration for the MCP RAG server.

All tunable parameters live here: model names, chunk sizes,
vector store paths, retrieval top-k, and retry limits.
"""

# TODO: implement

# LLM
LLM_MODEL: str = "phi3:mini"
EMBEDDING_MODEL: str = "nomic-embed-text"
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Chunking
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64

# Vector store
CHROMA_DB_PATH: str = "./chroma_db"
COLLECTION_NAME: str = "rag_collection"

# Retrieval
TOP_K: int = 5

# Corrective RAG graph
MAX_RETRIES: int = 2
