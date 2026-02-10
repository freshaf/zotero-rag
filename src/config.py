"""Configuration for the Zotero RAG pipeline.

All credentials and settings are loaded from environment variables.
Copy .env.example to .env and fill in your values.
"""

import os
import pathlib

# Zotero API (required)
ZOTERO_LIBRARY_ID = os.environ.get("ZOTERO_LIBRARY_ID", "")
ZOTERO_API_KEY = os.environ.get("ZOTERO_API_KEY", "")
ZOTERO_LIBRARY_TYPE = os.environ.get("ZOTERO_LIBRARY_TYPE", "user")

# Collection to index (optional — indexes entire library if not set)
COLLECTION_KEY = os.environ.get("ZOTERO_COLLECTION_KEY", "")

# Paths
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Chunking defaults
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 150
SHORT_DOC_THRESHOLD_TOKENS = 1000

# Embedding provider: "openai" or "ollama"
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")

# OpenAI embeddings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "1536"))

# Ollama embeddings (free, runs locally)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Pinecone vector database
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "zotero-rag")

# LLM provider for Q&A: "anthropic", "openai", or "ollama"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")

# Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# OpenAI chat
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Ollama chat (free, runs locally)
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1")

# Sync state file
SYNC_STATE_FILE = PROJECT_ROOT / "sync_state.json"

# Archive aliases (generated at index time from collection tree)
ARCHIVE_ALIASES_FILE = PROJECT_ROOT / "archive_aliases.json"
