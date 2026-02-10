"""Embedding client supporting OpenAI and Ollama backends."""

import logging
import time

from src.config import (
    EMBEDDING_PROVIDER, OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
)

logger = logging.getLogger(__name__)

_client = None
_provider = None


def init_embeddings():
    """Initialize the embedding client based on EMBEDDING_PROVIDER config."""
    global _client, _provider
    _provider = EMBEDDING_PROVIDER.lower()

    if _provider == "ollama":
        # Verify Ollama is reachable
        import urllib.request
        try:
            urllib.request.urlopen(OLLAMA_BASE_URL, timeout=5)
        except Exception:
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
                "Make sure Ollama is running (https://ollama.ai)."
            )
        logger.info(f"Using Ollama embeddings: {OLLAMA_EMBED_MODEL}")
    else:
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Export it or add to .env file.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"Using OpenAI embeddings: {EMBEDDING_MODEL}")


def _embed_ollama(texts):
    """Embed texts using Ollama's local API."""
    import json
    import urllib.request

    embeddings = []
    for text in texts:
        text = text.strip() or "[empty]"
        if len(text) > 30000:
            text = text[:30000]
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            data=json.dumps({"model": OLLAMA_EMBED_MODEL, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        embeddings.append(result["embedding"])
    return embeddings


def _embed_openai(texts):
    """Embed texts using OpenAI API."""
    sanitized = []
    for t in texts:
        t = t.strip() if t else ""
        if not t:
            t = "[empty]"
        if len(t) > 30000:
            t = t[:30000]
        sanitized.append(t)

    BATCH_SIZE = 2048
    all_embeddings = []

    for i in range(0, len(sanitized), BATCH_SIZE):
        batch = sanitized[i:i + BATCH_SIZE]
        try:
            response = _client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
        except Exception as e:
            if 'rate' in str(e).lower() or '429' in str(e):
                logger.warning("Rate limited, waiting 60s...")
                time.sleep(60)
                response = _client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                all_embeddings.extend([d.embedding for d in response.data])
            else:
                raise

    return all_embeddings


def embed_texts(texts):
    """Embed a batch of texts for indexing.

    Returns list of embedding vectors.
    """
    if not texts:
        return []
    if _provider == "ollama":
        return _embed_ollama(texts)
    return _embed_openai(texts)


def embed_query(query_text):
    """Embed a single search query."""
    if _provider == "ollama":
        return _embed_ollama([query_text])[0]
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_text,
    )
    return response.data[0].embedding


def get_embedding_dimension():
    """Return the dimension of embeddings from the current provider."""
    if _provider == "ollama":
        # Embed a test string to detect dimension
        test = embed_query("test")
        return len(test)
    return EMBEDDING_DIMENSION
