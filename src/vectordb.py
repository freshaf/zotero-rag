"""Pinecone vector database operations."""

import logging
from pinecone import Pinecone, ServerlessSpec

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

_pc = None
_index = None


def init_pinecone(dimension=None):
    """Initialize Pinecone client and ensure index exists.

    Args:
        dimension: embedding dimension (auto-detected if not provided)
    """
    global _pc, _index
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set. Export it or add to .env file.")

    dim = dimension or EMBEDDING_DIMENSION

    _pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (dimension={dim})...")
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created.")

    _index = _pc.Index(PINECONE_INDEX_NAME)
    return _index


def upsert_chunks(chunks_with_embeddings):
    """Upsert chunks with their embeddings into Pinecone.

    Args:
        chunks_with_embeddings: list of (chunk_id, embedding_vector, metadata_dict)
    """
    BATCH_SIZE = 100
    for i in range(0, len(chunks_with_embeddings), BATCH_SIZE):
        batch = chunks_with_embeddings[i:i + BATCH_SIZE]
        vectors = []
        for chunk_id, embedding, metadata in batch:
            clean_meta = _clean_metadata(metadata)
            vectors.append({
                'id': chunk_id,
                'values': embedding,
                'metadata': clean_meta,
            })
        _index.upsert(vectors=vectors)


def search(query_embedding, top_k=10, filters=None):
    """Search for similar chunks."""
    kwargs = {
        'vector': query_embedding,
        'top_k': top_k,
        'include_metadata': True,
    }
    if filters:
        kwargs['filter'] = filters

    results = _index.query(**kwargs)
    return [
        {
            'id': match.id,
            'score': match.score,
            'metadata': dict(match.metadata),
        }
        for match in results.matches
    ]


def delete_by_zotero_key(zotero_key):
    """Delete all chunks for a given Zotero item key."""
    _index.delete(filter={'zotero_key': zotero_key})


def get_index_stats():
    """Get stats about the current index."""
    return _index.describe_index_stats()


def _clean_metadata(metadata):
    """Ensure all metadata values are Pinecone-compatible types."""
    clean = {}
    for k, v in metadata.items():
        if v is None or v == '':
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = [str(x) for x in v if x]
        else:
            clean[k] = str(v)
    return clean
