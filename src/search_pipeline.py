"""Shared search pipeline for Zotero RAG.

Used by both the MCP server (server.py) and the web app (webapp.py).
"""

import json
import logging
import re

from flashrank import Ranker, RerankRequest

from src.config import ARCHIVE_ALIASES_FILE
from src.embeddings import init_embeddings, embed_query
from src.vectordb import init_pinecone, search, get_index_stats

logger = logging.getLogger(__name__)

_initialized = False
_archive_aliases = {}
_ranker = None

_SHORTHAND_KEYS = {'type', 'by', 'tag', 'in', 'from', 'to', 'collection', 'top'}


def init_pipeline():
    """Initialize embeddings, Pinecone, archive aliases, and reranker."""
    global _initialized, _archive_aliases, _ranker
    if _initialized:
        return
    init_embeddings()
    init_pinecone()
    if ARCHIVE_ALIASES_FILE.exists():
        data = json.loads(ARCHIVE_ALIASES_FILE.read_text())
        _archive_aliases = {k.lower(): v for k, v in data.get('aliases', {}).items()}
        logger.info(f"Loaded {len(_archive_aliases)} archive aliases")
    _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
    logger.info("Flashrank reranker initialized")
    _initialized = True


def get_archive_aliases():
    """Return the loaded archive aliases dict (lowercase key -> canonical name)."""
    return dict(_archive_aliases)


def parse_shorthand(query):
    """Extract shorthand prefixes from query.

    Supports exact matching with = prefix: in:="Some Archive"
    Without =, matching is fuzzy (substring).

    Returns (cleaned_query, dict_of_filters).
    """
    filters = {}
    pattern = r'\b(' + '|'.join(_SHORTHAND_KEYS) + r'):(=?)(\"[^\"]+\"|\'[^\']+\'|\S+)'

    def _replace(m):
        key = m.group(1)
        exact = m.group(2)
        val = m.group(3).strip('"').strip("'")
        filters[key] = ('=' + val) if exact else val
        return ''

    cleaned = re.sub(pattern, _replace, query)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned, filters


def _match_filter(filter_val, target):
    """Match a filter value against a target string."""
    if not target:
        return False
    if filter_val.startswith('='):
        return filter_val[1:].lower() == target.lower()
    return filter_val.lower() in target.lower()


def _match_archive(filter_val, archive_collection):
    """Match an archive filter against an item's archive_collection."""
    if not archive_collection:
        return False
    exact = filter_val.startswith('=')
    raw_val = filter_val[1:] if exact else filter_val
    canonical = _archive_aliases.get(raw_val.lower())
    if canonical:
        return archive_collection.lower() == canonical.lower()
    if exact:
        return raw_val.lower() == archive_collection.lower()
    return raw_val.lower() in archive_collection.lower()


def run_search(query_str, top_k=10, item_type=None, author=None, tag=None,
               collection=None, archive=None, date_from=None, date_to=None):
    """Run the full search pipeline: parse shorthand -> embed -> Pinecone -> filter -> rerank.

    Returns list of result dicts with keys: id, score, metadata, rerank_score.
    """
    init_pipeline()

    query, parsed = parse_shorthand(query_str)
    item_type = item_type or parsed.get('type')
    author = author or parsed.get('by')
    tag = tag or parsed.get('tag')
    collection = collection or parsed.get('collection')
    archive = archive or parsed.get('in')
    date_from = date_from or parsed.get('from')
    date_to = date_to or parsed.get('to')
    if parsed.get('top'):
        try:
            top_k = int(parsed['top'])
        except ValueError:
            pass

    top_k = min(max(top_k, 1), 20)

    filters = {}
    if item_type:
        filters['item_type'] = item_type.lstrip('=')
    pinecone_filter = filters if filters else None

    query_embedding = embed_query(query)

    needs_client_filter = any([author, tag, collection, archive, date_from, date_to])
    search_k = top_k * 5 if needs_client_filter else max(top_k * 3, 30)
    results = search(query_embedding, top_k=search_k, filters=pinecone_filter)

    if needs_client_filter:
        filtered = []
        for r in results:
            meta = r['metadata']
            if author and not _match_filter(author, ' '.join(meta.get('authors', []))):
                continue
            if tag and not any(_match_filter(tag, t) for t in meta.get('tags', [])):
                continue
            if collection and not any(_match_filter(collection, c) for c in meta.get('collections', [])):
                continue
            if archive and not _match_archive(archive, meta.get('archive_collection', '')):
                continue
            if date_from and meta.get('date', '') and meta['date'] < date_from:
                continue
            if date_to and meta.get('date', '') and meta['date'] > date_to:
                continue
            filtered.append(r)
        results = filtered

    if results and _ranker:
        passages = [
            {"id": i, "text": r['metadata'].get('text', '')[:1500]}
            for i, r in enumerate(results)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked = _ranker.rerank(rerank_req)
        reranked = []
        for rr in ranked[:top_k]:
            original = results[rr["id"]]
            original['rerank_score'] = rr["score"]
            reranked.append(original)
        results = reranked

    return results
