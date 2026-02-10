"""Zotero API client and collection hierarchy mapper."""

import re
from pyzotero import zotero
from src.config import (
    ZOTERO_LIBRARY_ID, ZOTERO_API_KEY, ZOTERO_LIBRARY_TYPE, COLLECTION_KEY,
)


def get_zotero_client():
    if not ZOTERO_LIBRARY_ID or not ZOTERO_API_KEY:
        raise ValueError(
            "ZOTERO_LIBRARY_ID and ZOTERO_API_KEY must be set. "
            "See .env.example for details."
        )
    return zotero.Zotero(ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE, ZOTERO_API_KEY)


def build_collection_tree(zot, root_key=None):
    """Build a complete tree of a collection and all subcollections.

    If root_key is None, uses COLLECTION_KEY from config.
    If that is also empty, returns an empty dict (indexes entire library).

    Returns:
        dict mapping collection_key -> {
            'name': str,
            'parent_key': str or None,
            'path': list[str],
            'archive_name': str or None,
            'visit_date': str or None,
        }
    """
    root_key = root_key or COLLECTION_KEY
    if not root_key:
        return {}

    coll_lookup = {}

    def _discover(parent_key):
        subs = zot.collections_sub(parent_key)
        for c in subs:
            if c['key'] not in coll_lookup:
                coll_lookup[c['key']] = {
                    'name': c['data']['name'],
                    'parent_key': c['data'].get('parentCollection', None),
                }
                _discover(c['key'])

    _discover(root_key)

    # Add the root collection
    root = zot.collection(root_key)
    coll_lookup[root_key] = {
        'name': root['data']['name'],
        'parent_key': None,
    }

    # Build paths by walking up the tree
    def get_path(key):
        path = []
        current = key
        while current and current in coll_lookup:
            path.append(coll_lookup[current]['name'])
            current = coll_lookup[current]['parent_key']
        path.reverse()
        return path

    # Parse visit dates from collection names (e.g., "DTRP: 2025/12/17")
    visit_date_pattern = re.compile(r'(\d{4}/\d{2}/\d{2})')

    # Auto-detect archive collections: any collection that is a direct child of
    # the root and has subcollections is treated as an "archive" collection.
    # This generalizes the approach — no hardcoded archive list needed.
    archive_keys = {}
    for key, data in coll_lookup.items():
        if data['parent_key'] == root_key:
            # Check if this collection has subcollections
            has_children = any(
                v['parent_key'] == key for v in coll_lookup.values()
            )
            if has_children:
                archive_keys[key] = data['name']

    result = {}
    for key, data in coll_lookup.items():
        path = get_path(key)

        # Determine archive name: walk up the path to find the archive collection
        archive_name = None
        current = key
        while current and current in coll_lookup:
            if current in archive_keys:
                archive_name = archive_keys[current]
                break
            current = coll_lookup[current]['parent_key']

        # Parse visit date from name
        visit_date = None
        match = visit_date_pattern.search(data['name'])
        if match:
            visit_date = match.group(1)

        result[key] = {
            'name': data['name'],
            'parent_key': data['parent_key'],
            'path': path,
            'archive_name': archive_name,
            'visit_date': visit_date,
        }

    return result


def get_all_items(zot, collection_tree):
    """Fetch all items from the target collection(s).

    If collection_tree is populated, fetches from each subcollection and
    deduplicates. Otherwise, fetches the entire library.

    Returns list of items with enriched metadata including collection info.
    """
    seen_keys = set()
    all_top_level = []

    if collection_tree:
        collection_keys = list(collection_tree.keys())
        for coll_key in collection_keys:
            items = zot.everything(zot.collection_items(coll_key))
            for item in items:
                if item['data']['itemType'] in ('attachment', 'note'):
                    continue
                if item['key'] not in seen_keys:
                    seen_keys.add(item['key'])
                    all_top_level.append(item)
    else:
        # No collection specified — index entire library
        items = zot.everything(zot.top())
        for item in items:
            if item['data']['itemType'] in ('attachment', 'note'):
                continue
            if item['key'] not in seen_keys:
                seen_keys.add(item['key'])
                all_top_level.append(item)

    # Enrich each item with collection metadata
    for item in all_top_level:
        item_colls = item['data'].get('collections', [])
        coll_info = []
        archive_name = None
        visit_date = None

        for ck in item_colls:
            if ck in collection_tree:
                ct = collection_tree[ck]
                coll_info.append({
                    'key': ck,
                    'name': ct['name'],
                    'path': ct['path'],
                })
                if ct['archive_name'] and not archive_name:
                    archive_name = ct['archive_name']
                if ct['visit_date'] and not visit_date:
                    visit_date = ct['visit_date']

        item['_rag'] = {
            'zotero_collections': coll_info,
            'archive_collection': archive_name,
            'archive_visit_date': visit_date,
        }

    return all_top_level


def get_child_attachments(zot, parent_key):
    """Get all child attachments for a parent item."""
    children = zot.children(parent_key)
    return [c for c in children if c['data']['itemType'] == 'attachment']


def get_child_notes(zot, parent_key):
    """Get all child notes for a parent item."""
    children = zot.children(parent_key)
    return [c for c in children if c['data']['itemType'] == 'note']
