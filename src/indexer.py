"""Main indexing pipeline: Zotero items -> text extraction -> chunking -> embedding -> Pinecone."""

import json
import logging
import re
import sys

from src.config import SYNC_STATE_FILE, ARCHIVE_ALIASES_FILE, COLLECTION_KEY
from src.zotero_client import (
    get_zotero_client, build_collection_tree, get_all_items,
    get_child_attachments,
)
from src.extractors import (
    select_best_attachment, extract_pdf_text, extract_epub_text,
    extract_html_text, extract_item_metadata,
)
from src.chunker import chunk_document, chunk_epub
from src.embeddings import init_embeddings, embed_texts, get_embedding_dimension
from src.vectordb import init_pinecone, upsert_chunks, delete_by_zotero_key

logger = logging.getLogger(__name__)


def _build_context_header(metadata):
    """Build a metadata header to prepend to chunk text before embedding."""
    parts = []
    if metadata.get('title'):
        parts.append(f"Title: {metadata['title']}")
    if metadata.get('authors'):
        parts.append(f"Authors: {', '.join(metadata['authors'])}")
    if metadata.get('date'):
        parts.append(f"Date: {metadata['date']}")
    if metadata.get('item_type'):
        parts.append(f"Type: {metadata['item_type']}")
    if metadata.get('archive'):
        parts.append(f"Archive: {metadata['archive']}")
    if metadata.get('archive_location'):
        parts.append(f"Location: {metadata['archive_location']}")
    if not parts:
        return ""
    return '\n'.join(parts) + '\n---\n'


def build_archive_aliases(collection_tree):
    """Build an alias map from the collection tree.

    Extracts acronym prefixes from subcollection names (e.g., "DTRP: 2025/12/17" -> DTRP).
    """
    aliases = {}

    acronym_pattern = re.compile(r'^([A-Z]{2,})\s*:')
    for ct in collection_tree.values():
        match = acronym_pattern.match(ct['name'])
        if match and ct['archive_name']:
            acronym = match.group(1)
            if acronym not in aliases:
                aliases[acronym] = ct['archive_name']

    result = {'aliases': aliases}
    ARCHIVE_ALIASES_FILE.write_text(json.dumps(result, indent=2))
    logger.info(f"Archive aliases saved: {len(aliases)} acronyms")
    return result


def load_sync_state():
    if SYNC_STATE_FILE.exists():
        return json.loads(SYNC_STATE_FILE.read_text())
    return {'library_version': 0, 'indexed_keys': []}


def save_sync_state(state):
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


def process_item(zot, item, collection_tree):
    """Process a single Zotero item: extract text, chunk, return chunks."""
    metadata = extract_item_metadata(item)
    item_key = item['key']
    item_type = item['data']['itemType']
    title = metadata['title'] or f"[Untitled {item_type}, {metadata['date']}]"
    metadata['title'] = title

    chunks = []

    attachments = get_child_attachments(zot, item_key)
    best_att, att_type = select_best_attachment(attachments)

    if best_att and att_type == 'pdf':
        try:
            pdf_bytes = zot.file(best_att['key'])
            text, page_count = extract_pdf_text(pdf_bytes, best_att['key'])
            if text.strip():
                metadata['attachment_key'] = best_att['key']
                metadata['attachment_type'] = 'pdf'
                metadata['page_count'] = page_count
                chunks = chunk_document(text, item_type, metadata)
        except Exception as e:
            logger.warning(f"Failed to extract PDF for {item_key}: {e}")

    elif best_att and att_type == 'epub':
        try:
            epub_bytes = zot.file(best_att['key'])
            chapters = extract_epub_text(epub_bytes, best_att['key'])
            if chapters:
                metadata['attachment_key'] = best_att['key']
                metadata['attachment_type'] = 'epub'
                chunks = chunk_epub(chapters, metadata)
        except Exception as e:
            logger.warning(f"Failed to extract EPUB for {item_key}: {e}")

    elif best_att and att_type == 'snapshot':
        try:
            html_bytes = zot.file(best_att['key'])
            text = extract_html_text(html_bytes)
            if text.strip():
                chunks = chunk_document(text, item_type, metadata)
        except Exception as e:
            logger.warning(f"Failed to extract snapshot for {item_key}: {e}")

    else:
        abstract = metadata.get('abstract', '')
        if abstract:
            chunks = chunk_document(abstract, item_type, metadata)
        else:
            meta_text = f"{title}. {', '.join(metadata['authors'])}. {metadata['date']}."
            if metadata['archive']:
                meta_text += f" {metadata['archive']}."
            if metadata['tags']:
                meta_text += f" Tags: {', '.join(metadata['tags'])}."
            chunks = [{'text': meta_text, 'chunk_index': 0, 'total_chunks': 1, 'metadata': metadata}]

    return chunks


def index_items(items, zot, collection_tree, batch_size=50):
    """Index a list of items: extract, chunk, embed, upsert to Pinecone."""
    all_chunks = []
    skipped = 0
    processed = 0

    for i, item in enumerate(items):
        title = item['data'].get('title', '')[:60]
        logger.info(f"[{i+1}/{len(items)}] Processing: {title}")

        chunks = process_item(zot, item, collection_tree)
        if chunks:
            all_chunks.extend(chunks)
            processed += 1
        else:
            skipped += 1
            logger.info(f"  -> skipped (no extractable content)")

    logger.info(f"Processed {processed} items, skipped {skipped}, total chunks: {len(all_chunks)}")

    if not all_chunks:
        logger.info("No chunks to index.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks...")
    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start:batch_start + batch_size]
        texts = [
            _build_context_header(c['metadata']) + c['text']
            for c in batch
        ]
        embeddings = embed_texts(texts)

        vectors = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            chunk_id = f"{chunk['metadata']['zotero_key']}_c{chunk['chunk_index']}"
            source_type = chunk['metadata'].get('source_type', 'document')
            flat_meta = {
                'text': chunk['text'][:2000],
                'zotero_key': chunk['metadata']['zotero_key'],
                'title': chunk['metadata']['title'],
                'authors': chunk['metadata']['authors'],
                'item_type': chunk['metadata']['item_type'],
                'date': chunk['metadata']['date'],
                'archive': chunk['metadata'].get('archive', ''),
                'archive_location': chunk['metadata'].get('archive_location', ''),
                'tags': chunk['metadata']['tags'],
                'collections': chunk['metadata']['collections'],
                'archive_collection': chunk['metadata'].get('archive_collection', ''),
                'chunk_index': chunk['chunk_index'],
                'total_chunks': chunk['total_chunks'],
                'source_type': source_type,
                'page_start': chunk['metadata'].get('page_start', 0),
                'page_end': chunk['metadata'].get('page_end', 0),
                'page_count': chunk['metadata'].get('page_count', 0),
                'pdf_page': chunk['metadata'].get('pdf_page', 0),
            }
            if 'attachment_key' in chunk['metadata']:
                flat_meta['attachment_key'] = chunk['metadata']['attachment_key']
            if 'attachment_type' in chunk['metadata']:
                flat_meta['attachment_type'] = chunk['metadata']['attachment_type']
            if 'chapter' in chunk['metadata']:
                flat_meta['chapter'] = chunk['metadata']['chapter']

            vectors.append((chunk_id, embedding, flat_meta))

        upsert_chunks(vectors)
        logger.info(f"  Upserted batch {batch_start//batch_size + 1} ({len(batch)} chunks)")

    logger.info("Indexing complete.")


def run_full_index():
    """Full re-index of the target collection (or entire library)."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    logger.info("Initializing services...")
    init_embeddings()
    dim = get_embedding_dimension()
    init_pinecone(dimension=dim)

    logger.info("Connecting to Zotero...")
    zot = get_zotero_client()

    tree = {}
    if COLLECTION_KEY:
        logger.info("Building collection tree...")
        tree = build_collection_tree(zot)
        logger.info(f"Found {len(tree)} collections")
        build_archive_aliases(tree)
    else:
        logger.info("No ZOTERO_COLLECTION_KEY set - indexing entire library")

    logger.info("Fetching all items...")
    items = get_all_items(zot, tree)
    logger.info(f"Found {len(items)} items")

    index_items(items, zot, tree)

    state = {
        'library_version': zot.last_modified_version(),
        'indexed_keys': [i['key'] for i in items],
    }
    save_sync_state(state)
    logger.info(f"Sync state saved (version {state['library_version']})")


def run_incremental_update():
    """Only index new/changed items since last sync."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    state = load_sync_state()
    last_version = state.get('library_version', 0)
    indexed_keys = set(state.get('indexed_keys', []))

    logger.info(f"Last sync version: {last_version}")

    init_embeddings()
    dim = get_embedding_dimension()
    init_pinecone(dimension=dim)

    zot = get_zotero_client()
    tree = {}
    if COLLECTION_KEY:
        tree = build_collection_tree(zot)
        build_archive_aliases(tree)

    items = get_all_items(zot, tree)
    current_keys = {i['key'] for i in items}

    removed = indexed_keys - current_keys
    for key in removed:
        logger.info(f"Removing deleted item: {key}")
        delete_by_zotero_key(key)

    new_items = []
    for item in items:
        item_version = item.get('version', 0)
        if item['key'] not in indexed_keys or item_version > last_version:
            new_items.append(item)

    if new_items:
        logger.info(f"Indexing {len(new_items)} new/changed items...")
        for item in new_items:
            if item['key'] in indexed_keys:
                delete_by_zotero_key(item['key'])
        index_items(new_items, zot, tree)
    else:
        logger.info("No new items to index.")

    state = {
        'library_version': zot.last_modified_version(),
        'indexed_keys': list(current_keys),
    }
    save_sync_state(state)
    logger.info("Sync state updated.")


if __name__ == '__main__':
    if '--update' in sys.argv:
        run_incremental_update()
    else:
        run_full_index()
