"""Text extraction from Zotero attachments: PDF, EPUB, HTML snapshots, and notes."""

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)


def preprocess_text(text):
    """Clean up extracted text: rejoin hyphenated line breaks, normalize whitespace."""
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\S\n\f]+', ' ', text)
    return text.strip()


def extract_pdf_text(pdf_bytes, item_key):
    """Extract text from PDF bytes using pdftotext -layout.

    Returns (text, page_count) tuple.
    """
    cache_path = CACHE_DIR / f"{item_key}.txt"
    if cache_path.exists():
        text = cache_path.read_text(encoding='utf-8')
        page_count = text.count('\f') + 1
        return preprocess_text(text), page_count

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ['pdftotext', '-layout', tmp_path, '-'],
            capture_output=True, timeout=120
        )
        text = result.stdout.decode('utf-8', errors='replace')
        page_count = text.count('\f') + 1

        cache_path.write_text(text, encoding='utf-8')

        return preprocess_text(text), page_count
    except FileNotFoundError:
        logger.error(
            "pdftotext not found. Install it:\n"
            "  macOS:  brew install poppler\n"
            "  Ubuntu: sudo apt install poppler-utils\n"
            "  Windows: download from https://github.com/oschwartz10612/poppler-windows/releases"
        )
        return "", 0
    except subprocess.TimeoutExpired:
        logger.warning(f"pdftotext timed out for {item_key}")
        return "", 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_epub_text(epub_bytes, item_key):
    """Extract text from EPUB bytes using ebooklib.

    Returns list of (chapter_title, chapter_text) tuples.
    """
    import ebooklib
    from ebooklib import epub

    with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
        f.write(epub_bytes)
        tmp_path = f.name

    try:
        book = epub.read_epub(tmp_path)
        chapters = []

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'lxml')
            heading = soup.find(['h1', 'h2', 'h3'])
            title = heading.get_text(strip=True) if heading else item.get_name()
            text = soup.get_text(separator='\n')
            text = preprocess_text(text)
            if text.strip():
                chapters.append((title, text))

        return chapters
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_html_text(html_content):
    """Extract text from HTML snapshot content."""
    if isinstance(html_content, bytes):
        html_content = html_content.decode('utf-8', errors='replace')

    soup = BeautifulSoup(html_content, 'lxml')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
        tag.decompose()

    text = soup.get_text(separator='\n')
    return preprocess_text(text)


def extract_note_text(note_html):
    """Extract plain text from a Zotero note (HTML)."""
    soup = BeautifulSoup(note_html, 'lxml')
    text = soup.get_text(separator='\n')
    return preprocess_text(text)


def select_best_attachment(attachments):
    """Select the best attachment from a list, following priority:
    EPUB > PDF (longest) > Snapshot > skip JPEG.

    Returns (attachment, attachment_type) or (None, None).
    """
    epubs = []
    pdfs = []
    snapshots = []

    for att in attachments:
        data = att['data']
        content_type = data.get('contentType', '')
        link_mode = data.get('linkMode', '')
        filename = data.get('filename', '')

        if content_type == 'application/epub+zip' or filename.endswith('.epub'):
            epubs.append(att)
        elif content_type == 'application/pdf' or filename.endswith('.pdf'):
            pdfs.append(att)
        elif link_mode == 'imported_url' or content_type.startswith('text/html'):
            snapshots.append(att)

    if epubs:
        return epubs[0], 'epub'

    if pdfs:
        if len(pdfs) == 1:
            return pdfs[0], 'pdf'
        best = pdfs[0]
        best_pages = 0
        for pdf in pdfs:
            pages = pdf['data'].get('numPages', 0) or 0
            filename = pdf['data'].get('filename', '')
            if re.search(r'_from_\d+_to_\d+|_part_\d+', filename):
                continue
            if pages > best_pages:
                best = pdf
                best_pages = pages
        return best, 'pdf'

    if snapshots:
        return snapshots[0], 'snapshot'

    return None, None


def extract_item_metadata(item):
    """Extract structured metadata from a Zotero item for chunk tagging."""
    data = item['data']
    rag = item.get('_rag', {})

    creators = data.get('creators', [])
    authors = []
    for c in creators:
        name_parts = []
        if c.get('firstName'):
            name_parts.append(c['firstName'])
        if c.get('lastName'):
            name_parts.append(c['lastName'])
        if name_parts:
            authors.append(' '.join(name_parts))
        elif c.get('name'):
            authors.append(c['name'])

    title = data.get('title', '').strip()

    coll_names_set = set()
    for c in rag.get('zotero_collections', []):
        coll_names_set.add(c['name'])
        for ancestor in c.get('path', []):
            coll_names_set.add(ancestor)
    coll_names = sorted(coll_names_set)

    return {
        'zotero_key': item['key'],
        'title': title,
        'authors': authors,
        'item_type': data.get('itemType', ''),
        'date': data.get('date', ''),
        'archive': data.get('archive', ''),
        'archive_location': data.get('archiveLocation', ''),
        'tags': [t['tag'] for t in data.get('tags', [])],
        'collections': coll_names,
        'archive_collection': rag.get('archive_collection', ''),
        'archive_visit_date': rag.get('archive_visit_date', ''),
        'abstract': data.get('abstractNote', ''),
    }
