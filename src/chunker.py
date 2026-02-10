"""Adaptive chunking for different document types."""

import re
import tiktoken

from src.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, SHORT_DOC_THRESHOLD_TOKENS

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(_enc.encode(text))


def _split_by_tokens(text, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """Split text into chunks of roughly chunk_size tokens with overlap."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = _enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap
    return chunks


def _split_at_boundaries(text, separators, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """Split text at structural boundaries, then sub-split if chunks are too large."""
    segments = [text]
    for sep in separators:
        new_segments = []
        for seg in segments:
            parts = re.split(sep, seg)
            new_segments.extend(p for p in parts if p.strip())
        if len(new_segments) > len(segments):
            segments = new_segments
            break

    chunks = []
    current = ""
    for seg in segments:
        combined = (current + "\n\n" + seg).strip() if current else seg
        if count_tokens(combined) <= chunk_size:
            current = combined
        else:
            if current:
                chunks.append(current)
            if count_tokens(seg) > chunk_size:
                chunks.extend(_split_by_tokens(seg, chunk_size, overlap))
            else:
                current = seg
                continue
            current = ""
    if current:
        chunks.append(current)

    return chunks


# Congressional hearing structural markers
HEARING_SEPARATORS = [
    r'\n(?=STATEMENT OF [A-Z])',
    r'\n(?=The CHAIRMAN\.)',
    r'\n(?=Senator [A-Z]+\.)',
    r'\n(?=Secretary [A-Z]+\.)',
    r'\n(?=Mr\. [A-Z]+\.)',
    r'\n(?=CONCLUSION)',
    r'\n(?=[A-Z][A-Z ]{10,})\n',
]

# Meeting minutes markers
MINUTES_SEPARATORS = [
    r'\n(?=(?:AGENDA ITEM|Item|ITEM)\s*(?:#|\d))',
    r'\n(?=(?:OLD BUSINESS|NEW BUSINESS|ROLL CALL|ADJOURNMENT))',
    r'\n(?=[A-Z][A-Z ]{10,})\n',
]

# Book/report section markers
SECTION_SEPARATORS = [
    r'\n(?=(?:CHAPTER|Chapter)\s+\d)',
    r'\n(?=(?:PART|Part)\s+(?:\d|[IVX]))',
    r'\n(?=(?:SECTION|Section)\s+\d)',
    r'\n(?=[A-Z][A-Z ]{10,})\n',
]


def classify_document(item_type, text_length_tokens):
    """Classify a document for chunking strategy."""
    long_types = {'hearing', 'book'}
    medium_types = {'report', 'document'}

    if item_type in long_types:
        return 'long'
    if item_type in medium_types and text_length_tokens > SHORT_DOC_THRESHOLD_TOKENS:
        return 'medium'
    if text_length_tokens <= SHORT_DOC_THRESHOLD_TOKENS:
        return 'short'
    if text_length_tokens > CHUNK_SIZE_TOKENS * 2:
        return 'medium'
    return 'short'


def chunk_document(text, item_type, metadata):
    """Chunk a document adaptively based on its type and length.

    Args:
        text: extracted full text
        item_type: Zotero item type
        metadata: dict from extract_item_metadata

    Returns:
        list of chunk dicts
    """
    if not text.strip():
        return []

    token_count = count_tokens(text)
    doc_class = classify_document(item_type, token_count)

    page_map = _build_page_map(text)
    page_at_offset = _build_page_at_offset(text)

    stripped = text.replace('\f', '')

    if doc_class == 'short':
        result = [_make_chunk(stripped, 0, 1, metadata)]
        _assign_pages_by_position(result, stripped, page_at_offset, page_map)
        return result

    if doc_class == 'long':
        if item_type == 'hearing':
            separators = HEARING_SEPARATORS
        else:
            separators = SECTION_SEPARATORS
        chunks = _split_at_boundaries(stripped, separators)
    elif doc_class == 'medium':
        separators = SECTION_SEPARATORS + [r'\n\n']
        chunks = _split_at_boundaries(stripped, separators)
    else:
        chunks = _split_by_tokens(stripped)

    total = len(chunks)
    result = [_make_chunk(chunk_text, i, total, metadata) for i, chunk_text in enumerate(chunks)]
    _assign_pages_by_position(result, stripped, page_at_offset, page_map)
    return result


def chunk_epub(chapters, metadata):
    """Chunk an EPUB by chapters, sub-splitting long chapters."""
    all_chunks = []
    chunk_index = 0

    for chapter_title, chapter_text in chapters:
        if not chapter_text.strip():
            continue

        token_count = count_tokens(chapter_text)
        chapter_meta = {**metadata, 'chapter': chapter_title}

        if token_count <= CHUNK_SIZE_TOKENS:
            all_chunks.append(_make_chunk(chapter_text, chunk_index, -1, chapter_meta))
            chunk_index += 1
        else:
            sub_chunks = _split_at_boundaries(
                chapter_text,
                SECTION_SEPARATORS + [r'\n\n'],
            )
            for sc in sub_chunks:
                all_chunks.append(_make_chunk(sc, chunk_index, -1, chapter_meta))
                chunk_index += 1

    total = len(all_chunks)
    for c in all_chunks:
        c['total_chunks'] = total

    return all_chunks


def chunk_note(text, metadata, source_type='child_note'):
    """Chunk a Zotero note."""
    if not text.strip():
        return []

    note_meta = {**metadata, 'source_type': source_type}
    token_count = count_tokens(text)

    if token_count <= CHUNK_SIZE_TOKENS:
        return [_make_chunk(text, 0, 1, note_meta)]

    chunks = _split_at_boundaries(text, [r'\n\n'])
    total = len(chunks)
    return [_make_chunk(ct, i, total, note_meta) for i, ct in enumerate(chunks)]


def _build_page_map(text):
    """Detect printed page numbers from the first line of each PDF page."""
    pages = text.split('\f')
    detected = {}

    for i, page_text in enumerate(pages):
        pdf_page = i + 1
        lines = page_text.strip().split('\n')
        if not lines:
            continue
        first_line = lines[0].strip()
        m = re.match(r'^(\d{1,5})$', first_line)
        if m:
            detected[pdf_page] = int(m.group(1))

    total_pages = len(pages)
    if not detected or len(detected) / max(total_pages, 1) < 0.3:
        return None

    page_map = {}
    sorted_detected = sorted(detected.items())

    for pdf_page in range(1, total_pages + 1):
        if pdf_page in detected:
            page_map[pdf_page] = detected[pdf_page]
        else:
            best_dist = float('inf')
            best_offset = 0
            for det_pdf, det_printed in sorted_detected:
                dist = abs(pdf_page - det_pdf)
                if dist < best_dist:
                    best_dist = dist
                    best_offset = det_pdf - det_printed
            page_map[pdf_page] = max(1, pdf_page - best_offset)

    return page_map


def _build_page_at_offset(text):
    """Build an array mapping each character offset in \\f-stripped text to its PDF page."""
    result = []
    pdf_page = 1
    for ch in text:
        if ch == '\f':
            pdf_page += 1
        else:
            result.append(pdf_page)
    return result


def _assign_pages_by_position(chunks, stripped_text, page_at_offset, page_map):
    """Assign page numbers to chunks by finding their position in the original text."""
    search_start = 0
    for chunk in chunks:
        text = chunk['text']
        needle = text[:min(100, len(text))]
        pos = stripped_text.find(needle, max(0, search_start - 500))
        if pos >= 0 and pos < len(page_at_offset):
            pdf_start = page_at_offset[pos]
            end_pos = min(pos + len(text) - 1, len(page_at_offset) - 1)
            pdf_end = page_at_offset[end_pos] if end_pos >= 0 else pdf_start
            search_start = pos + 1
        else:
            pdf_start = search_start
            pdf_end = pdf_start

        if page_map:
            chunk['metadata']['page_start'] = page_map.get(pdf_start, pdf_start)
            chunk['metadata']['page_end'] = page_map.get(pdf_end, pdf_end)
        else:
            chunk['metadata']['page_start'] = pdf_start
            chunk['metadata']['page_end'] = pdf_end
        chunk['metadata']['pdf_page'] = pdf_start


def _make_chunk(text, chunk_index, total_chunks, metadata):
    return {
        'text': text.strip(),
        'chunk_index': chunk_index,
        'total_chunks': total_chunks,
        'metadata': dict(metadata),
    }
