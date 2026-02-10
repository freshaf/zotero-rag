#!/usr/bin/env python3
"""MCP server for Zotero RAG semantic search.

Integrates with Claude Desktop for natural language search over your Zotero library.
"""

import json
import logging
import os
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

from src.search_pipeline import init_pipeline, run_search, get_archive_aliases
from src.vectordb import get_index_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("zotero-rag")

LINK_SERVER_PORT = 19285

_last_results = []


class _ZoteroLinkHandler(BaseHTTPRequestHandler):
    """Handles localhost requests by opening zotero:// URLs."""

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path.startswith('/pdf/'):
            key = parsed.path[5:]
            url = f"zotero://open-pdf/library/items/{key}"
            page = params.get('page', [None])[0]
            if page:
                url += f"?page={page}"
        elif parsed.path.startswith('/item/'):
            key = parsed.path[6:]
            url = f"zotero://select/library/items/{key}"
        else:
            self.send_response(404)
            self.end_headers()
            return

        if sys.platform == "darwin":
            subprocess.Popen(['open', url])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(['xdg-open', url])
        elif sys.platform == "win32":
            os.startfile(url)

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<html><body><p>Opened in Zotero.</p>'
                         b'<script>window.close()</script></body></html>')

    def log_message(self, format, *args):
        pass


def _start_link_server():
    """Start the localhost link server in a background thread."""
    try:
        server = HTTPServer(('127.0.0.1', LINK_SERVER_PORT), _ZoteroLinkHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Zotero link server running on http://127.0.0.1:{LINK_SERVER_PORT}")
    except OSError as e:
        logger.warning(f"Could not start link server on port {LINK_SERVER_PORT}: {e}")


@mcp.tool()
async def search_zotero(
    query: str,
    top_k: int = 10,
    item_type: Optional[str] = None,
    author: Optional[str] = None,
    tag: Optional[str] = None,
    collection: Optional[str] = None,
    archive: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """Search your Zotero library using semantic search.

    The query parameter is for WHAT the content is ABOUT (semantic search by meaning).
    Filters narrow WHICH documents to search within.

    Shorthand prefixes can be embedded in the query:
        type:hearing  by:Volcker  tag:monetary-policy  in:"Reagan Library"
        from:1981  to:1985  collection:DTRP  top:5

    Args:
        query: What the content should be about (include time periods, events, topics)
        top_k: Number of results to return (default 10, max 20)
        item_type: Filter by Zotero item type (e.g., "hearing", "manuscript", "book")
        author: Filter by author name (partial match)
        tag: Filter by Zotero tag (partial match)
        collection: Filter by collection name (partial match)
        archive: Filter by archive collection (partial match, prefix with = for exact)
        date_from: Restrict to documents dated on or after this (YYYY or YYYY-MM-DD)
        date_to: Restrict to documents dated on or before this (YYYY or YYYY-MM-DD)
    """
    results = run_search(
        query, top_k=top_k, item_type=item_type, author=author,
        tag=tag, collection=collection, archive=archive,
        date_from=date_from, date_to=date_to,
    )

    global _last_results
    _last_results = results

    if not results:
        return "No results found."

    base = f"http://127.0.0.1:{LINK_SERVER_PORT}"
    output_parts = [
        f"Found {len(results)} results.\n\n"
        "FORMATTING RULES:\n"
        "1. Write citations as [[N]](url) using the link from each source.\n"
        "2. At the END of your response, copy the SOURCES block below as-is.\n"
        "3. Be neutral and professional.\n\n"
    ]
    source_lines = []
    for i, r in enumerate(results, 1):
        meta = r['metadata']
        score = r['score']

        authors = ', '.join(meta.get('authors', [])) or 'Unknown'
        title = meta.get('title', 'Untitled')
        date = meta.get('date', '')
        item_type_str = meta.get('item_type', '')

        citation = f'"{title}"'
        if authors != 'Unknown':
            citation += f" -- {authors}"
        if date:
            citation += f" ({date})"

        attachment_type = meta.get('attachment_type', 'pdf')
        page_start = meta.get('page_start', 0)
        page_end = meta.get('page_end', 0)
        chapter = meta.get('chapter', '')
        page_str = ''
        if attachment_type == 'epub' and chapter:
            page_str = f", ch. \"{chapter}\""
        elif page_start > 0:
            if page_end > page_start:
                page_str = f", pp. {page_start}-{page_end}"
            else:
                page_str = f", p. {page_start}"

        zotero_key = meta.get('zotero_key', '')
        attachment_key = meta.get('attachment_key', '')
        pdf_page = meta.get('pdf_page', page_start)
        if attachment_key:
            if attachment_type == 'pdf' and pdf_page > 0:
                link_url = f"{base}/pdf/{attachment_key}?page={pdf_page}"
            else:
                link_url = f"{base}/pdf/{attachment_key}"
        elif zotero_key:
            link_url = f"{base}/item/{zotero_key}"
        else:
            link_url = ""

        archive_info = ''
        arch = meta.get('archive', '')
        arch_loc = meta.get('archive_location', '')
        if arch:
            archive_info = f"\n   Archive: {arch}"
            if arch_loc:
                archive_info += f", {arch_loc}"

        text_preview = meta.get('text', '')[:800]
        if len(meta.get('text', '')) > 800:
            text_preview += '...'

        rerank_score = r.get('rerank_score')
        score_str = f"Embed: {score:.3f}"
        if rerank_score is not None:
            score_str += f" | Rerank: {rerank_score:.3f}"

        output_parts.append(
            f"--- SOURCE [{i}] {citation}{page_str} ---\n"
            f"   link: {link_url}\n"
            f"   Type: {item_type_str} | {score_str}"
            f"{archive_info}\n"
            f"   TEXT: {text_preview}\n"
        )

        source_lines.append(f"- [[{i}]]({link_url}) {citation}{page_str}")

    output_parts.append(
        "\n---\nSOURCES (copy this block verbatim at the end of your response):\n\n"
        + '\n'.join(source_lines)
    )

    return '\n'.join(output_parts)


@mcp.tool()
async def open_zotero_source(
    citation_number: Optional[int] = None,
    zotero_key: Optional[str] = None,
    attachment_key: Optional[str] = None,
    page: Optional[int] = None,
) -> str:
    """Open a Zotero source directly on your computer.

    Use this when the user wants to open/view a source from search results.

    Args:
        citation_number: The [N] citation number from the last search results.
        zotero_key: Zotero item key (fallback if no citation_number)
        attachment_key: Zotero attachment key for PDF (fallback)
        page: Page number to open to (fallback)
    """
    if citation_number is not None:
        idx = citation_number - 1
        if idx < 0 or idx >= len(_last_results):
            return f"Citation [{citation_number}] not found. Last search had {len(_last_results)} results."
        meta = _last_results[idx]['metadata']
        zotero_key = meta.get('zotero_key', '')
        attachment_key = meta.get('attachment_key', '')
        page = meta.get('pdf_page', meta.get('page_start', 0)) or None

    if attachment_key:
        url = f"zotero://open-pdf/library/items/{attachment_key}"
        if page:
            url += f"?page={page}"
        label = f"PDF (p. {page})" if page else "PDF"
    elif zotero_key:
        url = f"zotero://select/library/items/{zotero_key}"
        label = "item"
    else:
        return "No zotero_key or attachment_key provided, and no citation_number to look up."

    if sys.platform == "darwin":
        subprocess.Popen(['open', url])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(['xdg-open', url])
    elif sys.platform == "win32":
        os.startfile(url)

    return f"Opened {label} in Zotero: {url}"


@mcp.tool()
async def zotero_index_stats() -> str:
    """Get statistics about the indexed Zotero collection."""
    init_pipeline()
    stats = get_index_stats()
    total = stats.total_vector_count
    return f"Zotero RAG index: {total} chunks indexed"


def main():
    logger.info("Zotero RAG MCP Server starting...")
    _start_link_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
