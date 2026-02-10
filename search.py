#!/usr/bin/env python3
"""Standalone CLI for searching the Zotero RAG index.

No LLM required -- just embeds your query and searches Pinecone.

Usage:
    python search.py "monetary policy"
    python search.py "banking reform type:hearing by:Volcker"
    python search.py "deregulation from:1983" --top 5

Shorthand filters:
    type:hearing  by:Author  tag:topic  in:"Archive Name"
    from:1981  to:1985  collection:Name  top:5

After results display, enter a number to open that source in Zotero,
type a new query to search again, or 'q' to quit.
"""

import argparse
import os
import re
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

from src.search_pipeline import init_pipeline, run_search
from src.vectordb import get_index_stats


def format_results(results, query_str):
    if not results:
        return "No results found."

    lines = [f'Search: "{query_str}"', f"{len(results)} results", ""]

    for i, r in enumerate(results, 1):
        meta = r['metadata']

        authors = ', '.join(meta.get('authors', [])) or 'Unknown'
        title = meta.get('title', 'Untitled')
        date = meta.get('date', '')
        item_type = meta.get('item_type', '')

        page_start = meta.get('page_start', 0)
        page_end = meta.get('page_end', 0)
        page_str = ''
        if page_start > 0:
            page_str = f", pp. {page_start}-{page_end}" if page_end > page_start else f", p. {page_start}"

        rerank_score = r.get('rerank_score')
        score_str = f"Embed: {r['score']:.3f}"
        if rerank_score is not None:
            score_str += f" | Rerank: {rerank_score:.3f}"

        archive_info = ''
        arch = meta.get('archive', '')
        arch_loc = meta.get('archive_location', '')
        if arch:
            archive_info = f"  Archive: {arch}"
            if arch_loc:
                archive_info += f", {arch_loc}"

        text = meta.get('text', '').strip()
        preview = text[:600]
        if len(text) > 600:
            preview += '...'

        lines.append(f"[{i}] {title}")
        lines.append(f"    {authors} ({date}) -- {item_type}{page_str}")
        lines.append(f"    {score_str}")
        if archive_info:
            lines.append(f"   {archive_info}")
        lines.append(f"    {preview}")
        lines.append("")

    return '\n'.join(lines)


def open_result(results, number):
    """Open a result in Zotero by its [N] number."""
    idx = number - 1
    if idx < 0 or idx >= len(results):
        print(f"No result [{number}]. Got {len(results)} results.")
        return
    meta = results[idx]['metadata']
    attachment_key = meta.get('attachment_key', '')
    zotero_key = meta.get('zotero_key', '')
    page = meta.get('pdf_page', meta.get('page_start', 0))

    if attachment_key:
        url = f"zotero://open-pdf/library/items/{attachment_key}"
        if page:
            url += f"?page={page}"
    elif zotero_key:
        url = f"zotero://select/library/items/{zotero_key}"
    else:
        print("No Zotero key for this result.")
        return

    if sys.platform == "darwin":
        subprocess.Popen(['open', url])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(['xdg-open', url])
    elif sys.platform == "win32":
        os.startfile(url)
    print(f"Opened in Zotero: {url}")


def main():
    parser = argparse.ArgumentParser(
        description='Search your Zotero library (no LLM needed).',
        epilog='Filters: type:hearing by:Author tag:topic in:Archive from:1981 to:1985',
    )
    parser.add_argument('query', nargs='*', help='Search query with optional shorthand filters')
    parser.add_argument('--top', '-n', type=int, default=10, help='Number of results (default: 10)')
    parser.add_argument('--stats', action='store_true', help='Show index stats and exit')
    args = parser.parse_args()

    print("Initializing...", end=' ', flush=True)
    init_pipeline()
    print("done.")

    if args.stats:
        stats = get_index_stats()
        print(f"Index: {stats.total_vector_count} vectors")
        return

    if not args.query:
        parser.print_help()
        return

    query_str = ' '.join(args.query)
    results = run_search(query_str, top_k=args.top)
    print(format_results(results, query_str))

    if results and sys.stdin.isatty():
        print("Enter a number to open in Zotero, a new query to search, or 'q' to quit.")
        while True:
            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not cmd or cmd.lower() in ('q', 'quit', 'exit'):
                break
            m = re.match(r'^(?:open\s+)?(\d+)$', cmd, re.IGNORECASE)
            if m:
                open_result(results, int(m.group(1)))
                continue
            results = run_search(cmd, top_k=args.top)
            print(format_results(results, cmd))
            if results:
                print("Enter a number to open in Zotero, a new query to search, or 'q' to quit.")


if __name__ == '__main__':
    main()
