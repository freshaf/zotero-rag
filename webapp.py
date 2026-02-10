#!/usr/bin/env python3
"""Web interface for Zotero RAG with AI-powered Q&A.

Supports multiple LLM providers: Anthropic Claude, OpenAI, or Ollama (free/local).

Usage:
    source .venv/bin/activate
    python webapp.py

Then open http://localhost:5001 in your browser.
"""

import json
import logging
import os
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse

from src.config import (
    LLM_PROVIDER, ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    OPENAI_API_KEY, OPENAI_CHAT_MODEL,
    OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL,
    ARCHIVE_ALIASES_FILE,
)
from src.search_pipeline import init_pipeline, run_search, get_archive_aliases

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

SYSTEM_PROMPT = """You are a research assistant helping a scholar with their research. You answer questions using ONLY the provided source documents.

Rules:
1. Answer ONLY from the provided sources. Do not use outside knowledge.
2. Cite sources using [N] notation matching the source numbers provided.
3. If the sources are insufficient to fully answer the question, explicitly state what information is missing.
4. Be neutral and professional. Report what the sources say without editorializing.
5. When quoting, use exact text from the sources.
6. End your response with a "Sources" section listing all cited sources.

The user may ask follow-up questions -- use conversation context plus any new sources provided."""


def _build_source_context(sources):
    """Format search results into a context block for the LLM."""
    if not sources:
        return "\n[No sources found for this query.]\n"
    parts = ["\n--- BEGIN SOURCES ---"]
    for i, s in enumerate(sources, 1):
        authors = ', '.join(s.get('authors', [])) or 'Unknown'
        title = s.get('title', 'Untitled')
        date = s.get('date', '')
        item_type = s.get('item_type', '')
        text = s.get('text', '')

        page_start = s.get('page_start', 0)
        page_end = s.get('page_end', 0)
        page_str = ''
        if page_start > 0:
            page_str = f", pp. {page_start}-{page_end}" if page_end > page_start else f", p. {page_start}"

        archive = s.get('archive', '')
        archive_loc = s.get('archive_location', '')
        archive_str = ''
        if archive:
            archive_str = f"\nArchive: {archive}"
            if archive_loc:
                archive_str += f", {archive_loc}"

        parts.append(
            f"\n[{i}] \"{title}\" -- {authors}"
            f"{' (' + date + ')' if date else ''}{page_str}"
            f"\nType: {item_type}{archive_str}"
            f"\n{text}\n"
        )
    parts.append("--- END SOURCES ---\n")
    return '\n'.join(parts)


def _format_source_for_client(result):
    """Convert a search result dict into the format sent to the frontend."""
    meta = result['metadata']
    attachment_key = meta.get('attachment_key', '')
    attachment_type = meta.get('attachment_type', 'pdf')
    zotero_key = meta.get('zotero_key', '')
    pdf_page = meta.get('pdf_page', meta.get('page_start', 0))

    zotero_url = ''
    if attachment_key:
        zotero_url = f"/zotero/pdf/{attachment_key}"
        if pdf_page and attachment_type == 'pdf':
            zotero_url += f"?page={pdf_page}"
    elif zotero_key:
        zotero_url = f"/zotero/item/{zotero_key}"

    return {
        'title': meta.get('title', 'Untitled'),
        'authors': meta.get('authors', []),
        'date': meta.get('date', ''),
        'item_type': meta.get('item_type', ''),
        'archive': meta.get('archive', ''),
        'archive_location': meta.get('archive_location', ''),
        'page_start': meta.get('page_start', 0),
        'page_end': meta.get('page_end', 0),
        'text': meta.get('text', '')[:600],
        'zotero_url': zotero_url,
        'score': float(result.get('score', 0)),
        'rerank_score': float(result['rerank_score']) if result.get('rerank_score') is not None else None,
    }


def _stream_anthropic(messages, system_prompt):
    """Stream response from Anthropic Claude."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _stream_openai(messages, system_prompt):
    """Stream response from OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    stream = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=full_messages,
        max_tokens=4096,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def _stream_ollama(messages, system_prompt):
    """Stream response from Ollama (local)."""
    import urllib.request
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    payload = json.dumps({
        "model": OLLAMA_CHAT_MODEL,
        "messages": full_messages,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            if line.strip():
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content


def _get_llm_stream(messages, system_prompt):
    """Get the appropriate LLM stream based on config."""
    provider = LLM_PROVIDER.lower()
    if provider == "anthropic":
        return _stream_anthropic(messages, system_prompt)
    elif provider == "ollama":
        return _stream_ollama(messages, system_prompt)
    else:
        return _stream_openai(messages, system_prompt)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/filters")
async def filters():
    """Return available filter values for sidebar dropdowns."""
    aliases = get_archive_aliases()

    archive_options = []
    for acronym, full_name in sorted(aliases.items(), key=lambda x: x[1]):
        archive_options.append({
            'value': acronym,
            'label': f"{acronym.upper()} -- {full_name}",
        })

    item_types = [
        'book', 'bookSection', 'conferencePaper', 'document',
        'hearing', 'journalArticle', 'letter', 'manuscript',
        'newspaperArticle', 'report', 'statute', 'thesis', 'webpage',
    ]

    return JSONResponse({
        'archives': archive_options,
        'item_types': item_types,
    })


@app.post("/api/chat")
async def chat(request: Request):
    """Search + stream LLM response via SSE."""
    body = await request.json()
    message = body.get('message', '').strip()
    prev_conversation = body.get('conversation', [])
    filter_vals = body.get('filters', {})
    top_k = body.get('top_k', 10)

    if not message:
        return JSONResponse({'error': 'Empty message'}, status_code=400)

    async def generate():
        try:
            results = run_search(
                message,
                top_k=top_k,
                item_type=filter_vals.get('item_type') or None,
                author=filter_vals.get('author') or None,
                tag=filter_vals.get('tag') or None,
                collection=filter_vals.get('collection') or None,
                archive=filter_vals.get('archive') or None,
                date_from=filter_vals.get('date_from') or None,
                date_to=filter_vals.get('date_to') or None,
            )

            client_sources = [_format_source_for_client(r) for r in results]
            yield f"data: {json.dumps({'type': 'sources', 'sources': client_sources})}\n\n"

            source_context = _build_source_context([
                {
                    'title': r['metadata'].get('title', ''),
                    'authors': r['metadata'].get('authors', []),
                    'date': r['metadata'].get('date', ''),
                    'item_type': r['metadata'].get('item_type', ''),
                    'archive': r['metadata'].get('archive', ''),
                    'archive_location': r['metadata'].get('archive_location', ''),
                    'page_start': r['metadata'].get('page_start', 0),
                    'page_end': r['metadata'].get('page_end', 0),
                    'text': r['metadata'].get('text', ''),
                }
                for r in results
            ])

            messages = []
            for msg in prev_conversation:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content'],
                })

            user_content = f"{source_context}\n\nUser question: {message}"
            messages.append({'role': 'user', 'content': user_content})

            for text in _get_llm_stream(messages, SYSTEM_PROMPT):
                yield f"data: {json.dumps({'type': 'delta', 'text': text})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.exception("Chat error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/zotero/pdf/{key}")
async def open_pdf(key: str, page: int = 0):
    """Open a PDF in Zotero via zotero:// URL."""
    url = f"zotero://open-pdf/library/items/{key}"
    if page:
        url += f"?page={page}"
    # macOS
    if sys.platform == "darwin":
        subprocess.Popen(['open', url])
    # Linux
    elif sys.platform.startswith("linux"):
        subprocess.Popen(['xdg-open', url])
    # Windows
    elif sys.platform == "win32":
        os.startfile(url)
    return HTMLResponse(
        '<html><body><p>Opened in Zotero.</p>'
        '<script>window.close()</script></body></html>'
    )


@app.get("/zotero/item/{key}")
async def open_item(key: str):
    """Open a Zotero item via zotero:// URL."""
    url = f"zotero://select/library/items/{key}"
    if sys.platform == "darwin":
        subprocess.Popen(['open', url])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(['xdg-open', url])
    elif sys.platform == "win32":
        os.startfile(url)
    return HTMLResponse(
        '<html><body><p>Opened in Zotero.</p>'
        '<script>window.close()</script></body></html>'
    )


if __name__ == '__main__':
    import uvicorn

    print("Initializing search pipeline...", end=' ', flush=True)
    init_pipeline()
    print("done.")
    print(f"Using LLM provider: {LLM_PROVIDER}")
    print("Open http://localhost:5001 in your browser.")
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="warning")
