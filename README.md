# Zotero RAG: AI-Powered Research Assistant for Your Zotero Library

Search your Zotero library using natural language and get AI-generated answers grounded in your actual sources, with citations and links back to the original documents.

## Why This Exists

I built this tool while writing a history dissertation. My Zotero library had hundreds of PDFs -- books, journal articles, archival manuscripts, congressional hearings, government reports -- and I needed a way to search across all of them by meaning, not just keywords.

Zotero's built-in search is limited to exact text matching, which doesn't help when the same concept appears under different language across different decades and document types. What I wanted was something like Google's NotebookLM -- a "chat with your sources" interface where you can ask questions and get grounded, cited answers -- but connected to Zotero instead of uploaded files, and able to handle a full research library rather than a handful of documents.

This tool recreates that NotebookLM-style experience on top of your Zotero library. It indexes all of your PDFs and EPUBs into a searchable vector database, then lets you ask questions in natural language. The AI retrieves the most relevant passages, synthesizes an answer citing specific sources with `[1]`, `[2]` notation, and gives you clickable links that open the original PDF at the exact page in Zotero. Unlike NotebookLM, it works with hundreds or thousands of documents at once, integrates with your existing Zotero workflow, and gives you full control over the AI provider and your data.

It's designed for the kinds of materials historians work with:
- **Books and book chapters** -- chunked at chapter and section boundaries
- **Journal articles** -- kept whole when short, split intelligently when long
- **Archival primary sources** -- manuscripts, letters, memoranda with archive/collection metadata preserved
- **Congressional hearings** -- split at speaker boundaries so testimony stays coherent
- **Government reports, statutes, meeting minutes** -- each with appropriate structural parsing

No coding knowledge is required to set it up. If you can edit a text file and run a few commands in the terminal, you can get this running.

## What It Does

1. **Indexes your Zotero library** -- extracts text from your PDFs and EPUBs, splits them into searchable chunks, and stores them in a vector database
2. **Semantic search** -- find relevant passages by meaning, not just keywords (e.g., "arguments against deregulation" finds passages about "opposition to removing regulatory barriers")
3. **AI-powered Q&A** -- ask questions in a chat interface and get answers that cite your sources with `[1]`, `[2]` notation
4. **Deep links to Zotero** -- click any source to open the original PDF at the exact page in Zotero

### Three Ways to Use It

- **Web app** (`webapp.py`) -- a browser-based chat interface with filters and source cards

<img width="1602" height="857" alt="image" src="https://github.com/user-attachments/assets/8417971c-1894-492e-a864-15f52cb732d9" />

- **CLI search** (`search.py`) -- quick terminal search, no AI needed (just embeddings)
- **Claude Desktop integration** (`server.py`) -- use as an MCP server inside Claude Desktop

## Cost Options

| Setup | Indexing Cost | Per-Query Cost | Notes |
|-------|-------------|----------------|-------|
| **OpenAI embeddings + Anthropic chat** | ~$0.10 per 1,000 docs | ~$0.001/query (search) + ~$0.01/query (chat) | Recommended. Best answer quality |
| **OpenAI embeddings + GPT-4o-mini chat** | ~$0.10 per 1,000 docs | ~$0.001/query (search) + ~$0.001/query (chat) | Cheapest paid option |
| **Ollama + Pinecone** | Free (but slower) | Free (search) + Free (chat) | Requires decent hardware (8GB+ RAM) |
| **OpenAI embeddings + Ollama chat** | ~$0.10 per 1,000 docs | ~$0.001/query (search) + Free (chat) | Good middle ground |

Pinecone's free tier supports up to ~100,000 chunks, which is enough for most research libraries.

---

## Setup Guide

### Prerequisites

- **Python 3.10+** -- [Download Python](https://www.python.org/downloads/)
- **Zotero 6 or 7** -- [Download Zotero](https://www.zotero.org/download/)
- **pdftotext** -- for extracting text from PDFs (see below)

#### Install pdftotext

- **macOS:** `brew install poppler` (install [Homebrew](https://brew.sh) first if you don't have it)
- **Ubuntu/Debian:** `sudo apt install poppler-utils`
- **Windows:** Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases) and add to your PATH

### Step 1: Download the Code

```bash
git clone https://github.com/freshaf/zotero-rag.git
cd zotero-rag
```

Or download and unzip from the GitHub page (click the green "Code" button > "Download ZIP").

### Step 2: Set Up Python Environment

Open a terminal in the `zotero-rag` folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

You'll need to run `source .venv/bin/activate` each time you open a new terminal.

### Step 3: Get Your API Keys

You need three things: Zotero API credentials, a Pinecone account, and an embedding provider. If you want to use the **web app chat interface** (AI-generated answers, not just search), you'll also need an LLM API key (see below).

#### Zotero API Key

1. Go to https://www.zotero.org/settings/keys
2. Note your **userID** (shown at the top -- this is your Library ID)
3. Click **"Create new private key"**
4. Name it (e.g., "RAG Pipeline")
5. Under **Personal Library**, check **"Allow library access"**
6. Click **Save Key** and copy the key

#### Pinecone (Free)

1. Sign up at https://app.pinecone.io (free, no credit card)
2. After signing in, go to **API Keys** in the left sidebar
3. Copy your API key

#### Embeddings (choose one)

**Option A -- OpenAI (recommended):**
1. Sign up at https://platform.openai.com
2. Go to https://platform.openai.com/api-keys
3. Click **"Create new secret key"** and copy it
4. Add a few dollars of credit ($5 is plenty for thousands of documents)

**Option B -- Ollama (free, local):**
1. Download from https://ollama.ai
2. Install and open Ollama
3. In your terminal, run: `ollama pull nomic-embed-text`
4. For chat, also run: `ollama pull llama3.1`

#### LLM for Chat (optional -- needed for web app Q&A)

The web app's chat feature requires an LLM to synthesize answers from your sources. Search and indexing work without one. Choose one:

**Option A -- Anthropic Claude (recommended):**
1. Sign up at https://console.anthropic.com
2. Go to https://console.anthropic.com/settings/keys
3. Click **"Create Key"** and copy it
4. Add a few dollars of credit ($5 goes a long way)
5. In your `.env`, set:
   ```
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=your-key-here
   ```
   The default model is Claude Sonnet, which produces excellent research answers with citations.

**Option B -- OpenAI GPT-4o-mini (cheaper):**
If you're already using OpenAI for embeddings, this is the easiest option -- no extra API key needed.
1. Uses the same `OPENAI_API_KEY` from the embeddings step
2. In your `.env`, set:
   ```
   LLM_PROVIDER=openai
   ```
   The default model is GPT-4o-mini, which costs ~$0.15 per million tokens -- roughly 10x cheaper than Claude Sonnet per query, with slightly lower quality.

**Option C -- Ollama (free, local):**
If you already installed Ollama for embeddings, just pull a chat model:
```bash
ollama pull llama3.1
```
In your `.env`, set `LLM_PROVIDER=ollama`. Requires decent hardware (8GB+ RAM).

### Step 4: Configure

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in your values. The file has detailed comments explaining each setting. At minimum, you need:

```
ZOTERO_LIBRARY_ID=your-library-id
ZOTERO_API_KEY=your-api-key
PINECONE_API_KEY=your-pinecone-key
```

**Optional:** To index only a specific collection (instead of your entire library), add the collection key:

```
ZOTERO_COLLECTION_KEY=ABCD1234
```

To find a collection key: in Zotero, right-click a collection > Copy Collection Link. The key is the 8-character code at the end of the URL.

### Step 5: Index Your Library

```bash
source .venv/bin/activate
python index.py
```

This will:
- Connect to your Zotero library via the API
- Download and extract text from each PDF/EPUB
- Split documents into searchable chunks
- Generate embeddings and store them in Pinecone

**How long does it take?** Depends on library size and embedding provider:
- ~100 items with OpenAI: a few minutes
- ~1,000 items with OpenAI: 15-30 minutes
- With Ollama: 3-5x slower (runs on your CPU/GPU)

Text extraction results are cached in the `cache/` folder, so re-indexing is faster.

### Step 6: Start Searching

**Web interface (recommended):**

```bash
python webapp.py
```

Open http://localhost:5001 in your browser. Type a question and get AI-powered answers with citations.

**Command-line search (no LLM needed):**

```bash
python search.py "your search query"
python search.py "banking reform type:hearing by:Volcker"
```

---

## Usage

### Web App

The web app has three panels:

- **Left: Filters** -- narrow results by item type, archive, date range, author, tag, or collection
- **Center: Chat** -- ask questions in natural language; the AI answers using your sources
- **Right: Sources** -- see which documents were retrieved, with links to open them in Zotero

Click any `[N]` citation in the response to highlight the corresponding source card.

### Search Filters

You can use filters in the sidebar or type shorthand prefixes directly in your query:

| Prefix | Filters by | Example |
|--------|-----------|---------|
| `type:` | Zotero item type | `type:hearing interest rate policy` |
| `by:` | Author | `by:Smith monetary policy` |
| `in:` | Archive/collection | `in:"National Archives" war records` |
| `tag:` | Zotero tag | `tag:economics trade policy` |
| `collection:` | Collection name | `collection:"Chapter 3" sources` |
| `from:` / `to:` | Date range | `from:1941 to:1945 wartime production` |
| `top:` | Number of results | `top:20 labor movement` |

### Keeping Your Index Updated

After adding new items to Zotero:

```bash
source .venv/bin/activate
python index.py --update    # incremental -- only new/changed items
python index.py             # full re-index
```

### Claude Desktop Integration (MCP Server)

To use this as a tool inside Claude Desktop:

1. Open Claude Desktop settings
2. Go to Developer > Model Context Protocol
3. Add this configuration (adjust the path to match your setup):

```json
{
  "mcpServers": {
    "zotero-rag": {
      "command": "/path/to/zotero-rag/.venv/bin/python3",
      "args": ["/path/to/zotero-rag/server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key",
        "PINECONE_API_KEY": "your-key",
        "ZOTERO_LIBRARY_ID": "your-id",
        "ZOTERO_API_KEY": "your-key"
      }
    }
  }
}
```

Then you can ask Claude to "search my Zotero library for..." and it will use the semantic search tool.

---

## Architecture

```
Zotero Web API
    |
    v
Text Extraction (PDF/EPUB/HTML)  -->  cached in cache/
    |
    v
Adaptive Chunking (by document type: books, hearings, articles, etc.)
    |
    v
Embeddings (OpenAI or Ollama)
    |
    v
Pinecone Vector Database
    |
    v
Search: embed query --> Pinecone --> filter --> FlashRank reranking
    |
    v
Web App / CLI / MCP Server
```

### Key Design Decisions

- **Adaptive chunking**: Congressional hearings are split at speaker boundaries, books at chapter boundaries, short articles kept whole. This preserves context better than uniform chunking.
- **Metadata-enriched embeddings**: Each chunk is embedded with a header containing title, author, date, and archive info, so the vector captures document context alongside the text content.
- **FlashRank reranking**: After Pinecone returns candidates by vector similarity, a cross-encoder reranker (runs locally, no API cost) improves result ordering.
- **Incremental sync**: The `--update` flag only processes new or changed items, tracked via Zotero's library version number.

---

## Troubleshooting

**"pdftotext not found"**
Install poppler (see Prerequisites above).

**"OPENAI_API_KEY not set"**
Make sure you copied `.env.example` to `.env` and filled in your keys.

**"Cannot reach Ollama"**
Make sure Ollama is running. Open the Ollama app or run `ollama serve` in a terminal.

**"Pinecone index dimension mismatch"**
If you switch embedding providers after already creating an index, you need to delete the old Pinecone index (in the Pinecone dashboard) and re-run `python index.py`.

**Indexing is very slow**
- With Ollama, indexing is CPU-bound. A GPU helps significantly.
- OpenAI embeddings are much faster (batched API calls).
- Text extraction is cached, so re-runs skip already-extracted files.

**"No results found"**
- Make sure you've run `python index.py` first
- Check that your Zotero items have PDF/EPUB attachments (items without attachments get minimal indexing based on metadata only)

---

## Project Structure

```
zotero-rag/
|-- .env.example          # Configuration template
|-- requirements.txt      # Python dependencies
|-- index.py              # Run this to index your library
|-- webapp.py             # Web chat interface
|-- search.py             # CLI search (no LLM needed)
|-- server.py             # MCP server for Claude Desktop
|-- static/
|   +-- index.html        # Web app frontend
+-- src/
    |-- config.py          # Configuration loader
    |-- zotero_client.py   # Zotero API integration
    |-- extractors.py      # PDF/EPUB/HTML text extraction
    |-- chunker.py         # Adaptive document chunking
    |-- embeddings.py      # OpenAI/Ollama embedding client
    |-- vectordb.py        # Pinecone operations
    |-- indexer.py         # Main indexing pipeline
    +-- search_pipeline.py # Shared search logic
```

## License

MIT
