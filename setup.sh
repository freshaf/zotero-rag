#!/bin/bash
# Zotero RAG - Quick Setup Script
# Run this after cloning the repository.

set -e

echo "=== Zotero RAG Setup ==="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Download it from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check pdftotext
if ! command -v pdftotext &> /dev/null; then
    echo ""
    echo "WARNING: pdftotext is not installed (needed for PDF text extraction)."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Install with: brew install poppler"
    elif [[ "$OSTYPE" == "linux"* ]]; then
        echo "  Install with: sudo apt install poppler-utils"
    else
        echo "  Download from: https://github.com/oschwartz10612/poppler-windows/releases"
    fi
    echo ""
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate and install dependencies
echo "Installing dependencies..."
source .venv/bin/activate
pip install -q -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "============================================"
    echo "  IMPORTANT: Edit .env with your API keys"
    echo "============================================"
    echo ""
    echo "Open .env in a text editor and fill in:"
    echo "  1. ZOTERO_LIBRARY_ID  (from https://www.zotero.org/settings/keys)"
    echo "  2. ZOTERO_API_KEY     (create at the same page)"
    echo "  3. PINECONE_API_KEY   (sign up free at https://app.pinecone.io)"
    echo "  4. OPENAI_API_KEY     (from https://platform.openai.com/api-keys)"
    echo "     OR set EMBEDDING_PROVIDER=ollama for free local embeddings"
    echo ""
else
    echo ".env already exists (not overwriting)."
fi

echo ""
echo "Setup complete! Next steps:"
echo ""
echo "  1. Edit .env with your API keys (if you haven't already)"
echo "  2. Activate the environment:  source .venv/bin/activate"
echo "  3. Index your library:        python index.py"
echo "  4. Start the web app:         python webapp.py"
echo "  5. Open http://localhost:5001 in your browser"
echo ""
