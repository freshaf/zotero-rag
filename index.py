#!/usr/bin/env python3
"""Run the indexing pipeline.

Usage:
    python index.py          # Full re-index
    python index.py --update # Incremental update (new/changed items only)
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from src.indexer import run_full_index, run_incremental_update

if __name__ == '__main__':
    if '--update' in sys.argv:
        run_incremental_update()
    else:
        run_full_index()
