# backend/local_db/vector_db/index_sync.py

import os
import sys
import sqlite3
from pathlib import Path
from typing import Tuple

import pandas as pd

# --- add project root to sys.path so "backend" package can be imported ---
ROOT = Path(__file__).resolve().parents[3]  # PriceScouter/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Adjust these if your paths/constants are different
DB_PATH = "app.db"
FAISS_DIR = Path("backend/local_db/Data/faiss")
INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "meta.parquet"

# Import the builder you already wrote
from backend.local_db.vector_db.build_faiss_from_sqlite import (
    build_faiss_from_canonical_view,
)


def get_canonical_row_count() -> int:
    """How many canonical rows are in the SQLite view?"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM canonical_product_view WHERE canonical_key IS NOT NULL AND title IS NOT NULL;")
    (count,) = cur.fetchone()
    conn.close()
    return int(count)


def get_index_row_count() -> int:
    """How many rows are in the FAISS meta parquet (one per vector)?"""
    if not META_PATH.exists():
        return 0
    df = pd.read_parquet(META_PATH)
    return len(df)


def sync_canonical_index(force: bool = False) -> Tuple[int, int]:
    """
    Ensure FAISS canonical index matches the current contents of canonical_product_view.

    Returns (source_count, index_count) after any rebuild.
    """
    source_count = get_canonical_row_count()
    index_count = get_index_row_count()

    needs_rebuild = force or (not INDEX_PATH.exists()) or (source_count != index_count)

    if needs_rebuild:
        print(
            f"[index_sync] Rebuilding canonical FAISS index: "
            f"source_count={source_count}, index_count={index_count}"
        )
        build_faiss_from_canonical_view(limit=None)
        # recompute after rebuild
        source_count = get_canonical_row_count()
        index_count = get_index_row_count()
        print(
            f"[index_sync] Rebuild complete: source_count={source_count}, index_count={index_count}"
        )
    else:
        print(
            f"[index_sync] Canonical FAISS index already in sync: "
            f"source_count={source_count}, index_count={index_count}"
        )

    return source_count, index_count


if __name__ == "__main__":
    sync_canonical_index(force=False)
