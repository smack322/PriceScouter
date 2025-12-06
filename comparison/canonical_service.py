import sys
from pathlib import Path

# --- Ensure project root is on sys.path ---
ROOT = Path(__file__).resolve().parents[1]  # PriceScouter/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.local_db.vector_db.index_sync import sync_canonical_index
from backend.local_db.vector_db.query_canonical_faiss import search_canonical_products

def search_canonical_with_sync(query: str, k: int = 10):
    # cheap no-op most of the time, but guarantees alignment
    sync_canonical_index(force=False)
    return search_canonical_products(query, k=k)