# backend/local_db/vector_db/query_canonical_faiss.py

import argparse
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

FAISS_DIR = Path("backend/local_db/Data/faiss")
INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "meta.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # should match build_faiss_from_sqlite


_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_meta_df: pd.DataFrame | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_index() -> faiss.Index:
    global _index
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    return _index


def get_meta() -> pd.DataFrame:
    global _meta_df
    if _meta_df is None:
        _meta_df = pd.read_parquet(META_PATH)
    return _meta_df


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    emb = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # must match build step
    )
    return emb.astype("float32")


def search_canonical_products(query: str, k: int = 10) -> List[Dict[str, Any]]:
    index = get_index()
    meta = get_meta()

    q_vec = embed_query(query)
    scores, ids = index.search(q_vec, k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = meta.iloc[int(idx)]

        results.append(
            {
                "score": float(score),
                "canonical_id": int(row["canonical_id"]),
                "canonical_key": row["canonical_key"],
                "title": row["title"],
                "avg_price": float(row["avg_price"]) if row["avg_price"] is not None else None,
                "min_price": float(row["min_price"]) if row["min_price"] is not None else None,
                "max_price": float(row["max_price"]) if row["max_price"] is not None else None,
                "seller_count": int(row["seller_count"]) if row["seller_count"] is not None else None,
                "total_listings": int(row["total_listings"]) if row["total_listings"] is not None else None,
                "representative_url": row["representative_url"],
            }
        )

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", "-q", required=True, help="Free-text query")
    ap.add_argument("--k", type=int, default=10, help="Number of results")
    args = ap.parse_args()

    results = search_canonical_products(args.query, k=args.k)

    print(f"Top {len(results)} canonical results for: {args.query!r}\n")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] score={r['score']:.3f}  canonical_id={r['canonical_id']}")
        print(f"    title: {r['title']}")
        print(f"    avg_price: {r['avg_price']}  min: {r['min_price']}  max: {r['max_price']}")
        print(f"    sellers: {r['seller_count']}  listings: {r['total_listings']}")
        print(f"    url: {r['representative_url']}")
        print()

if __name__ == "__main__":
    main()
