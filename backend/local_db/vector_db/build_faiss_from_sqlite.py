# backend/local_db/vector_db/build_faiss_from_sqlite.py

import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # <-- NEW

DB_PATH = "app.db"
FAISS_DIR = Path("backend/local_db/Data/faiss")
INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "meta.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 is 384-dim


# cache the model so we don't re-load it every call
_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Use the same embedding model style as build_faiss_from_csv.py,
    but without importing embed_batch from another module.
    """
    model = get_model()
    emb = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalized for cosine/IP
    )
    return emb


def build_text(row: pd.Series) -> str:
    """
    Build a canonical-level description string for embedding.
    """
    parts: list[str] = []

    title = str(row.get("title") or "").strip()
    if title:
        parts.append(title)

    seller_count = row.get("seller_count")
    if seller_count is not None:
        parts.append(f"sellers {int(seller_count)}")

    avg_price = row.get("avg_price")
    if avg_price is not None:
        parts.append(f"avg_price {avg_price:.2f} USD")

    min_price = row.get("min_price")
    max_price = row.get("max_price")
    if min_price is not None and max_price is not None:
        parts.append(f"range {min_price:.2f}-{max_price:.2f} USD")

    total_listings = row.get("total_listings")
    if total_listings is not None:
        parts.append(f"listings {int(total_listings)}")

    ckey = row.get("canonical_key")
    if ckey:
        parts.append(f"key {str(ckey)}")

    text = " | ".join(parts)
    return text[:512]


def build_faiss_from_canonical_view(limit: int | None = None) -> None:
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT
            canonical_id,
            canonical_key,
            title,
            min_price,
            avg_price,
            max_price,
            seller_count,
            total_listings,
            representative_url
        FROM canonical_product_view
        WHERE canonical_key IS NOT NULL
          AND title IS NOT NULL
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        raise ValueError("No rows in canonical_product_view; nothing to index.")

    # build text
    df["text"] = df.apply(build_text, axis=1)
    texts = df["text"].tolist()

    # embed
    vectors = embed_texts(texts)
    if vectors.shape[0] != len(df):
        raise RuntimeError(f"Embedding count mismatch: {vectors.shape[0]} vs {len(df)} rows")

    # FAISS index (cosine via IP)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))

    faiss.write_index(index, str(INDEX_PATH))

    # meta parquet, idmap aligned with FAISS ids
    df = df.reset_index(drop=True)
    df["faiss_id"] = df.index

    df.to_parquet(META_PATH, index=False)
    print(f"Built canonical FAISS index with {index.ntotal} vectors (dim={vectors.shape[1]}).")
    print("Saved index:", INDEX_PATH)
    print("Saved meta:", META_PATH)


if __name__ == "__main__":
    build_faiss_from_canonical_view(limit=None)
