# vectors/build_faiss_from_csv.py
import os
import json
import math
import argparse
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from typing import List
from sentence_transformers import SentenceTransformer
import faiss

# ---------- Config ----------
DEFAULT_EXPORT_DIR = "backend/local_db/data/exports"   # parent folder with timestamp subfolders
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast & solid
OUT_DIR = "backend/local_db/data/faiss"                # where to store index + metadata
TEXT_COL_MAXLEN = 512                                  # clip overly long texts

def pick_latest_export(export_root: str) -> str:
    # pick most recent timestamped export folder
    abs_root = os.path.abspath(export_root)
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(f"Export root not found: {abs_root}")
    candidates = [os.path.join(abs_root, d) for d in os.listdir(abs_root) if os.path.isdir(os.path.join(abs_root, d))]
    if not candidates:
        raise FileNotFoundError(f"No timestamped export folders under {abs_root}")
    return sorted(candidates)[-1]

def build_text(row) -> str:
    """
    High-signal text for embedding.
    Uses product title + a bit of context (seller/source/price).
    """
    parts: List[str] = []
    title = str(row.get("title") or "").strip()
    if title:
        parts.append(title)
    seller = str(row.get("seller") or "").strip()
    if seller:
        parts.append(f"Seller: {seller}")
    source = str(row.get("source") or "").strip()
    if source:
        parts.append(f"Source: {source}")
    total = row.get("total")
    currency = str(row.get("currency") or "").strip()
    if pd.notna(total):
        parts.append(f"Total: {total} {currency}".strip())
    text = " | ".join(parts)
    return text[:TEXT_COL_MAXLEN]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR, help="Root exports dir containing timestamped folders")
    ap.add_argument("--export-ts", default=None, help="Specific timestamp folder name under export-dir (optional)")
    ap.add_argument("--limit", type=int, default=None, help="Limit rows for quick tests")
    ap.add_argument("--model", default=MODEL_NAME, help="Sentence-Transformers model name")
    ap.add_argument("--out-dir", default=OUT_DIR, help="Where to write index.faiss + meta")
    args = ap.parse_args()

    # Locate CSVs
    ts_dir = os.path.join(args.export_dir, args.export_ts) if args.export_ts else pick_latest_export(args.export_dir)
    pr_csv = os.path.join(ts_dir, "product_results.csv")
    sh_csv = os.path.join(ts_dir, "search_history.csv")
    if not os.path.exists(pr_csv):
        raise FileNotFoundError(f"Missing {pr_csv}")
    if not os.path.exists(sh_csv):
        raise FileNotFoundError(f"Missing {sh_csv}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load CSVs
    pr = pd.read_csv(pr_csv)
    sh = pd.read_csv(sh_csv)

    # Optional: join original query text into metadata
    # product_results has "search_id" FK to search_history.id
    if "search_id" in pr.columns and "id" in sh.columns:
        sh_min = sh[["id", "query"]].rename(columns={"id": "search_id"})
        pr = pr.merge(sh_min, on="search_id", how="left")
    else:
        pr["query"] = None

    # Keep useful columns; drop rows with empty titles
    keep_cols = [
        "id", "title", "source", "seller", "price", "shipping", "total", "currency",
        "link", "created_at", "search_id", "query"
    ]
    pr = pr[[c for c in keep_cols if c in pr.columns]].copy()
    pr = pr[pr["title"].notna() & (pr["title"].astype(str).str.strip() != "")]
    pr = pr.drop_duplicates(subset=["id"]).reset_index(drop=True)

    if args.limit is not None:
        pr = pr.head(args.limit)

    # Build text + metadata
    pr["text"] = pr.apply(build_text, axis=1)

    def to_meta(row):
        meta = {
            "product_id": int(row["id"]),
            "query": row.get("query"),
            "source": row.get("source"),
            "seller": row.get("seller"),
            "price": row.get("price"),
            "shipping": row.get("shipping"),
            "total": row.get("total"),
            "currency": row.get("currency"),
            "link": row.get("link"),
            "created_at": row.get("created_at"),
            "search_id": row.get("search_id"),
            "title": row.get("title"),
        }
        return meta

    pr["metadata_json"] = pr.apply(lambda r: json.dumps(to_meta(r), ensure_ascii=False), axis=1)

    # Embed
    model = SentenceTransformer(args.model)
    texts = pr["text"].tolist()
    emb = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    # emb shape: (N, D); already L2-normalized for cosine via inner product.

    # Build FAISS index (cosine via inner product on normalized vectors)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # Save index + sidecars
    index_path = os.path.join(args.out_dir, "index.faiss")
    faiss.write_index(index, index_path)

    # Save id mapping and metadata (parquet is handy)
    idmap = pr["id"].astype(np.int64).to_numpy()
    np.save(os.path.join(args.out_dir, "idmap.npy"), idmap)

    # Save metadata+text as parquet for fast reload
    meta_tbl = pa.Table.from_pandas(pr[["id", "text", "metadata_json"]], preserve_index=False)
    pq.write_table(meta_tbl, os.path.join(args.out_dir, "meta.parquet"))

    print(f"Built FAISS index with {len(pr)} vectors (dim={dim}).")
    print("Saved:")
    print(" -", index_path)
    print(" -", os.path.join(args.out_dir, "idmap.npy"))
    print(" -", os.path.join(args.out_dir, "meta.parquet"))

if __name__ == "__main__":
    main()
