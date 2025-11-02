# vectors/build_faiss_from_csv.py
import os
import json
import math
import re
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
    parts = []
    title = str(row.get("title") or "").strip()
    if title: parts.append(title)

    seller = str(row.get("seller") or "").strip()
    if seller: parts.append(f"seller {seller}")

    source = _norm_source(row.get("source"))
    parts.append(f"source {source}")

    total = _clean_price(row.get("total"))
    currency = _norm_currency(row.get("currency"))
    if total is not None:
        parts.append(f"price {total} {currency}")

    if row.get("query"):
        parts.append(f"for query {row.get('query')}")

    return " | ".join(parts)[:TEXT_COL_MAXLEN]

def _is_nan(x):
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False

def _norm_source(s):
    if s is None or _is_nan(s):
        return "unknown"
    s = str(s).strip().lower()
    if "ebay" in s: return "ebay"
    if "keepa" in s: return "amazon"   # keepa rows are amazon products
    if "serp"  in s: return "google"
    return s or "unknown"

def _norm_currency(cur):
    if cur is None or _is_nan(cur):
        return "USD"
    s = str(cur).strip().upper()
    if s in ("", "NAN", "NULL", "NONE"): return "USD"
    # common symbol mapping
    if s in ("$", "USD"): return "USD"
    if s in ("€", "EUR"): return "EUR"
    if s in ("£", "GBP"): return "GBP"
    return s

def _clean_price(x):
    if x is None or _is_nan(x):
        return None
    # handle strings like "NaN", "$1,234.56", "  12.34 "
    xs = str(x).strip()
    if xs.upper() in ("", "NAN", "NULL", "NONE"): return None
    # strip leading currency symbols and commas
    if xs and not xs[0].isdigit() and xs[0] not in "+-":
        xs = xs[1:]
    xs = xs.replace(",", "")
    try:
        return float(xs)
    except Exception:
        return None

def _norm_link(link, asin=None):
    if link is None or _is_nan(link):
        return f"https://www.amazon.com/dp/{asin}" if asin else None
    s = str(link).strip()
    if s.lower() == "nan" or s == "":
        return f"https://www.amazon.com/dp/{asin}" if asin else None
    return s

def _slug(s):
    s = "" if s is None or _is_nan(s) else str(s).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")

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
    # Normalize/clean fields safely
    if "source" in pr.columns:
        pr["source"] = pr["source"].apply(_norm_source)
    else:
        pr["source"] = "unknown"

    if "currency" in pr.columns:
        pr["currency"] = pr["currency"].apply(_norm_currency)
    else:
        pr["currency"] = "USD"

    for col in ("total", "price", "shipping"):
        if col in pr.columns:
            pr[col] = pr[col].apply(_clean_price)

    # Links
    if "link" in pr.columns:
        pr["link"] = pr["link"].apply(lambda v: None if v is None or _is_nan(v) or str(v).strip().lower()=="nan" else str(v).strip())
    else:
        pr["link"] = None

    # Build a duplicate key and keep the cheapest per key
    dup_keys = []
    for _, r in pr.iterrows():
        if r.get("link"):
            dup_keys.append(("link", r["link"]))
        else:
            dup_keys.append(("sig", _slug(r.get("title")), _slug(r.get("seller")), r.get("source"), r.get("total")))
    pr["__dupkey"] = dup_keys

    pr.sort_values(by=["total"], ascending=[True], inplace=True, na_position="last")
    pr = pr.drop_duplicates(subset=["__dupkey"], keep="first").drop(columns=["__dupkey"]).reset_index(drop=True)

    # pr = pr[[c for c in keep_cols if c in pr.columns]].copy()
    # pr = pr[pr["title"].notna() & (pr["title"].astype(str).str.strip() != "")]
    # pr = pr.drop_duplicates(subset=["id"]).reset_index(drop=True)

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
