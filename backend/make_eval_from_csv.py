#!/usr/bin/env python3
import os, json, itertools, csv
import pandas as pd

SRC_RESULTS = os.getenv("SRC_RESULTS", "backend/local_db/data/exports/20251101T155327Z/product_results.csv")
SRC_SEARCH  = os.getenv("SRC_SEARCH",  "backend/local_db/data/exports/20251101T155327Z/search_history.csv")
OUT_PROD    = os.getenv("OUT_PROD",   "backend/local_db/data/eval/products_eval.csv")
OUT_PAIRS   = os.getenv("OUT_PAIRS",  "backend/local_db/data/eval/labeled_pairs.csv")

def safe_get(d, path, default=None):
    cur = d
    for p in path:
        if cur is None: return default
        cur = cur.get(p) if isinstance(cur, dict) else default
    return cur if cur is not None else default

def parse_raw(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

def main():
    os.makedirs(os.path.dirname(OUT_PROD), exist_ok=True)

    df = pd.read_csv(SRC_RESULTS)
    # Build listing_id as a stable key: prefer the DB pk if present; else combine search_id + row index
    if "id" in df.columns:
        df["listing_id"] = df["id"].astype(str)
    else:
        df["listing_id"] = df.reset_index().apply(lambda r: f"{r['search_id']}:{r['index']}", axis=1)

    # Parse raw json to extract canonical fields
    rawj = df["raw"].astype(str).apply(parse_raw)
    df["product_id"]   = rawj.apply(lambda r: safe_get(r, ["product_id"], None))
    df["brand_guess"]  = rawj.apply(lambda r: safe_get(r, ["brand_guess"], None))
    df["title2"]       = rawj.apply(lambda r: safe_get(r, ["title"], None)).fillna(df["title"])
    df["total2"]       = rawj.apply(lambda r: safe_get(r, ["total_cost"], None))
    df["total"]        = df["total"].fillna(df["total2"])
    df["title"]        = df["title2"].fillna(df["title"])

    # Keep only rows with a product_id (we need it to build gold labels)
    df = df[~df["product_id"].isna()].copy()
    df["product_id"] = df["product_id"].astype(str)

    # ---- Write products_eval.csv for the clustering pipeline ----
    # Map to the fields your ProductRecord expects
    out_products = df[[
        "listing_id","source","title","brand_guess","product_id","total"
    ]].rename(columns={
        "source":"vendor",
        "brand_guess":"brand",
        "product_id":"upc",     # repurpose 'upc' slot as an item-level hard key
        "total":"price",
    }).copy()

    out_products.to_csv(OUT_PROD, index=False)

    # ---- Build labeled duplicate pairs by (search_id, product_id) ----
    pairs = []
    pair_id = 1
    for pid, g in df.groupby("product_id"):
        lids = sorted(g["listing_id"].astype(str).tolist())
        if len(lids) < 2:
            continue
        for a_idx in range(len(lids)):
            for b_idx in range(a_idx+1, len(lids)):
                a, b = lids[a_idx], lids[b_idx]
                pairs.append((pair_id, a, b, 1))  # duplicate
                pair_id += 1

    # Optional: add some **non-duplicate** pairs for balance (sample within same search_id but different product_id)
    # Keep it modest to avoid exploding the dataset.
    # For each search, take a small sample of cross-product_id pairs:
    # Negatives: sample across different product_id globally
    neg = []
    # Small deterministic pool
    pool = df[["listing_id","product_id"]].sample(n=min(300, len(df)), random_state=13)
    pool = pool.reset_index(drop=True).astype(str)

    for i in range(min(len(pool), 80)):
        for j in range(i+1, min(len(pool), 80)):
            if pool.loc[i, "product_id"] != pool.loc[j, "product_id"]:
                neg.append((pair_id, pool.loc[i,"listing_id"], pool.loc[j,"listing_id"], 0))
                pair_id += 1
            if len(neg) >= 200:
                break
        if len(neg) >= 200:
            break

    with open(OUT_PAIRS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id","id_a","id_b","is_duplicate"])
        for row in pairs + neg:
            w.writerow(row)

    print(f"Wrote {OUT_PROD} ({len(out_products)} rows) and {OUT_PAIRS} ({len(pairs)+len(neg)} pairs)")

if __name__ == "__main__":
    main()
