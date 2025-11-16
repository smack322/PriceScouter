# tools/query_faiss_table.py
import argparse, json, re
import pandas as pd
import pyarrow.parquet as pq

DEF_META = "backend/local_db/data/faiss/meta.parquet"

def like(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.contains(pattern, flags=re.I, regex=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=DEF_META, help="Path to meta.parquet")
    ap.add_argument("--source", help="exact source match (amazon|ebay|google|unknown)")
    ap.add_argument("--seller-like", help="regex / case-insensitive contains for seller")
    ap.add_argument("--title-like", help="regex / case-insensitive contains for title")
    ap.add_argument("--price-min", type=float)
    ap.add_argument("--price-max", type=float)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", action="store_true", help="print JSON rows")
    args = ap.parse_args()

    df = pq.read_table(args.meta).to_pandas()
    # meta.parquet has columns: id, text, metadata_json
    meta = pd.json_normalize(df["metadata_json"].apply(json.loads))
    # join back any columns you want from df; here we keep text too
    meta["text"] = df["text"]

    out = meta
    if args.source:
        out = out[out["source"] == args.source]
    if args.seller-like:
        out = out[like(out["seller"], args.seller_like)]
    if args.title_like:
        out = out[like(out["title"], args.title_like)]
    if args.price_min is not None:
        out = out[(out["total"].fillna(1e18) >= args.price_min)]
    if args.price_max is not None:
        out = out[(out["total"].fillna(-1e18) <= args.price_max)]

    out = out.sort_values(by=["total", "created_at"], na_position="last").head(args.limit)

    if args.json:
        print(out.to_json(orient="records"))
    else:
        # pretty print a few useful cols
        cols = ["product_id","title","seller","source","total","currency","link","query","created_at"]
        cols = [c for c in cols if c in out.columns]
        print(out[cols].to_string(index=False))

if __name__ == "__main__":
    main()
