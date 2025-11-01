import os
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import faiss

DEF_OUT_DIR = "backend/local_db/data/faiss"
DEF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_index(out_dir: str):
    index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
    idmap = np.load(os.path.join(out_dir, "idmap.npy"))
    meta = pq.read_table(os.path.join(out_dir, "meta.parquet")).to_pandas()
    meta = meta.set_index("id")  # id → (text, metadata_json)
    return index, idmap, meta

def search(query: str, k: int, index, model, idmap, meta_df: pd.DataFrame):
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(qv, k)
    idxs = idxs[0]
    scores = scores[0]
    results = []
    for rank, (pi, score) in enumerate(zip(idxs, scores), start=1):
        if pi < 0:
            continue
        prod_id = int(idmap[pi])
        row = meta_df.loc[prod_id]
        meta = json.loads(row["metadata_json"])
        results.append({
            "rank": rank,
            "score": float(score),           # cosine similarity in [-1, 1]
            "product_id": prod_id,
            "text": row["text"],
            "title": meta.get("title"),
            "source": meta.get("source"),
            "seller": meta.get("seller"),
            "total": meta.get("total"),
            "currency": meta.get("currency"),
            "link": meta.get("link"),
            "query": meta.get("query"),
        })
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DEF_OUT_DIR)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("query", nargs="+")
    args = ap.parse_args()

    index, idmap, meta = load_index(args.out_dir)
    model = SentenceTransformer(args.model)

    q = " ".join(args.query)
    hits = search(q, args.k, index, model, idmap, meta)
    for h in hits:
        print(f"[{h['rank']:>2}] score={h['score']:.3f}  id={h['product_id']}  title={h['title']}")
        print(f"     text: {h['text']}")
        print(f"     src: {h['source']}  seller: {h['seller']}  total: {h['total']} {h['currency']}")
        print(f"     link: {h['link']}")
        if h.get("query"):
            print(f"     orig_query: {h['query']}")
        print()

if __name__ == "__main__":
    main()