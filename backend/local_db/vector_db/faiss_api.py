# api_simple_faiss.py
from fastapi import FastAPI, Query
from typing import List, Any, Dict
import os, json
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import faiss

FAISS_DIR = "backend/local_db/data/faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI()
_index = _idmap = _meta = _model = None

def _load_faiss():
    global _index, _idmap, _meta, _model
    if _index is None:
        _index = faiss.read_index(os.path.join(FAISS_DIR, "index.faiss"))
        _idmap = np.load(os.path.join(FAISS_DIR, "idmap.npy"))
        _meta = pq.read_table(os.path.join(FAISS_DIR, "meta.parquet")).to_pandas().set_index("id")
        _model = SentenceTransformer(MODEL_NAME)

def _faiss_search(q: str, k: int) -> List[Dict[str, Any]]:
    v = _model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = _index.search(v, k)
    out = []
    for rank, (pi, sc) in enumerate(zip(idxs[0], scores[0]), start=1):
        if pi < 0: 
            continue
        pid = int(_idmap[pi])
        row = _meta.loc[pid]
        meta = json.loads(row["metadata_json"])
        out.append({
            "rank": rank,
            "score": float(sc),
            "product_id": pid,
            "title": meta.get("title"),
            "seller": meta.get("seller"),
            "source": meta.get("source"),
            "total": meta.get("total"),
            "currency": meta.get("currency"),
            "link": meta.get("link"),
            "query": meta.get("query"),
            "text": row["text"],
        })
    return out

@app.get("/ui/search_simple")
def search_simple(q: str = Query(...), k: int = Query(10, ge=1, le=50)):
    _load_faiss()
    return {"items": _faiss_search(q, k)}
