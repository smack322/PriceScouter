#!/usr/bin/env python3
import os
import time
from vector_utils import Embedder, VectorStore

# Provider: "openai" or "sbert"
PROVIDER = os.getenv("EMBED_PROVIDER", "openai").strip().lower()

# Model + dims by provider (defaults are safe)
if PROVIDER == "openai":
    MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    DIM = int(os.getenv("PGVECTOR_DIM", "1536"))
else:
    MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    DIM = int(os.getenv("PGVECTOR_DIM", "384"))

# Where your FAISS index + sidecars live

OUT_DIR = os.getenv("FAISS_DIR", "/Users/scottmcanally/Desktop/PriceScouter/backend/local_db/data/faiss")

QUERIES = [
    "iphone 13 case",
    "wireless bluetooth earbuds",
    "gaming laptop 16gb ram",
    "sauna",
]

def main():
    embedder = Embedder(provider=PROVIDER, model=MODEL, dim=DIM)
    store = VectorStore(embedder=embedder, out_dir=OUT_DIR)

    print(f"Provider={PROVIDER}  Model={MODEL}  Dim={DIM}")
    for q in QUERIES:
        t0 = time.perf_counter()
        hits = store.search(q, k=10)
        t_ms = (time.perf_counter() - t0) * 1000
        print(f"{q:30s} -> {len(hits):2d} hits in {t_ms:7.1f} ms")
        # Optional: show top-3
        for h in hits[:3]:
            print(f"   score={h['score']:.3f}  src={h.get('source')}  title={h.get('title')}")
        print()

if __name__ == "__main__":
    main()
