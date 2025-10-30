#!/usr/bin/env python3
import os, time, random
from vector_utils import Embedder, VectorStore


MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small") if os.getenv("EMBED_PROVIDER","openai")=="openai" else os.getenv("SBERT_MODEL","all-MiniLM-L6-v2")
DIM = int(os.getenv("PGVECTOR_DIM", "1536" if os.getenv("EMBED_PROVIDER","openai")=="openai" else "384"))


QUERIES = [
"iphone 13 case",
"wireless bluetooth earbuds",
"gaming laptop 16gb ram",
]


embedder = Embedder(model=MODEL, dim=DIM)
store = VectorStore(embedder=embedder)


for q in QUERIES:
  t0 = time.perf_counter()
  res = store.search(q, k=10)
  t_ms = (time.perf_counter()-t0)*1000
  print(f"{q:30s} -> {len(res)} hits in {t_ms:.1f} ms")