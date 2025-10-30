#!/usr/bin/env python3
import os, sys, csv
import pandas as pd
from vector_utils import Embedder, VectorStore


MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small") if os.getenv("EMBED_PROVIDER","openai")=="openai" else os.getenv("SBERT_MODEL","all-MiniLM-L6-v2")
DIM = int(os.getenv("PGVECTOR_DIM", "1536" if os.getenv("EMBED_PROVIDER","openai")=="openai" else "384"))


# Example source: a CSV with columns [product_id, title]
SRC = os.getenv("EMBED_SOURCE_CSV", "data/products_titles.csv")
BATCH = int(os.getenv("EMBED_BATCH", "128"))




def main():
  embedder = Embedder(model=MODEL, dim=DIM)
  store = VectorStore(embedder=embedder)


  df = pd.read_csv(SRC)
  rows = list(zip(df["product_id"].astype(str).tolist(), df["title"].astype(str).tolist()))


  for i in range(0, len(rows), BATCH):
    chunk = rows[i:i+BATCH]
    ids, texts = zip(*chunk)
    vecs = embedder.embed(list(texts))
    store.upsert_embeddings(list(zip(ids, vecs)))
    print(f"Upserted {i+len(chunk)}/{len(rows)}")


if __name__ == "__main__":
  main()