import os, math, tempfile, json
import numpy as np
import pytest


# Force FAISS backend locally for tests
os.environ.setdefault("VECTOR_BACKEND", "faiss")
os.environ.setdefault("EMBED_PROVIDER", "sbert")
os.environ.setdefault("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("PGVECTOR_DIM", "384")


# Optionally monkeypatch the embedder to avoid external calls
class DummyEmbedder:
  def __init__(self, dim=8):
    self.model = "dummy"
    self.dim = dim
  def embed(self, texts):
  # deterministic toy embeddings: hash → vector on unit sphere
    out = []
    for t in texts:
    rng = np.random.default_rng(abs(hash(t)) % (2**32))
    v = rng.normal(size=self.dim).astype("float32")
    v /= np.linalg.norm(v) + 1e-9
    out.append(v.tolist())
    return out


@pytest.fixture
def vector_store_tmp(monkeypatch, tmp_path):
  from vector_utils import VectorStore
  dummy = DummyEmbedder(dim=32)
  monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path/"idx.bin"))
  monkeypatch.setenv("FAISS_IDS_PATH", str(tmp_path/"ids.json"))
  vs = VectorStore(embedder=dummy)
  return vs


def test_upsert_and_search_similarity(vector_store_tmp):
  vs = vector_store_tmp
  items = [
  ("p1", vs.embedder.embed(["apple iphone 13 smartphone"])[0]),
  ("p2", vs.embedder.embed(["iphone 13 by apple phone 128gb"])[0]),
  ("p3", vs.embedder.embed(["garden hose 50ft"])[0]),
  ]
  vs.upsert_embeddings(items)


  res = vs.search("iphone 13", k=2)
  assert res, "should return neighbors"
  ids = [r[0] for r in res]
  assert "p1" in ids and "p2" in ids, "similar items should rank at top"




def test_dimensionality_matches(monkeypatch):
  from vector_utils import Embedder
  dummy = DummyEmbedder(dim=64)
  assert dummy.dim == 64


@pytest.mark.slow
def test_latency_under_100ms(vector_store_tmp):
  vs = vector_store_tmp
  # Populate ~1000 random items
  vecs = []
  for i in range(1000):
  pid = f"pid-{i}"
  emb = vs.embedder.embed([f"item {i}"])[0]
  vecs.append((pid, emb))
  vs.upsert_embeddings(vecs)


  import time
  t0 = time.perf_counter()
  _ = vs.search("item 42", k=10)
  dt = (time.perf_counter() - t0) * 1000
  assert dt <= 100.0, f"expected ≤100 ms, got {dt:.1f} ms"