# tests/test_vector_utils.py
import os
import pytest
import numpy as np
# from tests.helpers import upsert_titles, seed_n, timed_ms

# from conftest import upsert_titles, seed_n, timed_ms

def test_embedding_dimensionality(dummy_embedder_384):
    vecs = dummy_embedder_384.embed(["foo"])
    assert len(vecs[0]) == 384

def test_similarity_ranking(faiss_store_dummy):
    items = [
        ("p1", "apple iphone 13 128gb"),
        ("p2", "iphone 13 by apple"),
        ("p3", "garden hose 50ft"),
    ]
    upsert_titles(faiss_store_dummy, items)
    hits = faiss_store_dummy.search("iphone 13", k=2)
    ids = [i for i, _ in hits]
    assert {"p1", "p2"}.issubset(set(ids))

def test_upsert_idempotency(faiss_store_dummy):
    upsert_titles(faiss_store_dummy, [("pX", "test title A")])
    first = faiss_store_dummy.search("test title A", k=1)
    upsert_titles(faiss_store_dummy, [("pX", "test title B")])  # same id, new vec
    second = faiss_store_dummy.search("test title B", k=1)
    assert first and second  # both calls succeed
    # same internal id should map to pX still
    assert second[0][0] == "pX"

def test_search_empty_title_returns_empty(faiss_store_dummy):
    assert faiss_store_dummy.search("", k=10) == []

@pytest.mark.performance
def test_ann_latency_under_100ms(faiss_store_dummy):
    seed_n(faiss_store_dummy, n=1000)
    # warmup
    faiss_store_dummy.search("item 42", k=10)
    ms = timed_ms(lambda: faiss_store_dummy.search("item 42", k=10))
    assert ms <= 100.0, f"Expected ≤100 ms, got {ms:.1f} ms"
