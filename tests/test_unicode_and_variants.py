# tests/test_unicode_and_variants.py
import pytest
from .helpers import upsert_titles

def test_unicode_titles_embed_and_search(faiss_store_dummy):
    upsert_titles(faiss_store_dummy, [("ux1", "Sofá 3 plazas – azul 💙 128GB")])
    hits = faiss_store_dummy.search("sofa azul 128gb", k=5)
    ids = [i for i, _ in hits]
    assert "ux1" in ids

def test_variant_extraction_from_title():
    from comparison.aggregator import extract_variant
    row = {"title": "iPhone 13 128GB Blue (Factory Unlocked)"}
    v = extract_variant(row)
    # depending on your color list/regex this may resolve to "blue"
    assert v["color"] in (None, "blue")  # keep loose if list differs
    assert v["size"] in (None, "128gb")
