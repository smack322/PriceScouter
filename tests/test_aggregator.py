# tests/test_aggregator.py
import math
import numpy as np
import pytest

# Adjust this import if your path differs:
from comparison.clustering import (
    ProductRecord,
    cluster_products,
    stable_uuid,
    blocking_key,
    normalize_brand,
    embed_texts,
)

def _rec(id, title, price, brand="Apple", vendor="s1", upc=None, attrs=None):
    return ProductRecord(
        listing_id=id,
        vendor=vendor,
        title=title,
        brand=brand,
        upc=upc,
        attrs=attrs or {},
        price=price,
    )

def _extract_by_cid(outputs, memberships):
    """helper -> {cid: (canon_row, [membership_rows])}"""
    by = {}
    for o in outputs:
        by[o["canonical_product_id"]] = [o, []]
    for m in memberships:
        cid = m["canonical_product_id"]
        if cid in by:
            by[cid][1].append(m)
    return by

# ------------------------- Unit-ish checks on pipeline outputs -------------------------

def test_stats_and_medoid_for_simple_cluster():
    # Keep tokens identical across titles so they land in the same block
    records = [
        _rec("A", "Apple iPhone 13 128GB Black", 599.0, brand="Apple", vendor="s1"),
        _rec("B", "Apple iPhone 13 128GB Black", 579.0, brand="Apple", vendor="s2"),
        _rec("C", "Apple iPhone 13 128GB Black", 589.0, brand="Apple", vendor="s1"),
    ]
    outputs, memberships = cluster_products(records, theta=0.60)

    # Expect ONE canonical cluster of size 3
    assert len(outputs) == 1
    canon = outputs[0]
    assert math.isclose(canon["price_min"], 579.0, rel_tol=1e-6)
    assert math.isclose(canon["price_max"], 599.0, rel_tol=1e-6)
    assert math.isclose(canon["price_avg"], (599.0 + 579.0 + 589.0) / 3.0, rel_tol=1e-6)
    assert canon["listing_count"] == 3
    assert canon["title"] in {r.title for r in records}

    # All memberships should reference the same canonical id and have a bounded similarity
    cid = canon["canonical_product_id"]
    sims = [m["similarity_to_centroid"] for m in memberships if m["canonical_product_id"] == cid]
    assert len(sims) == 3
    for s in sims:
        assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


def test_brand_gate_splits_clusters_when_brands_differ():
    # Two Samsung listings share tokens (same block), third is brand Apple (brand gate splits)
    records = [
        _rec("A", "Samsung Galaxy S21 128GB Black", 499.0, brand="Samsung"),
        _rec("B", "Samsung Galaxy S21 Black 128GB", 509.0, brand="Samsung"),
        _rec("C", "Samsung Galaxy S21 128GB Black", 505.0, brand="Apple"),  # title same but brand different
    ]
    outputs, _ = cluster_products(records, theta=0.60)

    # Expect two clusters: Samsung pair (size 2) and Apple singleton (size 1)
    sizes = sorted(o["listing_count"] for o in outputs)
    assert sizes == [1, 2]

def test_upc_groups_split_by_brand_purity():
    # Same UPC but mixed brands → should split per brand purity path
    recs = [
        _rec("A", "Widget 1", 10.0, brand="Acme",   upc="000111222333"),
        _rec("B", "Widget 1", 11.0, brand="Acme",   upc="000111222333"),
        _rec("C", "Widget 1", 12.0, brand="Globex", upc="000111222333"),
    ]
    outputs, _ = cluster_products(recs, theta=0.85)
    # Two clusters: Acme(2) + Globex(1)
    sizes = sorted(o["listing_count"] for o in outputs)
    assert sizes == [1, 2]

def test_blocking_key_is_deterministic_and_brand_sensitive():
    r1 = _rec("A", "The Apple iPhone 13 with 128GB", 599.0, brand="Apple")
    r2 = _rec("B", "Apple iPhone 13 128GB", 599.0, brand="Apple")
    k1 = blocking_key(r1)
    k2 = blocking_key(r2)
    assert k1 == k2  # same normalized/stopword-filtered token key
    # Changing brand affects the key
    r3 = _rec("C", "Apple iPhone 13 128GB", 599.0, brand="NotApple")
    k3 = blocking_key(r3)
    assert k3 != k2

def test_stable_uuid_is_deterministic_for_same_inputs():
    brand = "Apple"
    upc = "190199247748"
    # emulate key_tokens from blocking_key
    kt = tuple(sorted("iphone 13 128gb".split()))
    cid1 = stable_uuid(brand, upc, kt)
    cid2 = stable_uuid(brand, upc, kt)
    assert cid1 == cid2

def test_embed_texts_normalizes_and_nonzero_vectors():
    X = embed_texts(["Apple iPhone 13", "iPhone 13 Apple", "Garden Hose 50ft"])
    # Vectors are L2-normalized; norms ≈ 1 (except all-zero which shouldn’t happen with tokens)
    norms = np.linalg.norm(X, axis=1)
    for n in norms:
        assert 0.99 <= n <= 1.01
    # First two sentences share same bag-of-words → identical vectors in this toy embedder
    assert np.allclose(X[0], X[1])
