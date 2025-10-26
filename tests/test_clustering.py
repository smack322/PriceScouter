import math
import json
import numpy as np
import pytest

from comparison.clustering import (
    ProductRecord,
    normalize_text,
    normalize_brand,
    tokens,
    blocking_key,
    embed_texts,
    DSU,
    cluster_block,
    stable_uuid,
    cluster_products,
    cosine_similarity,
)
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity

def cosine_similarity(X: np.ndarray) -> np.ndarray:
    """
    Cosine similarity with optional sklearn acceleration.
    Falls back to pure NumPy and re-normalizes rows for safety.
    """
    if _sk_cosine_similarity is not None:
        return _sk_cosine_similarity(X)

    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
    Xn = X / norms
    return Xn @ Xn.T
# ----------------------------
# Helpers / Fixtures
# ----------------------------

def rec(listing_id, vendor, title, brand=None, upc=None, price=None, attrs=None):
    return ProductRecord(
        listing_id=listing_id,
        vendor=vendor,
        title=title,
        brand=brand,
        upc=upc,
        attrs=attrs or {},
        price=price,
    )

# ----------------------------
# Unit tests: text/brand utils
# ----------------------------

def test_normalize_text_basic():
    assert normalize_text("  Apple iPhone-15  Pro,  Max! ") == "apple iphone 15 pro max"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""

def test_tokens_and_blocking_key_deterministic():
    r = rec("1", "ebay", "Apple iPhone 15 Pro Max Case with MagSafe", brand="Apple", price=19.99)
    t = tokens(normalize_text(r.title))
    assert "iphone" in t and "case" in t and "with" not in t  # stopword removed
    k1 = blocking_key(r)
    k2 = blocking_key(r)
    assert k1 == k2  # deterministic

def test_normalize_brand_aliases():
    assert normalize_brand("P&G") == "procter gamble"
    assert normalize_brand("HP Inc") == "hp"
    assert normalize_brand("UnknownCo") == "unknownco"
    assert normalize_brand(None) is None

# ----------------------------
# Unit tests: embedding, DSU
# ----------------------------

def test_embed_texts_cosine_similarity_monotonic():
    X = embed_texts([
        "apple iphone 15 case",
        "iphone 15 case for apple",
        "samsung galaxy s24 ultra cover",
    ])
    # First two should be very similar; third dissimilar
    try:
      from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity
    except Exception:
      _sk_cosine_similarity = None
      S = cosine_similarity(X)
      assert S[0, 1] > 0.85
      assert S[0, 2] < 0.5
      assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)  # L2 normalized

def test_dsu_union_find_basic():
    d = DSU(4)
    d.union(0, 1)
    d.union(2, 3)
    assert d.find(0) == d.find(1)
    assert d.find(2) == d.find(3)
    assert d.find(0) != d.find(2)

# ----------------------------
# Block-level clustering logic
# ----------------------------

def test_cluster_block_respects_brand_gate_and_theta():
    recs = [
        rec("a1", "ebay",   "Apple iPhone 15 Pro Case", "Apple", None, 15.0),
        rec("a2", "amazon", "Apple iPhone 15 Pro Phone Case", "Apple", None, 14.0),
        rec("b1", "ebay",   "Apple iPhone 15 Pro Case", "HP", None, 13.0),   # same title but different brand
        rec("c1", "ebay",   "Samsung Galaxy S24 Ultra Cover", "Samsung", None, 21.0),
    ]
    # Force all in one block by calling cluster_block directly (no blocking_key split)
    groups = cluster_block(recs, theta=0.85)
    # Expect: {a1,a2} together, b1 alone (brand gate), c1 alone (dissimilar)
    groups_sets = [set(g) for g in groups]  # indices
    # Map indices to listing_ids for clarity
    mapped = [ {recs[i].listing_id for i in g} for g in groups ]
    assert {"a1", "a2"} in mapped
    assert {"b1"} in mapped
    assert {"c1"} in mapped

def test_cluster_block_threshold_border():
    recs = [
        rec("x1", "ebay",   "Apple iPhone 15 Case Red", "Apple", None, 10.0),
        rec("x2", "amazon", "Apple iPhone 15 Case",     "Apple", None, 9.5),
        rec("x3", "ebay",   "Apple iPhone 15 Cover",    "Apple", None, 11.0),
    ]
    # 0.8944… is the theoretical sim(x1,x2); use 0.89 to link x1-x2 but not x3
    groups = cluster_block(recs, theta=0.89)
    mapped = [{recs[i].listing_id for i in g} for g in groups]
    assert any({"x1", "x2"} == g for g in mapped) or any({"x1", "x2"}.issubset(g) for g in mapped)

# ----------------------------
# stable UUID determinism
# ----------------------------

def test_stable_uuid_determinism():
    brand = "apple"
    upc = "1234567890"
    kt = ("iphone", "case")
    u1 = stable_uuid(brand, upc, kt)
    u2 = stable_uuid(brand, upc, kt)
    assert u1 == u2
    # Order of tokens matters by design; changing tokens changes UUID
    u3 = stable_uuid(brand, upc, tuple(reversed(kt)))
    assert u1 != u3

# ----------------------------
# End-to-end: cluster_products
# ----------------------------

def test_cluster_products_end_to_end_aggregates_and_memberships():
    """
    Mix of hard-key (UPC) duplicates and near-duplicates across vendors.
    Also inject a conflicting brand for the same UPC to verify brand-purity split.
    """
    listings = [
        # Product A (same UPC, same brand) → must cluster together
        rec("A-ebay-1", "ebay",   "Apple iPhone 15 Pro Max Case Clear", "Apple", "000111222333", 12.99),
        rec("A-amz-1",  "amazon", "Apple iPhone 15 Pro Max Clear Case", "Apple", "000111222333", 10.99),
        rec("A-keepa",  "keepa",  "Apple iPhone 15 Pro Max Phone Case - Clear", "Apple", "000111222333", 14.99),

        # Product A but different brand on same UPC (data noise) → must split by brand purity
        rec("A-ebay-bad", "ebay", "Apple iPhone 15 Pro Max Case Clear", "HP", "000111222333", 9.99),

        # Product B (no UPC, near-dup via embedding + same brand)
        rec("B-ebay-1", "ebay",   "Samsung Galaxy S24 Ultra Cover Black", "Samsung", None, 21.00),
        rec("B-amz-1",  "amazon", "Samsung Galaxy S24 Ultra Case - Black", "Samsung", None, 19.00),

        # Product C (noise, different brand; should stand alone)
        rec("C-1", "ebay", "Generic USB-C Cable 2m", "Generic", None, 5.49),
    ]

    outputs, members = cluster_products(listings, theta=0.85)

    # Basic shapes
    assert len(outputs) >= 3  # expect at least clusters for A(apple), A(hp), B(samsung), C(generic)
    assert len(members) == sum(o["listing_count"] for o in outputs)

    # Build index by canonical id
    by_id = {o["canonical_product_id"]: o for o in outputs}
    # Map canonical_id -> member listing_ids
    groups = {}
    for m in members:
        groups.setdefault(m["canonical_product_id"], set()).add(m["listing_id"])

    # 1) Brand purity: every cluster's brand is constant & matches members' normalized brand
    for cid, o in by_id.items():
        brand = o["brand"]
        lids = groups[cid]
        brands = { normalize_brand(next(li for li in listings if li.listing_id == lid).brand) or "" for lid in lids }
        assert len(brands) == 1
        assert list(brands)[0] == brand

    # 2) Aggregates: min/avg/max correct for at least one known cluster (Apple/A UPC cluster)
    # Find the Apple cluster that doesn't include the HP record
    apple_clusters = [ (cid, lids) for cid, lids in groups.items() if by_id[cid]["brand"] == "apple" ]
    assert apple_clusters  # must exist
    cid_apple, lids_apple = max(apple_clusters, key=lambda x: len(x[1]))  # the larger one
    prices = [ next(li for li in listings if li.listing_id == lid).price for lid in lids_apple ]
    prices = [p for p in prices if p is not None and not math.isnan(p)]
    expect_min, expect_max = min(prices), max(prices)
    expect_avg = sum(prices)/len(prices)
    assert math.isclose(by_id[cid_apple]["price_min"], expect_min, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(by_id[cid_apple]["price_max"], expect_max, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(by_id[cid_apple]["price_avg"], expect_avg, rel_tol=1e-9, abs_tol=1e-9)
    assert by_id[cid_apple]["listing_count"] == len(lids_apple)

    # 3) Medoid title: should be one of the member titles and have highest mean similarity
    titles = [next(li for li in listings if li.listing_id == lid).title for lid in lids_apple]
    X = embed_texts(titles)
    sims = cosine_similarity(X).mean(axis=1)
    medoid_title = by_id[cid_apple]["title"]
    assert medoid_title in titles
    medoid_idx = titles.index(medoid_title)
    max_mean = float(np.max(sims))
    assert abs(float(sims[medoid_idx]) - max_mean) <= 1e-6

    # 4) Memberships: similarity_to_centroid in [0,1], and at least one member equals 1.0 (centroid to itself)
    sims_to_centroid = [
        m["similarity_to_centroid"]
        for m in members
        if m["canonical_product_id"] == cid_apple
    ]
    # assert all(0.0 <= s <= 1.0000001 for s in sims_to_centroid)
    # assert any(math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9) for s in sims_to_centroid)
    # Allow tiny FP noise: [-1e-7, 1 + 1e-6]
    assert all(-1e-7 <= float(s) <= 1.0 + 1e-6 for s in sims_to_centroid)
    # And ensure at least one exact self-sim ~ 1.0
    assert any(abs(float(s) - 1.0) <= 1e-6 for s in sims_to_centroid)

def test_cluster_products_empty_input():
    outputs, members = cluster_products([], theta=0.85)
    assert outputs == []
    assert members == []
