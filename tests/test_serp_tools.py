import os
import pytest
from urllib.parse import urlparse, parse_qs, quote_plus

import agents.serp_tools as serp_tools


# -----------------------------
# Links restoration + fallback
# -----------------------------
def test_google_shopping_search_restores_links_and_builds_fallback(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "testkey")

    def fake_raw(**kwargs):
        return {
            "shopping_results": [
                # 1) Has explicit product_link + link
                {
                    "title": "Has Links",
                    "price": "$10.00",
                    "seller": "ShopA",
                    "link": "https://shopa.example/item-1",
                    "product_link": "https://www.google.com/shopping/product/ABC123",
                    "product_id": "ABC123",
                },
                # 2) Missing product_link but has product_id (should build fallback)
                {
                    "title": "Needs Fallback",
                    "price": "$12.00",
                    "source": "ShopB",
                    "product_id": "ZZZ999",
                },
                # 3) No product_id and no links (remain None)
                {
                    "title": "No Links",
                    "price": "$15.00",
                    "seller": "ShopC",
                },
            ]
        }

    monkeypatch.setattr(serp_tools, "google_shopping_search_raw", fake_raw)

    q = "iphone case"
    results = serp_tools.google_shopping_search(q)

    assert len(results) == 3

    # Case 1: pass-through
    assert results[0]["link"] == "https://shopa.example/item-1"
    assert results[0]["product_link"] == "https://www.google.com/shopping/product/ABC123"

    # Case 2: fallback builder used
    assert results[1]["product_link"] is not None
    url = results[1]["product_link"]
    assert url.startswith("https://www.google.com/shopping/product/ZZZ999")
    # ensure q param is present
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    assert qs.get("q", [""])[0] == q

    # and also ensure the raw URL has the encoded q
    assert f"q={quote_plus(q)}" in url

    # Case 3: still None when no product_id and no links
    assert results[2]["product_link"] is None
    assert results[2]["link"] is None


# ---------------------------------------
# Shipping parsing → total_cost + flags
# ---------------------------------------
def test_total_cost_and_free_shipping_flag_from_extensions(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "testkey")

    def fake_raw(**kwargs):
        return {
            "shopping_results": [
                {
                    "title": "Free ship via extensions",
                    "price": "$20.00",
                    "seller": "X",
                    "extensions": ["Free shipping", "Some other text"],
                    "shipping": None,   # ignored due to extension
                },
                {
                    "title": "Paid shipping",
                    "price": "$20.00",
                    "seller": "Y",
                    "shipping": "$5.00",
                    "extensions": [],
                },
                {
                    "title": "Unknown shipping",
                    "price": "$20.00",
                    "seller": "Z",
                    "shipping": None,   # treat as 0 for total
                    "extensions": [],
                },
            ]
        }

    monkeypatch.setattr(serp_tools, "google_shopping_search_raw", fake_raw)
    results = serp_tools.google_shopping_search("q")

    # 1) Free shipping via extension -> shipping parsed as 0.0; total_cost = 20.0
    r0 = results[0]
    assert r0["free_shipping"] is True
    assert r0["shipping"] == 0.0
    assert r0["total_cost"] == pytest.approx(20.0)

    # 2) Paid shipping -> total_cost = 25.0
    r1 = results[1]
    assert r1["free_shipping"] is False
    assert r1["shipping"] == pytest.approx(5.0)
    assert r1["total_cost"] == pytest.approx(25.0)

    # 3) Missing shipping -> treat as 0 for total; flag should be False
    r2 = results[2]
    assert r2["shipping"] is None
    assert r2["total_cost"] == pytest.approx(20.0)
    assert r2["free_shipping"] is False


# -----------------------------------------------------
# Condition + brand guessing; delivery/speed indicators
# -----------------------------------------------------
def test_brand_and_condition_guess_and_speed_flags(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "testkey")

    def fake_raw(**kwargs):
        return {
            "shopping_results": [
                {
                    "title": "Apple iPhone 13 (Renewed)",
                    "price": "$300.00",
                    "seller": "A",
                    "extensions": ["Free shipping", "In-store pickup available"],
                    "delivery": "Get it Today",  # fast_delivery => True
                },
                {
                    "title": "Samsung Galaxy S23 (Open Box)",
                    "price": "$450.00",
                    "seller": "B",
                    "extensions": ["Store pickup"],
                    "delivery": "2-day shipping",
                },
            ]
        }

    monkeypatch.setattr(serp_tools, "google_shopping_search_raw", fake_raw)
    res = serp_tools.google_shopping_search("phone")

    # Row 0
    r0 = res[0]
    assert r0["brand_guess"] == "Apple"
    assert r0["condition_guess"] == "Refurbished"
    assert r0["in_store_pickup"] is True
    assert r0["fast_delivery"] is True

    # Row 1
    r1 = res[1]
    assert r1["brand_guess"] == "Samsung"
    assert r1["condition_guess"] == "Open Box"
    assert r1["in_store_pickup"] is True
    # "2-day shipping" should not trigger the 'today/tomorrow/1 day' heuristic
    assert r1["fast_delivery"] is False


# -----------------------------------
# Shipping parser focused unit tests
# -----------------------------------
@pytest.mark.parametrize("shipping_str,extensions,expected_val,expected_flag", [
    ("$7.99", [], 7.99, False),
    ("From $7", [], 7.0, False),
    (None, ["Free shipping"], 0.0, True),
])
def test_parse_shipping_cases(shipping_str, extensions, expected_val, expected_flag):
    info = serp_tools._parse_shipping(shipping_str, extensions)
    assert info["shipping"] == (pytest.approx(expected_val) if expected_val is not None else None)
    free = ("free" in (shipping_str or "").lower()) or any("free shipping" in (e or "").lower() for e in (extensions or []))
    assert (expected_flag is True) == free


# ---------------------------------------------------
# sort_by_usd keeps unsupported currencies at the end
# ---------------------------------------------------
def test_sort_by_usd_places_unsupported_last():
    items = [
        {"price": 10, "currency": "EUR"},
        {"price": 10, "currency": "JPY"},  # unsupported in test rates
        {"price": 10, "currency": "USD"},
    ]
    norm = serp_tools.sort_by_usd(items)
    # JPY should have usd_price None and be last
    assert norm[-1]["currency"] == "JPY"
    assert norm[-1]["usd_price"] is None


# ---------------------------------------
# Allowlist/robots defaults and behavior
# ---------------------------------------
def test_allowlist_defaults_true(monkeypatch):
    # With _ALLOWED_HOSTS None, is_url_in_allowlist should be True
    monkeypatch.setattr(serp_tools, "_ALLOWED_HOSTS", None)
    assert serp_tools.is_url_in_allowlist("http://anything.example") is True

def test_fetch_url_combined(monkeypatch):
    # If allowlist passes but robots fails, we should get robots skip
    monkeypatch.setattr(serp_tools, "is_url_in_allowlist", lambda url: True)
    monkeypatch.setattr(serp_tools, "is_allowed_by_robots", lambda url: False)
    out = serp_tools.fetch_url("http://x.example/item")
    assert out["status"] == "skipped_robots"
