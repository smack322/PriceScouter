# tests/test_agent_keepa.py
import pytest
from agents.agent_keepa import keepa_search
import logging
# Helper to call keepa_search whether it's a LangChain Tool or a plain function
def _call_keepa_search(keyword, domain=None, max_results=None):
    if hasattr(keepa_search, "invoke"):
        payload = {"keyword": keyword}
        if domain is not None:
            payload["domain"] = domain
        if max_results is not None:
            payload["max_results"] = max_results
        return keepa_search.invoke(payload)
    # plain function path
    if domain is None and max_results is None:
        return keepa_search(keyword)
    if max_results is None:
        return keepa_search(keyword, domain)
    return keepa_search(keyword, domain, max_results)

def fake_search_products(keyword, domain="US", max_results=10):
    return [
        {"title": f"{keyword} Product 1", "asin": "B000000001", "price": 19.99, "domain": domain},
        {"title": f"{keyword} Product 2", "asin": "B000000002", "price": 25.50, "domain": domain},
    ][:max_results]

@pytest.fixture
def patch_search_products(monkeypatch):
    # IMPORTANT: patch the import site used by agent_keepa.py
    monkeypatch.setattr("agents.agent_keepa.search_products", fake_search_products)

def test_keepa_search_returns_products(patch_search_products):
    products = _call_keepa_search("iphone", domain="US", max_results=2)
    assert isinstance(products, list)
    assert len(products) == 2
    assert products[0]["title"] == "iphone Product 1"
    assert products[1]["asin"] == "B000000002"

def test_keepa_search_max_results(patch_search_products):
    products = _call_keepa_search("ipad", domain="US", max_results=1)
    assert len(products) == 1
    assert products[0]["title"] == "ipad Product 1"

def test_keepa_search_default_domain(patch_search_products):
    products = _call_keepa_search("macbook")
    assert len(products) >= 1
    assert all(p["domain"] == "US" for p in products)

def test_keepa_search_empty_result(monkeypatch):
    def empty_search_products(keyword, domain="US", max_results=10):
        return []
    # Patch the import site again for this test
    monkeypatch.setattr("agents.agent_keepa.search_products", empty_search_products)
    products = _call_keepa_search("nonexistent", domain="US", max_results=10)
    assert products == []

# ----------------------------
# New tests for enhanced fields
# ----------------------------

def test_keepa_search_includes_enhanced_fields(monkeypatch):
    """Agent should pass through enhanced Keepa fields when present."""
    def enriched_search_products(keyword, domain="US", max_results=10):
        return [
            {
                "title": f"{keyword} A",
                "asin": "B001AAAAAA",
                "price_now": 17.49,
                "sales_rank_now": 12345,
                "offer_count_new_now": 7,
                "offer_count_used_now": 0,
                "deal_vs_avg90_pct": 12.3,
                "resellability_score": 78.5,
                "seller": "3P (FBA)",
                "buybox_is_fba": True,
                "amazon_competing": False,
                "link": "https://www.amazon.com/dp/B001AAAAAA",
                "domain": domain,
            },
            {
                "title": f"{keyword} B",
                "asin": "B002BBBBBB",
                "price_now": 29.99,
                "sales_rank_now": 55555,
                "offer_count_new_now": 15,
                "offer_count_used_now": 2,
                "deal_vs_avg90_pct": 3.0,
                "resellability_score": 62.0,
                "seller": "Amazon",
                "buybox_is_fba": True,
                "amazon_competing": True,
                "link": "https://www.amazon.com/dp/B002BBBBBB",
                "domain": domain,
            },
        ][:max_results]

    monkeypatch.setattr("agents.agent_keepa.search_products", enriched_search_products)
    rows = _call_keepa_search("anker charger", domain="US", max_results=2)

    assert isinstance(rows, list) and len(rows) == 2
    row0 = rows[0]
    # Core passthrough fields
    assert row0["asin"] == "B001AAAAAA"
    assert row0["price_now"] == 17.49
    assert row0["sales_rank_now"] == 12345
    assert row0["offer_count_new_now"] == 7
    assert row0["deal_vs_avg90_pct"] == 12.3
    assert row0["resellability_score"] == 78.5
    # Flags/labels
    assert row0["seller"] == "3P (FBA)"
    assert row0["buybox_is_fba"] is True
    assert row0["amazon_competing"] is False
    # Link
    assert row0["link"].endswith("/B001AAAAAA")

def test_keepa_search_handles_missing_optional_fields(monkeypatch):
    """Agent should tolerate rows missing the optional enhanced keys."""
    def sparse_search_products(keyword, domain="US", max_results=10):
        # No enhanced fields at all
        return [
            {"title": f"{keyword} Minimal", "asin": "B009MINI01", "price": 9.99, "domain": domain}
        ][:max_results]

    monkeypatch.setattr("agents.agent_keepa.search_products", sparse_search_products)
    rows = _call_keepa_search("usb c cable", domain="US", max_results=1)

    assert isinstance(rows, list) and len(rows) == 1
    r = rows[0]
    # Must at least have these core keys coming through
    assert r["title"] == "usb c cable Minimal"
    assert r["asin"] == "B009MINI01"
    # Optional keys may be absent; assert no KeyError by using .get()
    assert r.get("resellability_score") is None or isinstance(r.get("resellability_score"), (int, float))
    assert r.get("sales_rank_now") is None or isinstance(r.get("sales_rank_now"), int)

def test_keepa_search_passes_through_badges_and_flags(monkeypatch):
    """Ensure deal badges/flags from upstream are preserved."""
    def flagged_search_products(keyword, domain="US", max_results=10):
        return [
            {
                "title": f"{keyword} Flagged",
                "asin": "B00FLAGGED",
                "price_now": 49.99,
                "deal_vs_avg90_pct": 18.0,   # big discount
                "amazon_competing": False,
                "buybox_is_fba": False,
                "seller": "3P (MFN)",
                "domain": domain,
            }
        ][:max_results]

    monkeypatch.setattr("agents.agent_keepa.search_products", flagged_search_products)
    rows = _call_keepa_search("mechanical keyboard", domain="US", max_results=1)

    assert isinstance(rows, list) and len(rows) == 1
    r = rows[0]
    assert r["asin"] == "B00FLAGGED"
    assert r["deal_vs_avg90_pct"] == 18.0
    assert r["amazon_competing"] is False
    assert r["buybox_is_fba"] is False
    assert r["seller"] == "3P (MFN)"
