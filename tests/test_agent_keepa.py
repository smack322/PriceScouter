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
