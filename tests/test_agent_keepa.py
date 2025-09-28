import agents.agent_keepa
import pytest

def fake_search_products(keyword, domain="US", max_results=10):
    # Return a fixed result for testing
    return [
        {
            "title": f"{keyword} Product 1",
            "asin": "B000000001",
            "price": 19.99,
            "domain": domain
        },
        {
            "title": f"{keyword} Product 2",
            "asin": "B000000002",
            "price": 25.50,
            "domain": domain
        }
    ][:max_results]

@pytest.fixture
def patch_search_products(monkeypatch):
    monkeypatch.setattr(agent_keepa, "search_products", fake_search_products)

def test_keepa_search_returns_products(patch_search_products):
    products = agent_keepa.keepa_search("iphone", "US", 2)
    assert isinstance(products, list)
    assert len(products) == 2
    assert products[0]["title"] == "iphone Product 1"
    assert products[1]["asin"] == "B000000002"

def test_keepa_search_max_results(patch_search_products):
    products = agent_keepa.keepa_search("ipad", "US", 1)
    assert len(products) == 1
    assert products[0]["title"] == "ipad Product 1"

def test_keepa_search_default_domain(patch_search_products):
    products = agent_keepa.keepa_search("macbook")
    assert all(product["domain"] == "US" for product in products)

def test_keepa_search_empty_result(monkeypatch):
    def empty_search_products(keyword, domain="US", max_results=10):
        return []
    monkeypatch.setattr(agent_keepa, "search_products", empty_search_products)
    products = agent_keepa.keepa_search("nonexistent", "US", 10)
    assert products == []

