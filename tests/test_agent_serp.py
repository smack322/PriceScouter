import pytest
import agents.agent_serp

def fake_google_shopping_search(q, num=20, location=None):
    # Return a fixed result for testing
    return [
        {
            "title": f"Product for {q}",
            "price": 19.99,
            "location": location or "Default",
        },
        {
            "title": f"Another {q}",
            "price": 25.00,
            "location": location or "Default",
        }
    ][:num]

@pytest.fixture
def patch_google_shopping_search(monkeypatch):
    monkeypatch.setattr(agent_google_shopping, "google_shopping_search", fake_google_shopping_search)

def test_google_shopping_basic(patch_google_shopping_search):
    results = agent_google_shopping.google_shopping("iphone case")
    assert isinstance(results, list)
    assert results[0]["title"].startswith("Product for iphone case")

def test_google_shopping_num_limit(patch_google_shopping_search):
    results = agent_google_shopping.google_shopping("ipad", num=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["title"] == "Product for ipad"

def test_google_shopping_location(patch_google_shopping_search):
    loc = "Philadelphia, Pennsylvania, United States"
    results = agent_google_shopping.google_shopping("macbook", location=loc)
    assert all(r["location"] == loc for r in results)

def test_google_shopping_empty_result(monkeypatch):
    def empty_search(q, num=20, location=None):
        return []
    monkeypatch.setattr(agent_google_shopping, "google_shopping_search", empty_search)
    results = agent_google_shopping.google_shopping("nonexistent")
    assert results == []
