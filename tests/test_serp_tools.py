import pytest

import agents.serp_tools

# --------- _to_float tests ---------

@pytest.mark.parametrize("input_str,expected", [
    ("$12.99", 12.99),
    ("12,99 €", 12.99),
    ("12.99 – 15.49", 12.99),
    ("", None),
    (None, None),
    ("Not a price", None),
    ("1,299.00", 1299.00),
    ("$1,299.00 – $1,399.00", 1299.00),
])
def test_to_float(input_str, expected):
    assert serp_tools._to_float(input_str) == expected

# --------- google_shopping_search_raw tests ---------

def test_google_shopping_search_raw_raises_on_no_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        serp_tools.google_shopping_search_raw("test product")

def test_google_shopping_search_raw_passes_params(monkeypatch):
    # Set API key
    monkeypatch.setenv("SERPAPI_API_KEY", "testkey")
    # Patch GoogleSearch to return expected dict
    class DummySearch:
        def __init__(self, params):
            self.params = params
        def get_dict(self):
            return {"shopping_results": [{"title": "item1", "price": "$9.99"}]}
    monkeypatch.setattr(serp_tools, "GoogleSearch", DummySearch)
    result = serp_tools.google_shopping_search_raw("test product", num=5, location="Philadelphia")
    assert "shopping_results" in result
    assert result["shopping_results"][0]["title"] == "item1"

# --------- google_shopping_search tests ---------

def test_google_shopping_search_normalizes(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "testkey")
    # Patch raw function to return dummy results
    def fake_raw(**kwargs):
        return {
            "shopping_results": [
                {
                    "title": "Test Product",
                    "price": "$19.99",
                    "seller": "BestBuy",
                    "link": "https://bestbuy.com/item",
                    "rating": 4.5,
                },
                {
                    "name": "Fallback Name",
                    "price": "12.49",
                    "source": "Amazon",
                    "product_link": "https://amazon.com/item",
                },
                {
                    "product_id": "123456",
                    "price": "15.00"
                }
            ]
        }
    monkeypatch.setattr(serp_tools, "google_shopping_search_raw", fake_raw)
    results = serp_tools.google_shopping_search("iphone case")
    assert len(results) == 3
    assert results[0]["title"] == "Test Product"
    assert results[0]["price"] == 19.99
    assert results[0]["seller"] == "BestBuy"
    assert results[1]["title"] == "Fallback Name"
    assert results[1]["seller"] == "Amazon"
    assert results[2]["link"].endswith("123456")

def test_google_shopping_search_error(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "testkey")
    def fake_raw(**kwargs):
        return {"error": "Something went wrong"}
    monkeypatch.setattr(serp_tools, "google_shopping_search_raw", fake_raw)
    with pytest.raises(RuntimeError):
        serp_tools.google_shopping_search("random")

def test_google_shopping_search_empty(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "testkey")
    def fake_raw(**kwargs):
        return {"shopping_results": []}
    monkeypatch.setattr(serp_tools, "google_shopping_search_raw", fake_raw)
    results = serp_tools.google_shopping_search("nothing")
    assert results == []
