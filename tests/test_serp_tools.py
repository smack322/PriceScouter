import pytest

import agents.serp_tools as serp_tools
import agents.ebay_tool as ebay_tool
import os
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

def test_eur_to_usd_conversion():
    assert serp_tools.convert_to_usd(10, "EUR") == pytest.approx(10.75)  # Example value

def test_gbp_to_usd_conversion():
    assert serp_tools.convert_to_usd(10, "GBP") == pytest.approx(12.50)

def test_unknown_currency_flagged():
    result = serp_tools.normalize_price("100", "XYZ")
    assert result["unsupported_currency"] is True

def test_mixed_currencies_sorted():
    items = [
        {"price": 10, "currency": "USD"},
        {"price": 10, "currency": "EUR"},
        {"price": 10, "currency": "GBP"},
    ]
    norm = serp_tools.sort_by_usd(items)
    prices = [i["usd_price"] for i in norm]
    assert prices == sorted(prices)

def test_robots_txt_disallowed_url(monkeypatch):
    monkeypatch.setattr(serp_tools, "is_allowed_by_robots", lambda url: False)
    result = serp_tools.fetch_url("http://disallowed.com/item")
    assert result["status"] == "skipped_robots"

def test_url_not_in_allowlist(monkeypatch):
    monkeypatch.setattr(serp_tools, "is_url_in_allowlist", lambda url: False)
    result = serp_tools.fetch_url("http://notallowed.com/item")
    assert result["status"] == "skipped_allowlist"

def test_fail_fast_without_env(monkeypatch):
    monkeypatch.delenv("EBAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("EBAY_CLIENT_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        ebay_tool._get_basic_auth_header()

def test_secrets_redacted_in_logs(monkeypatch, caplog):
    os.environ["EBAY_CLIENT_ID"] = "secretid"
    os.environ["EBAY_CLIENT_SECRET"] = "secretsecret"
    # Simulate function that logs
    with caplog.at_level("INFO"):
        ebay_tool.log_vendor_call({"client_id": os.environ["EBAY_CLIENT_ID"]})
    for record in caplog.records:
        assert "secretid" not in record.getMessage()
        assert "secretsecret" not in record.getMessage()

def test_no_hardcoded_secrets():
    # Simple static scan for 'secret' or API keys in source
    import glob
    files = glob.glob("agents/*.py")
    for fname in files:
        with open(fname) as f:
            content = f.read()
            assert "sk_test_" not in content
            assert "AIza" not in content
