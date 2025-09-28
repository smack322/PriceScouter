import pytest
import os
import agents.ebay_tool

# ------------------- Input Schema ----------------------

def test_ebay_search_input_valid():
    # Should succeed
    input_data = {
        "query": "iphone 15 case",
        "zip_code": "19406",
        "country": "US",
        "limit": 50,
        "max_items": 10,
        "fixed_price_only": False,
        "sandbox": False
    }
    obj = ebay_tool.EbaySearchInput(**input_data)
    assert obj.query == "iphone 15 case"
    assert obj.zip_code == "19406"

def test_ebay_search_input_empty_query():
    # Should fail validation
    with pytest.raises(ValueError):
        ebay_tool.EbaySearchInput(query="", zip_code="19406")

def test_ebay_search_input_strip_query():
    obj = ebay_tool.EbaySearchInput(query="  test  ", zip_code="19406")
    assert obj.query == "test"

# ------------------- Helper Functions ----------------------

def test_get_endpoints_prod_and_sandbox():
    prod = ebay_tool._get_endpoints(False)
    sb = ebay_tool._get_endpoints(True)
    assert "api.ebay.com" in prod[0]
    assert "sandbox.ebay.com" in sb[0]

def test_total_cost_computes():
    item = {
        "price": {"value": 10},
        "shippingOptions": [{"shippingCost": {"value": 2}}]
    }
    assert ebay_tool._total_cost(item) == 12

def test_total_cost_no_shipping():
    item = {"price": {"value": 10}}
    assert ebay_tool._total_cost(item) == 10

# ------------------- Main API (mocked) ----------------------

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("EBAY_CLIENT_ID", "fakeid")
    monkeypatch.setenv("EBAY_CLIENT_SECRET", "fakesecret")
    monkeypatch.setenv("EBAY_MARKETPLACE", "EBAY_US")

def fake_request_token(token_url):
    return "test_token"

def fake_get(url, headers, params, timeout):
    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {
                "itemSummaries": [
                    {
                        "title": "iPhone 15 Case",
                        "itemId": "ITEM123",
                        "condition": "New",
                        "seller": {"username": "seller1"},
                        "price": {"value": 9.99},
                        "shippingOptions": [{"shippingCost": {"value": 2.0},
                                            "minEstimatedDeliveryDate": "2025-10-01",
                                            "maxEstimatedDeliveryDate": "2025-10-05"}],
                        "itemWebUrl": "http://example.com/item",
                        "itemLocation": {"city": "NYC"},
                        "buyingOptions": ["FIXED_PRICE"]
                    }
                ]
            }
    return FakeResp()

def test_search_ebay_cheapest_tool(monkeypatch):
    # Patch network and token
    monkeypatch.setattr(ebay_tool, "_request_token", fake_request_token)
    monkeypatch.setattr(ebay_tool.requests, "get", fake_get)
    result = ebay_tool.search_ebay_cheapest_tool(
        query="iphone 15 case",
        zip_code="19406",
        country="US",
        limit=1,
        max_items=1,
        fixed_price_only=True,
        sandbox=False
    )
    assert "items" in result
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["title"] == "iPhone 15 Case"
    assert item["total"] == 11.99
    assert item["seller"] == "seller1"
    assert item["location"] == "NYC"
    assert item["buying_options"] == ["FIXED_PRICE"]

def test_search_ebay_cheapest_tool_excludes_auctions(monkeypatch):
    monkeypatch.setattr(ebay_tool, "_request_token", fake_request_token)
    def fake_get_auction(url, headers, params, timeout):
        class FakeResp:
            def raise_for_status(self): pass
            def json(self):
                return {
                    "itemSummaries": [
                        {
                            "title": "Auction Only",
                            "itemId": "ITEM999",
                            "condition": "Used",
                            "seller": {"username": "seller2"},
                            "price": {"value": 5.0},
                            "shippingOptions": [{"shippingCost": {"value": 2.0}}],
                            "itemWebUrl": "http://example.com/auction",
                            "itemLocation": {"city": "LA"},
                            "buyingOptions": ["AUCTION"]
                        }
                    ]
                }
        return FakeResp()
    monkeypatch.setattr(ebay_tool.requests, "get", fake_get_auction)
    result = ebay_tool.search_ebay_cheapest_tool(
        query="auction",
        zip_code="19406",
        fixed_price_only=True
    )
    assert result["items"] == []

def test_get_basic_auth_header(monkeypatch):
    monkeypatch.setenv("EBAY_CLIENT_ID", "foo")
    monkeypatch.setenv("EBAY_CLIENT_SECRET", "bar")
    # Should not raise
    header = ebay_tool._get_basic_auth_header()
    assert header.startswith("Basic ")

def test_get_basic_auth_header_missing(monkeypatch):
    monkeypatch.delenv("EBAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("EBAY_CLIENT_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        ebay_tool._get_basic_auth_header()

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
