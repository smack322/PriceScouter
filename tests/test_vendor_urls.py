# tests/test_vendor_urls.py
import re
import pytest
from urllib.parse import urlparse

# --- eBay --------------------------------------------------------------------
import agents.ebay_tool as ebay_tools

def _is_http_url(u: str) -> bool:
    p = urlparse(u or "")
    return p.scheme in {"http", "https"} and bool(p.netloc)

def fake_token(_):
    return "access_token"

def _mk_item(**over):
    base = {
        "title": "iPhone 15 Case",
        "itemId": "ITEM123",
        "condition": "New",
        "seller": {"username": "seller1"},
        "price": {"value": 9.99},
        "shippingOptions": [{"shippingCost": {"value": 2.0}}],
        "itemWebUrl": "https://www.ebay.com/itm/ITEM123",
        "itemLocation": {"city": "NYC"},
        "buyingOptions": ["FIXED_PRICE"],
    }
    base.update(over)
    return base

class _Resp:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self): ...
    def json(self): return self._payload

def test_ebay_url_from_itemWebUrl(monkeypatch):
    monkeypatch.setattr(ebay_tools, "_request_token", fake_token)
    monkeypatch.setattr(ebay_tools.requests, "get",
        lambda url, headers, params, timeout: _Resp({"itemSummaries":[_mk_item()]}))
    out = ebay_tools.search_ebay_cheapest_tool("iphone", "19406", max_items=1)
    url = out["items"][0]["url"]
    assert _is_http_url(url) and urlparse(url).netloc.endswith("ebay.com")

def test_ebay_url_falls_back_to_itemHref(monkeypatch):
    monkeypatch.setattr(ebay_tools, "_request_token", fake_token)
    item = _mk_item(itemWebUrl=None, itemHref="https://www.ebay.com/itm/ITEM999")
    monkeypatch.setattr(ebay_tools.requests, "get",
        lambda url, headers, params, timeout: _Resp({"itemSummaries":[item]}))
    out = ebay_tools.search_ebay_cheapest_tool("iphone", "19406", max_items=1)
    url = out["items"][0]["url"]
    assert url == "https://www.ebay.com/itm/ITEM999"

def test_ebay_excludes_auction_when_fixed_price_only(monkeypatch):
    monkeypatch.setattr(ebay_tools, "_request_token", fake_token)
    item = _mk_item(buyingOptions=["AUCTION"])
    monkeypatch.setattr(ebay_tools.requests, "get",
        lambda url, headers, params, timeout: _Resp({"itemSummaries":[item]}))
    out = ebay_tools.search_ebay_cheapest_tool("iphone", "19406", max_items=5, fixed_price_only=True)
    assert out["items"] == []

# --- Amazon (Keepa) ----------------------------------------------------------

import agents.keepa_tools as keepa_tools

class _KeepaAPI:
    def product_finder(self, q, domain="US"):
        return ["B0ABCD1234"]
    def query(self, asins, domain="US", history=True, stats=30, buybox=True, offers=20):
        return [{
            "asin":"B0ABCD1234",
            "title":"iPhone 15 Case",
            "brand":"Acme",
            "data": {"BUY_BOX_SHIPPING":[None, 1299], "NEW":[1299]},
            "stats": {"buyBoxAvg90": 1399},
            "offers":[{"isBuyBoxWinner":True,"isFBA":True,"isPrime":True}]
        }]

def test_keepa_link_canonical_dp(monkeypatch):
    monkeypatch.setattr(keepa_tools, "api", _KeepaAPI())
    out = keepa_tools.search_products("iphone", max_results=1)
    link = out[0]["link"]
    p = urlparse(link or "")
    assert _is_http_url(link)
    assert p.netloc.endswith("amazon.com")
    assert re.search(r"/dp/B0ABCD1234", p.path)

def test_keepa_link_none_when_no_asin(monkeypatch):
    class _KeepaNoAsin(_KeepaAPI):
        def product_finder(self, q, domain="US"): return ["NOASIN"]
        def query(self, *args, **kwargs):
            return [{"title":"No Asin","data":{},"stats":{},"offers":[]}]
    monkeypatch.setattr(keepa_tools, "api", _KeepaNoAsin())
    out = keepa_tools.search_products("iphone", max_results=1)
    assert out[0]["link"] is None

# --- Google (Serp) -----------------------------------------------------------

import agents.serp_tools as serp_tools

class _SerpClient:
    def __init__(self, payload): self._p = payload
    def get_dict(self): return self._p

def _patch_serp(monkeypatch, payload):
    monkeypatch.setenv("SERPAPI_API_KEY", "x")
    # Patch the class used inside google_shopping_search_raw
    monkeypatch.setattr(serp_tools, "GoogleSearch", lambda params: _SerpClient(payload))

def test_serp_prefers_product_link(monkeypatch):
    payload = {"shopping_results":[
        {"title":"Case","price":"$9.99","product_link":"https://www.google.com/shopping/product/123",
         "link":"https://merchant.example/item", "product_id":"123"}
    ]}
    _patch_serp(monkeypatch, payload)
    rows = serp_tools.google_shopping_search("iphone")
    row = rows[0]
    assert row["product_link"] == "https://www.google.com/shopping/product/123"
    assert _is_http_url(row["product_link"])

def test_serp_builds_fallback_when_missing_product_link(monkeypatch):
    payload = {"shopping_results":[
        {"title":"Case","price":"$9.99","product_id":"999","link":"https://merchant.example/item"}
    ]}
    _patch_serp(monkeypatch, payload)
    rows = serp_tools.google_shopping_search("iphone")
    row = rows[0]
    assert row["product_link"].startswith("https://www.google.com/shopping/product/999")
    assert _is_http_url(row["product_link"])

def test_serp_total_cost_independent_of_link(monkeypatch):
    payload = {"shopping_results":[
        {"title":"A","price":"$10.00","shipping":"$2.00","product_link":"https://www.google.com/shopping/product/1"},
        {"title":"B","price":"$8.00","shipping":"Free","product_id":"2"},
    ]}
    _patch_serp(monkeypatch, payload)
    rows = serp_tools.google_shopping_search("iphone")
    # total_cost computed correctly
    assert rows[0]["total_cost"] == 12.0
    assert rows[1]["total_cost"] == 8.0
    assert _is_http_url(rows[0]["product_link"])
    assert _is_http_url(rows[1]["product_link"])
