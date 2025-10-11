import pytest
import agents.keepa_tools as keepa_tools

# ---------------- Unit tests for helpers ----------------

def test_seller_label_from_offer_amazon():
    assert keepa_tools._seller_label_from_offer({"isAmazon": True}) == "Amazon"

def test_seller_label_from_offer_fba():
    assert keepa_tools._seller_label_from_offer({"isFBA": True}) == "3P (FBA)"

def test_seller_label_from_offer_mfn():
    assert keepa_tools._seller_label_from_offer({"isAmazon": False, "isFBA": False}) == "3P (MFN)"

def test_seller_label_from_id_amazon_magic():
    assert keepa_tools._seller_label_from_id("ATVPDKIKX0DER") == "Amazon"


# ---------------- Integration tests (monkeypatched Keepa API) ----------------

class _KeepaAPIBase:
    def product_finder(self, query, domain="US"):
        return ["B0ABCD1234"]
    def query(self, asins, domain="US", history=True, stats=30, buybox=True, offers=20):
        raise NotImplementedError

def _mk_common_product(**over):
    # minimal but compatible structure with keepa_tools.search_products expectations
    base = {
        "asin": "B0ABCD1234",
        "title": "Sample",
        "brand": "BrandX",
        "data": {
            "BUY_BOX_SHIPPING": [None, 1299],
            "NEW": [1299],
            "SALES": [20000],
            "COUNT_NEW": [10],
            "COUNT_USED": [0],
        },
        "stats": {
            "buyBoxAvg90": 1399,
        },
        "offers": []
    }
    # Allow overriding nested dicts simply:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return base

def _api_with_products(products):
    class _API(_KeepaAPIBase):
        def query(self, *args, **kwargs):
            return products
    return _API()

def test_keepa_populates_seller_from_buybox_fba(monkeypatch):
    # BB winner is a 3P FBA offer
    bb_offer = {"isBuyBoxWinner": True, "isFBA": True, "isPrime": True, "sellerId": "A3PSELLER"}
    prod = _mk_common_product(offers=[bb_offer])
    monkeypatch.setattr(keepa_tools, "api", _api_with_products([prod]))
    out = keepa_tools.search_products("iphone", max_results=1)
    row = out[0]
    assert row["seller"] == "3P (FBA)"
    assert row["buybox_is_fba"] is True

def test_keepa_populates_seller_amazon_when_stats_says_amazon(monkeypatch):
    prod = _mk_common_product(stats={"buyBoxSellerId": "ATVPDKIKX0DER"})
    monkeypatch.setattr(keepa_tools, "api", _api_with_products([prod]))
    out = keepa_tools.search_products("iphone", max_results=1)
    row = out[0]
    assert row["seller"] == "Amazon"
    assert row["amazon_competing"] is True

def test_keepa_populates_seller_mfn_when_winner_is_mfn(monkeypatch):
    bb_offer = {"isBuyBoxWinner": True, "isFBA": False, "sellerId": "MFNSELLER"}
    prod = _mk_common_product(offers=[bb_offer])
    monkeypatch.setattr(keepa_tools, "api", _api_with_products([prod]))
    out = keepa_tools.search_products("iphone", max_results=1)
    row = out[0]
    assert row["seller"] == "3P (MFN)"
    assert row["buybox_is_fba"] is False

def test_keepa_seller_none_when_no_offers_and_no_buybox(monkeypatch):
    # Legit empty sellers case: no offers, no stats.buyBoxSellerId
    prod = _mk_common_product()  # offers=[], stats without buyBoxSellerId
    monkeypatch.setattr(keepa_tools, "api", _api_with_products([prod]))
    out = keepa_tools.search_products("iphone", max_results=1)
    row = out[0]
    assert row["seller"] is None  # legitimate empty, not misparsed
    # Ensure other fields still computed
    assert row["price_now"] is not None
    assert "resellability_score" in row

def test_keepa_regression_sellers_not_none_when_offers_present(monkeypatch):
    # Regression guard for bug: sellers returned None despite offers/BB
    bb_offer = {"isBuyBoxWinner": True, "isFBA": True, "sellerId": "A3PSELLER"}
    prod = _mk_common_product(offers=[bb_offer])
    monkeypatch.setattr(keepa_tools, "api", _api_with_products([prod]))
    out = keepa_tools.search_products("iphone", max_results=1)
    assert out[0]["seller"] is not None

def test_keepa_prefers_stats_buybox_seller_over_offer_flag(monkeypatch):
    # When stats.buyBoxSellerId says Amazon, prefer that label
    bb_offer = {"isBuyBoxWinner": True, "isFBA": True, "sellerId": "A3PSELLER"}
    prod = _mk_common_product(offers=[bb_offer], stats={"buyBoxSellerId": "ATVPDKIKX0DER"})
    monkeypatch.setattr(keepa_tools, "api", _api_with_products([prod]))
    out = keepa_tools.search_products("iphone", max_results=1)
    row = out[0]
    assert row["seller"] == "Amazon"   # stats wins
    assert row["buybox_is_fba"] is True
