import pytest
from agents.keepa_tools import last_nonneg
import agents.keepa_tools as kt

def test_last_nonneg_basic():
    assert last_nonneg([None, -1, 5, None, 8]) == 8

def test_last_nonneg_none():
    assert last_nonneg(None) is None

def test_last_nonneg_empty():
    assert last_nonneg([]) is None
# -----------------------------
# _clamp_offers (optional)
# -----------------------------
@pytest.mark.parametrize(
    "inp, expected",
    [
        (None, None),
        (10, 20),    # below lower bound -> clamp up
        (20, 20),    # at lower bound
        (65, 65),    # mid-range
        (150, 100),  # above upper bound -> clamp down
    ],
)
def test_clamp_offers_bounds(inp, expected):
    # Only run if the helper exists in your keepa_tools
    if not hasattr(kt, "_clamp_offers"):
        pytest.skip("_clamp_offers not present in keepa_tools; skip bounds test.")
    assert kt._clamp_offers(inp) == expected

# -----------------------------
# Fake Keepa client for E2E tests
# -----------------------------
class _FakeKeepa:
    """
    Minimal fake that records the 'offers' param and returns controlled data
    compatible with keepa_tools.search_products expectations.
    """
    def __init__(self):
        self.last_query_kwargs = None
        self._asins = ["B001AAAAAA", "B002BBBBBB"]

    def product_finder(self, q, domain="US"):
        # Simulate product finder: return a fixed asin list when title is present
        title = (q or {}).get("title", "")
        return list(self._asins) if title else []

    def query(self, asins, domain="US", history=True, stats=30, buybox=True, **kwargs):
        # Record call
        self.last_query_kwargs = dict(
            asins=list(asins),
            domain=domain,
            history=history,
            stats=stats,
            buybox=buybox,
            **kwargs
        )
        # Build two fake products with representative fields
        # Prices are already in dollars (matching your code path)
        def _p(asin, *, buybox_price, new_price, sales_rank, buybox_seller_id=None, any_amazon_offer=False,
               offer_count_new=7, offer_count_used=0, avg90_buybox=25.0, bb_is_fba=True):
            return {
                "asin": asin,
                "title": f"Title for {asin}",
                "brand": "BrandX",
                "reviewRating": 4.3,
                "reviewCount": 123,
                # stats with 90-day buy box avg and buyBoxSellerId
                "stats": {
                    "buyBoxAvg90": avg90_buybox,
                    "buyBoxSellerId": buybox_seller_id,
                },
                # offers list; mark one buy box winner
                "offers": [
                    {
                        "sellerId": buybox_seller_id or "A3PSELLERID",
                        "isBuyBoxWinner": True,
                        "isAmazon": False,
                        "isFBA": bb_is_fba,
                        "isPrime": True,
                    },
                    {
                        "sellerId": "ATVPDKIKX0DER" if any_amazon_offer else "A1OTHERSLR",
                        "isBuyBoxWinner": False,
                        "isAmazon": any_amazon_offer,
                        "isFBA": False,
                        "isPrime": False,
                    },
                ],
                # time-series data
                "data": {
                    "BUY_BOX_SHIPPING": [None, -1, buybox_price],
                    "NEW": [None, -1, new_price],
                    "SALES": [None, -1, sales_rank],
                    "COUNT_NEW": [None, -1, offer_count_new],
                    "COUNT_USED": [None, -1, offer_count_used],
                },
            }

        p1 = _p(
            asins[0],
            buybox_price=21.99,
            new_price=23.50,
            sales_rank=12500,
            buybox_seller_id="A3PSELLERID",
            any_amazon_offer=False,
            offer_count_new=7,
            offer_count_used=0,
            avg90_buybox=25.00,
            bb_is_fba=True,
        )
        # Second product: Amazon competing + worse rank
        p2 = _p(
            asins[1],
            buybox_price=29.99,
            new_price=31.00,
            sales_rank=55555,
            buybox_seller_id="ATVPDKIKX0DER",  # Amazon
            any_amazon_offer=True,
            offer_count_new=18,
            offer_count_used=2,
            avg90_buybox=28.00,
            bb_is_fba=False,
        )
        return [p1, p2]

# Fixture to patch the API client used by keepa_tools
@pytest.fixture
def fake_keepa(monkeypatch):
    fk = _FakeKeepa()
    monkeypatch.setattr(kt, "api", fk)
    return fk

# -----------------------------
# search_products behavior
# -----------------------------
def test_search_products_offers_param_is_clamped(fake_keepa, monkeypatch):
    """
    Ensure the offers param sent to Keepa is within [20, 100] even if caller asks for less.
    """
    # If your search_products signature includes offers_requested, call with a low value
    func = getattr(kt, "search_products")
    try:
        # Try signature with offers_requested
        func(keyword="usb hub", domain="US", max_results=2, offers_requested=5)
    except TypeError:
        # Fallback to plain signature; env var override if implemented
        monkeypatch.setenv("KEEPA_OFFERS", "5")
        func(keyword="usb hub", domain="US", max_results=2)

    sent_offers = fake_keepa.last_query_kwargs.get("offers")
    assert sent_offers is not None
    assert 20 <= int(sent_offers) <= 100

def test_search_products_enriched_fields(fake_keepa):
    rows = kt.search_products(keyword="anker charger", domain="US", max_results=2)
    assert isinstance(rows, list) and len(rows) == 2

    r0 = rows[0]
    # Identity & link
    assert r0["asin"] == "B001AAAAAA"
    assert r0["title"].startswith("Title for")
    assert r0["link"].endswith("/B001AAAAAA")

    # Price selection prefers buy box; values rounded
    assert r0.get("price_now") in (21.99, 21.99) or r0.get("price") in (21.99, f"${21.99:.2f}")

    # Demand & competition
    assert r0["sales_rank_now"] == 12500
    assert r0["offer_count_new_now"] == 7
    assert r0["offer_count_used_now"] == 0

    # Deal posture vs 90-day avg (25 → now 21.99 ~ 12% below)
    assert r0["avg90_buybox"] == 25.0
    assert r0["deal_vs_avg90_pct"] == pytest.approx(12.0, abs=1.0)

    # Seller & flags: not Amazon, BB is FBA
    assert r0["seller"] in ("3P (FBA)", "Amazon", "3P (MFN)")
    assert r0["buybox_is_fba"] is True
    assert r0["amazon_competing"] is False

    # Score present and bounded
    assert isinstance(r0["resellability_score"], (int, float))
    assert 0.0 <= float(r0["resellability_score"]) <= 100.0

def test_search_products_amazon_competing_negative_signal(fake_keepa):
    rows = kt.search_products(keyword="something", domain="US", max_results=2)
    r1 = rows[1]  # our fake second product has Amazon competing
    assert r1["asin"] == "B002BBBBBB"
    assert r1["amazon_competing"] is True
    # Seller should be Amazon for this one
    assert r1["seller"] == "Amazon"

def test_search_products_handles_empty_asins(monkeypatch):
    fk = _FakeKeepa()
    fk._asins = []  # No results from product_finder
    monkeypatch.setattr(kt, "api", fk)

    rows = kt.search_products(keyword="no match", domain="US", max_results=5)
    assert rows == []