# keepa_tools.py
import os
from typing import List, Dict, Any, Optional
import keepa
from keepa import Keepa

api = Keepa(os.environ.get("KEEPA_API_KEY"))

# ----------------- helpers -----------------

def last_nonneg(seq):
    if seq is None:
        return None
    try:
        it = list(seq)
    except TypeError:
        return None
    for v in reversed(it):
        if v is not None and v != -1:
            return v
    return None

def _seller_label_from_offer(offer: Optional[dict]) -> Optional[str]:
    if not offer:
        return None
    if offer.get("isAmazon"):
        return "Amazon"
    if offer.get("isFBA"):
        return "3P (FBA)"
    return "3P (MFN)"

def _seller_label_from_id(seller_id: Optional[str], offer: Optional[dict] = None, domain: str = "US") -> Optional[str]:
    # Amazon's US seller ID
    if seller_id == "ATVPDKIKX0DER":
        return "Amazon"
    return _seller_label_from_offer(offer)

def _bool(x) -> Optional[bool]:
    return bool(x) if x is not None else None

def _pct_drop(now: Optional[float], avg: Optional[float]) -> Optional[float]:
    if not now or not avg:
        return None
    try:
        return round(100.0 * (1.0 - (now / avg)), 1)
    except ZeroDivisionError:
        return None

def _estimate_sales_velocity(rank_now: Optional[int]) -> Optional[str]:
    """
    VERY rough, category-agnostic bucket from current rank.
    You can refine later using category sizes.
    """
    if rank_now is None:
        return None
    if rank_now <= 5_000:
        return "High"
    if rank_now <= 30_000:
        return "Medium"
    if rank_now <= 150_000:
        return "Low"
    return "Very Low"

def _resellability_score(
    *,
    rank_now: Optional[int],
    offer_count_new: Optional[int],
    amazon_competing: bool,
    buybox_vs_avg90_pct: Optional[float],
    bb_is_fba: Optional[bool],
    review_rating: Optional[float],
    review_count: Optional[int],
) -> float:
    """
    0–100 heuristic score: higher = more attractive to buy/resell.
    Tuned to be simple and explainable; adjust weights as you learn.
    """
    score = 50.0

    # Demand (sales rank)
    if rank_now is not None:
        if rank_now <= 5_000:
            score += 25
        elif rank_now <= 30_000:
            score += 12
        elif rank_now <= 150_000:
            score += 3
        else:
            score -= 10

    # Competition (new offer count)
    if offer_count_new is not None:
        if offer_count_new <= 5:
            score += 12
        elif offer_count_new <= 10:
            score += 5
        elif offer_count_new <= 20:
            score -= 5
        else:
            score -= 12

    # Amazon competing is a strong negative
    if amazon_competing:
        score -= 20

    # Price posture vs 90d average: cheaper now is good
    if buybox_vs_avg90_pct is not None:
        # e.g., -18% means it's 18% below avg (good)
        if buybox_vs_avg90_pct >= 15:
            score += 10
        elif buybox_vs_avg90_pct >= 5:
            score += 5
        elif buybox_vs_avg90_pct <= -5:
            score -= 5  # currently above avg -> watch out for reversion

    # FBA buy box slightly preferred
    if bb_is_fba:
        score += 3

    # Review signals: quality & social proof
    if review_rating is not None and review_count is not None:
        if review_rating >= 4.2 and review_count >= 100:
            score += 5
        elif review_rating < 3.6 or review_count < 25:
            score -= 5

    # Clamp
    return max(0.0, min(100.0, round(score, 1)))

# ----------------- main API -----------------

def search_products(keyword: str, domain: str = "US", max_results: int = 10) -> List[Dict[str, Any]]:
    # 1) Find candidate ASINs
    asins = api.product_finder({"title": keyword}, domain=domain) or []
    if not asins:
        return []

    # 2) Query products with the right knobs for resellability
    #    - history=True to get time series (SALES, NEW, BUY_BOX_SHIPPING, COUNT_NEW, COUNT_USED)
    #    - stats=30 to get 30/90 aggregates (Keepa includes 90d stats inside stats)
    #    - buybox=True for Buy Box price history
    #    - offers=10 for per-offer info (sellerId, isAmazon, isFBA, isPrime, isBuyBoxWinner, etc.)
    products = api.query(
        asins[:max_results],
        domain=domain,
        history=True,
        stats=30,
        buybox=True,
        offers=20,
    )

    out: List[Dict[str, Any]] = []
    for p in products:
        p = p or {}
        data = p.get("data") or {}
        stats = p.get("stats") or {}
        offers = p.get("offers") or []

        # Current-ish values from histories
        price_buybox_now = last_nonneg(data.get("BUY_BOX_SHIPPING"))
        price_new_now = last_nonneg(data.get("NEW"))
        price_now = price_buybox_now if price_buybox_now is not None else price_new_now

        sales_rank_now = last_nonneg(data.get("SALES"))
        offer_count_new_now = last_nonneg(data.get("COUNT_NEW"))
        offer_count_used_now = last_nonneg(data.get("COUNT_USED"))

        # Buy Box identification
        bb_offer = next((o for o in offers if o.get("isBuyBoxWinner")), None)
        seller_id = stats.get("buyBoxSellerId") or (bb_offer or {}).get("sellerId")
        seller_label = _seller_label_from_id(seller_id, bb_offer, domain=domain)

        # Amazon competing (any offer is Amazon or seller_id == Amazon)
        amazon_competing = any(o.get("isAmazon") for o in offers) or (seller_id == "ATVPDKIKX0DER")

        # FBA/Prime flags for current BB if known
        bb_is_fba = _bool((bb_offer or {}).get("isFBA"))
        bb_is_prime = _bool((bb_offer or {}).get("isPrime"))

        # Deal posture vs 90-day Buy Box average (if available from stats)
        avg90_buybox = stats.get("buyBoxAvg90")
        deal_vs_avg90_pct = _pct_drop(price_now, avg90_buybox)

        # Reviews (if present in your client)
        review_rating = p.get("reviewRating")
        review_count = p.get("reviewCount")

        # Velocity bucket
        sales_velocity = _estimate_sales_velocity(sales_rank_now)

        # Compute resellability score
        score = _resellability_score(
            rank_now=sales_rank_now,
            offer_count_new=offer_count_new_now,
            amazon_competing=bool(amazon_competing),
            buybox_vs_avg90_pct=deal_vs_avg90_pct,
            bb_is_fba=bb_is_fba,
            review_rating=review_rating,
            review_count=review_count,
        )

        asin = p.get("asin")
        amazon_link = f"https://www.amazon.com/dp/{asin}" if asin else None

        out.append({
            # Identity
            "asin": asin,
            "title": p.get("title"),
            "brand": p.get("brand"),
            "link": amazon_link,

            # Price now & context
            "price_now": None if price_now is None else round(price_now, 2),
            "price_new_now": None if price_new_now is None else round(price_new_now, 2),
            "avg90_buybox": avg90_buybox,
            "deal_vs_avg90_pct": deal_vs_avg90_pct,  # e.g., 18.5 means 18.5% below avg90

            # Demand & reviews
            "sales_rank_now": sales_rank_now,
            "sales_velocity": sales_velocity,  # "High" / "Medium" / "Low" / "Very Low"
            "review_rating": review_rating,
            "review_count": review_count,

            # Competition landscape
            "offer_count_new_now": offer_count_new_now,
            "offer_count_used_now": offer_count_used_now,
            "amazon_competing": bool(amazon_competing),

            # Buy Box / seller at-a-glance
            "buybox_seller_id": seller_id,
            "seller": seller_label,        # "Amazon", "3P (FBA)", "3P (MFN)" or None
            "buybox_is_fba": bb_is_fba,
            "buybox_is_prime": bb_is_prime,

            # Simple 0–100 attractiveness score for reselling
            "resellability_score": score,
        })

    # Optional: sort best-first
    out.sort(key=lambda r: (r.get("resellability_score") or 0, -(r.get("deal_vs_avg90_pct") or 0)), reverse=True)
    return out


    # out = []
    # for p in products:
    #     data = (p or {}).get("data") or {}
    #     price_new_cents = last_nonneg(data.get("NEW"))
    #     buybox_cents = last_nonneg(data.get("BUY_BOX_SHIPPING"))
    #     out.append({
    #         "asin": p.get("asin"),
    #         "title": p.get("title"),
    #         "brand": p.get("brand"),
    #         "price_new": None if price_new_cents is None else round(price_new_cents / 100, 2),
    #         "price_buybox": None if buybox_cents is None else round(buybox_cents / 100, 2),
    #         "sales_rank": last_nonneg(data.get("SALES")),
    #     })
    # return out
