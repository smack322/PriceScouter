# keepa_tools.py
import os
import keepa
from keepa import Keepa

api = Keepa(os.environ.get("KEEPA_API_KEY"))

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

def search_products(keyword: str, domain: str = "US", max_results: int = 10):
    asins = api.product_finder({"title": keyword}, domain=domain) or []
    if not asins:
        return []

    products = api.query(
        asins[:max_results],
        domain=domain,
        history=True,
        stats=30,
        buybox=True
    )

    out = []
    for p in products:
        data = (p or {}).get("data") or {}
        price_new_cents = last_nonneg(data.get("NEW"))
        buybox_cents = last_nonneg(data.get("BUY_BOX_SHIPPING"))

        # Add link (Amazon product page)
        asin = p.get("asin")
        amazon_link = f"https://www.amazon.com/dp/{asin}" if asin else None

        out.append({
            "asin": asin,
            "title": p.get("title"),
            "brand": p.get("brand"),
            # Use buybox price if available, else new price
            "price": None if buybox_cents is None else round(buybox_cents, 2),
            "price_new": None if price_new_cents is None else round(price_new_cents, 2),
            "sales_rank": last_nonneg(data.get("SALES")),
            "seller": None,  # Keepa API may not return seller; set to None or enrich if possible
            "link": amazon_link,
        })
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
