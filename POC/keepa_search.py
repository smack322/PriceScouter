from keepa import Keepa
import os


api = Keepa("")

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
        history=True,   # we read prices from history arrays
        stats=30,
        buybox=True
    )

    out = []
    for p in products:
        data = p.get("data") or {}
        out.append({
            "asin": p.get("asin"),
            "title": p.get("title"),
            "brand": p.get("brand"),
            # Keepa returns cents
            "price_new_cents": last_nonneg(data.get("NEW")),
            "price_buybox_cents": last_nonneg(data.get("BUY_BOX_SHIPPING")),
            "sales_rank": last_nonneg(data.get("SALES")),
        })
    return out

if __name__ == "__main__":
    rows = search_products("iphone 15 case")
    for r in rows:
        print(r)
