"""
Search eBay for a product and rank results by total cost (price + shipping)
Requires: Python 3.9+, requests

ENV VARS you must set:
- EBAY_CLIENT_ID
- EBAY_CLIENT_SECRET
Optional:
- EBAY_MARKETPLACE (default: EBAY_US)

Usage:
results = search_ebay_cheapest("iphone 15 case", zip_code="19406", country="US", limit=25, max_items=10)
for r in results: print(r)
"""
import base64
import os
import urllib.parse
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

#took out creds
load_dotenv(override=False)
## SANDBOX URL ##
# EBAY_TOKEN_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
# EBAY_BROWSE_SEARCH_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"

EBAY_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"

def _get_app_token() -> str:
    client_id = os.getenv("EBAY_CLIENT_ID")
    client_secret = os.getenv("EBAY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET environment variables.")
    b64 = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {b64}",
    }
    data = {
        "grant_type": "client_credentials",
        # Basic scope works for Browse search; add more scopes if you call other APIs
        "scope": "https://api.ebay.com/oauth/api_scope",
    }
    resp = requests.post(EBAY_TOKEN_URL, headers=headers, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]

def _headers(zip_code: str, country: str, token: str) -> Dict[str, str]:
    # Strongly recommended for accurate shipping estimates (+ needed for total-cost sorting)
    # Must be URL-encoded per eBay docs.
    # Example encoded: contextualLocation=country%3DUS%2Czip%3D19406
    ctx = urllib.parse.quote(f"country={country},zip={zip_code}")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "X-EBAY-C-MARKETPLACE-ID": os.getenv("EBAY_MARKETPLACE", "EBAY_US"),
        "X-EBAY-C-ENDUSERCTX": f"contextualLocation={ctx}",
    }

def _total_cost(item: Dict[str, Any]) -> float:
    price = float(item.get("price", {}).get("value", 0.0))
    ship = 0.0
    # shippingOptions may be absent depending on listing / site / headers
    for opt in item.get("shippingOptions", []) or []:
        cost = opt.get("shippingCost", {})
        if "value" in cost:
            ship = float(cost["value"])
            break
    return price + ship

def search_ebay_cheapest(
    query: str,
    zip_code: str,
    country: str = "US",
    limit: int = 50,
    max_items: int = 15,
) -> List[Dict[str, Any]]:
    """
    Returns a list of up to `max_items` items sorted by total cost (lowest first).
    Each dict includes: title, price, shipping, total, condition, seller, url, location, item_id, est_delivery
    """
    token = _get_app_token()
    headers = _headers(zip_code, country, token)

    params = {
        "q": query,
        # Sort by total cost (price + shipping) per eBay docs
        "sort": "price",
        "limit": str(limit),
        # Filter by delivery country to avoid cross-site surprises; add more filters as needed.
        # You can also add buyingOptions:{FIXED_PRICE} if you want to exclude auctions.
        "filter": f"deliveryCountry:{country}",
        # Uncomment to include shortDescription and itemLocation.city in response:
        # "fieldgroups": "EXTENDED",
    }

    resp = requests.get(EBAY_BROWSE_SEARCH_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("itemSummaries", []) or []

    results = []
    for it in items:
        price_val = float(it.get("price", {}).get("value", 0.0))
        ship_val = None
        est_min = est_max = None
        if it.get("shippingOptions"):
            so = it["shippingOptions"][0]
            if so.get("shippingCost") and "value" in so["shippingCost"]:
                ship_val = float(so["shippingCost"]["value"])
            est_min = so.get("minEstimatedDeliveryDate")
            est_max = so.get("maxEstimatedDeliveryDate")

        results.append({
            "title": it.get("title"),
            "item_id": it.get("itemId"),
            "condition": it.get("condition"),
            "seller": (it.get("seller") or {}).get("username"),
            "price": price_val,
            "shipping": ship_val,
            "total": _total_cost(it),
            "url": it.get("itemWebUrl") or it.get("itemHref"),
            "location": (it.get("itemLocation") or {}).get("city"),
            "est_delivery_min": est_min,
            "est_delivery_max": est_max,
        })

    # The API already sorts by total (price + shipping), but we defensively sort again.
    results.sort(key=lambda r: r["total"])
    return results[:max_items]

if __name__ == "__main__":
    # Demo: print the 5 cheapest iPhone 15 cases shipped to 19406
    for r in search_ebay_cheapest("iphone 15 case", zip_code="19406", country="US", max_items=5):
        print(f"${r['total']:.2f} total | price=${r['price']:.2f}"
              f"{'' if r['shipping'] is None else f', ship=${r['shipping']:.2f}'}"
              f" | {r['condition']} | {r['title']}\n{r['url']}\n")