# --- Add near the top with your other imports ---
from serpapi import GoogleSearch  # already present for Google Shopping
import typing as T
import os
import re
from typing import List, Optional, Dict, Any
from serpapi import GoogleSearch

from urllib.parse import quote_plus

from agents.serp_tools import (
    _to_float,
    _parse_shipping,
    _first_ext_flag,
    _guess_brand_from_title,
    _guess_condition_from_title,
)

# Reuse your existing helpers: _to_float, _parse_shipping, _first_ext_flag,
# _guess_condition_from_title, _guess_brand_from_title, convert_to_usd, etc.

# ---------- NEW: SerpApi Amazon (keyword) ----------
_ASIN_RE = re.compile(r"/dp/([A-Z0-9]{10})")

def _asin_from_link(link: Optional[str]) -> Optional[str]:
    if not link:
        return None
    m = _ASIN_RE.search(link)
    return m.group(1) if m else None

def amazon_search_raw(
    q: str,
    amazon_domain: str = "amazon.com",
    gl: str = "us",
    page: int = 1,
) -> Dict[str, Any]:
    """
    Low-level call to SerpApi's Amazon engine. Returns raw SerpApi response dict.
    Docs: https://serpapi.com/amazon-search-api
    """
    api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERP_API_KEY") or ""
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY not set. Export SERPAPI_API_KEY before running.")

    params = {
        "engine": "amazon",
        "k": q,                      # keyword
        "amazon_domain": amazon_domain,
        "gl": gl,                    # country code (two letters)
        "page": str(page),
        "api_key": api_key,
    }
    # Same client works for all SerpApi engines
    search = GoogleSearch(params)
    return search.get_dict()

def _as_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, set)):
        return " ".join(_as_text(v) for v in x)
    if isinstance(x, dict):
        # join dict values; tweak if you prefer keys too
        return " ".join(_as_text(v) for v in x.values())
    return str(x)

def _map_amazon_item(item: Dict[str, Any], q: str) -> Dict[str, Any]:
    title = (item.get("title") or "").strip() or "No title"
    link = item.get("link")
    asin = item.get("asin") or _asin_from_link(link)

    # --- price parsing (unchanged) ---
    price_raw = None
    price_val = None
    price_field = item.get("price")
    if isinstance(price_field, dict):
        price_raw = price_field.get("raw")
        price_val = price_field.get("extracted")
        if price_val is None:
            price_val = _to_float(price_raw)
    else:
        price_raw = price_field
        price_val = _to_float(price_field)

    # --- NEW: normalize delivery & extensions to strings ---
    delivery_raw = item.get("delivery")
    delivery = _as_text(delivery_raw)  # always a string now

    extensions_raw = item.get("extensions") or []
    if not isinstance(extensions_raw, list):
        extensions_raw = [extensions_raw] if extensions_raw else []
    extensions = [_as_text(e) for e in extensions_raw]

    ship_info = _parse_shipping(delivery, extensions)
    shipping = ship_info["shipping"]
    shipping_str = ship_info["shipping_str"]

    rating = item.get("rating")
    reviews = item.get("ratings_total") or item.get("reviews")
    seller = item.get("seller") or None
    badge = item.get("badge")

    brand_guess = _guess_brand_from_title(title)
    condition_guess = _guess_condition_from_title(title, extensions)

    free_shipping = (shipping == 0.0) or _first_ext_flag(extensions, "free shipping") or ("free delivery" in delivery.lower())
    fast_delivery = any(k in delivery.lower() for k in ["today", "tomorrow", "1 day", "two-day"])
    in_store_pickup = False

    total_cost = (price_val + (shipping or 0.0)) if price_val is not None else None
    product_link = link or (f"https://www.amazon.com/dp/{asin}" if asin else None)

    return {
        "title": title,
        "price": price_val,
        "price_str": price_raw,
        "seller": seller,
        "seller_domain": None,
        "link": link,
        "product_link": product_link,
        "rating": rating,
        "reviews_count": reviews,
        "product_id": asin,
        "extensions": extensions,        # now list[str]
        "delivery": delivery,            # now str
        "shipping_str": shipping_str,
        "shipping": shipping,
        "total_cost": total_cost,
        "free_shipping": free_shipping,
        "in_store_pickup": in_store_pickup,
        "fast_delivery": fast_delivery,
        "brand_guess": brand_guess,
        "condition_guess": condition_guess,
        "currency_guess": None,
        "badge": badge,
        "is_prime": bool(item.get("is_prime")) or ("prime" in delivery.lower()),
        "asin": asin,
        "source": "amazon_serp",
        "_source": "amazon",
    }


def amazon_search(
    q: str,
    amazon_domain: str = "amazon.com",
    gl: str = "us",
    page: int = 1,
    max_pages: int = 1,
) -> List[dict]:
    """
    High-level Amazon keyword search via SerpApi with simple pagination,
    returning normalized rows compatible with your Google Shopping path.
    """
    out: List[dict] = []
    seen: set[str] = set()

    for p in range(page, page + max_pages):
        raw = amazon_search_raw(q=q, amazon_domain=amazon_domain, gl=gl, page=p)

        if "error" in raw:
            # Surface the exact problem to callers/tests
            raise RuntimeError(f"SerpApi error: {raw['error']}")

        # SerpApi may use different buckets; gather what exists
        buckets: List[List[Dict[str, Any]]] = []
        for key in ("product_results", "organic_results", "shopping_results", "results"):
            val = raw.get(key)
            if isinstance(val, list):
                buckets.append(val)

        if not buckets:
            break  # nothing for this page

        for bucket in buckets:
            for item in bucket:
                row = _map_amazon_item(item, q)
                # de-dupe by ASIN or link to avoid repeats across modules/pages
                dedupe_key = row.get("asin") or row.get("product_link") or row.get("link")
                if not dedupe_key or dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                # keep only materially non-empty rows (avoid your previous “one empty row”)
                if any(v not in (None, "", [], {}) for v in row.values()):
                    out.append(row)

        # simple guardrail: if a page contributed nothing, stop early
        if len(seen) == 0:
            break

    return out
