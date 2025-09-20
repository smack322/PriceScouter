import os
import re
from typing import List, Optional, Dict, Any
from serpapi import GoogleSearch

PRICE_RE = re.compile(r"([0-9]+(?:[.,][0-9]{1,2})?)")

def _to_float(price_text: Optional[str]) -> Optional[float]:
    """
    Parse a price string like '$12.99', '12,99 €', or '12.99 – 15.49' to a single float (lower bound).
    Returns None if not parseable.
    """
    if not price_text:
        return None
    # Take the first numeric group as the lower bound if it's a range.
    m = PRICE_RE.search(price_text.replace("\u2013", "-"))  # en dash -> hyphen
    if not m:
        return None
    num = m.group(1).replace(",", "")  # '1,299.00' -> '1299.00'
    try:
        return float(num)
    except ValueError:
        return None

def google_shopping_search_raw(
    q: str,
    hl: str = "en",
    gl: str = "us",
    num: int = 20,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Low-level call to SerpApi's Google Shopping engine. Returns raw SerpApi response dict.
    """
    api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERP_API_KEY") or ""
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY not set. Export SERPAPI_API_KEY before running.")

    params = {
        "engine": "google_shopping",
        "q": q,
        "hl": hl,
        "gl": gl,
        "num": str(num),
        "api_key": api_key,
    }
    if location:
        params["location"] = location

    search = GoogleSearch(params)
    return search.get_dict()

def google_shopping_search(
    q: str,
    hl: str = "en",
    gl: str = "us",
    num: int = 20,
    location: Optional[str] = None,
) -> List[dict]:
    """
    High-level wrapper that normalizes results into a consistent schema.

    Output schema (list of dicts):
    - title: str
    - price: Optional[float]   # lower-bound numeric price
    - price_str: Optional[str] # original price text
    - source: Optional[str]    # vendor/site
    - rating: Optional[float]
    - link: Optional[str]      # listing link
    - product_link: Optional[str] # product details on Google
    - currency_guess: Optional[str] # best-effort, from symbol if present
    """
    raw = google_shopping_search_raw(q=q, hl=hl, gl=gl, num=num, location=location)

    # Quick error visibility
    if "error" in raw:
        # Raise so the tool chain surfaces the failure (easier to debug in dev)
        raise RuntimeError(f"SerpApi error: {raw['error']}")

    out: List[dict] = []
    for r in raw.get("shopping_results", []) or []:
        price_str = r.get("price")
        price = _to_float(price_str)

        # Very light currency guess based on symbol
        currency_guess = None
        if isinstance(price_str, str):
            if "$" in price_str:
                currency_guess = "USD"
            elif "€" in price_str:
                currency_guess = "EUR"
            elif "£" in price_str:
                currency_guess = "GBP"

        out.append({
            "title": r.get("title"),
            "price": price,
            "price_str": price_str,
            "source": r.get("source"),
            "rating": r.get("rating"),
            "link": r.get("link"),
            "product_link": r.get("product_link"),
            "currency_guess": currency_guess,
        })
    return out
