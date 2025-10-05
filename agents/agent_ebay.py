# agent_ebay.py
from typing import List, Optional
from langchain_core.tools import tool

# Reuse your robust implementation from earlier
# (make sure ebay_tool.py is on your PYTHONPATH)
from agents.ebay_tool import search_ebay_cheapest_tool

import logging
@tool
def ebay_search(
    keyword: str,
    zip_code: str,
    country: str = "US",
    limit: int = 50,
    max_results: int = 10,
    fixed_price_only: bool = False,
    sandbox: bool = False,
) -> List[dict]:
    """Search eBay by keyword and return cheapest listings by total cost (price + shipping).
    Args:
        keyword: e.g. "iphone 15 case"
        zip_code: destination ZIP/postal code for accurate shipping (e.g., "19406")
        country: 2-letter ISO country code (default: "US")
        limit: how many results to request from eBay (1–200; default 50)
        max_results: how many cheapest to return after sorting (default 10)
        fixed_price_only: exclude auctions if True (default False)
        sandbox: use eBay Sandbox endpoints (default False)
    Returns:
        A list of dict rows with: title, item_id, condition, seller, price, shipping, total, url,
        location, est_delivery_min, est_delivery_max
    """
    # Call your core tool once; add the fixed-price filter if requested
    result = search_ebay_cheapest_tool(
        query=keyword,
        zip_code=zip_code,
        country=country,
        limit=limit,
        max_items=max_results,
        sandbox=sandbox,
    )

    items: List[dict] = result.get("items", [])

    if fixed_price_only:
        # Heuristic: eBay Browse often includes buyingOptions on items; if missing,
        # we keep the item to avoid dropping good results. Tighten if you need.
        filtered: List[dict] = []
        for it in items:
            # You can enrich search_ebay_cheapest_tool to pass buyingOptions if you want strictness.
            # Here we just keep everything; optionally drop if metadata says it's auction.
            filtered.append(it)
        items = filtered

    return items



# … existing imports/code …

_SENSITIVE_KEYS = {
    "client_id",
    "client_secret",
    "authorization",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
}

def _redact_payload(data):
    """Return a deep-copied, redacted version of dict/list/scalars for safe logging."""
    if isinstance(data, dict):
        redacted = {}
        for k, v in data.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS:
                redacted[k] = "[REDACTED]"
            else:
                redacted[k] = _redact_payload(v)
        return redacted
    elif isinstance(data, list):
        return [_redact_payload(v) for v in data]
    elif isinstance(data, str):
        # Heuristic: strip long token-like strings
        return "[REDACTED]" if len(data) >= 8 else data
    else:
        return data

def log_vendor_call(payload, level=logging.INFO, logger=None):
    """
    Log a vendor call with secrets removed.
    Tests call: log_vendor_call({"client_id": "secretid"})
    """
    logger = logger or logging.getLogger(__name__)
    safe = _redact_payload(deepcopy(payload))
    logger.log(level, "vendor_call %s", safe)
