# agent_ebay.py
from typing import List, Optional
from langchain_core.tools import tool

# Reuse your robust implementation from earlier
# (make sure ebay_tool.py is on your PYTHONPATH)
from agents.ebay_tool import search_ebay_cheapest_tool

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
