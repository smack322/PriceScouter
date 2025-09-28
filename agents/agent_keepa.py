# agent_keepa.py
from typing import List, Optional
from langchain_core.tools import tool

from agents.keepa_tools import search_products

@tool
def keepa_search(keyword: str, domain: str = "US", max_results: int = 10) -> List[dict]:
    """Search Amazon via Keepa by keyword and return structured product rows.
    Args:
        keyword: e.g. "iphone 15 case"
        domain: two-letter Keepa domain code (default "US")
        max_results: max products to return (default 10)
    """
    return search_products(keyword=keyword, domain=domain, max_results=max_results)
