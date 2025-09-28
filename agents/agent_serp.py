from typing import List, Optional
from langchain_core.tools import tool
from agents.serp_tools import google_shopping_search

@tool
def google_shopping(q: str, num: int = 20, location: Optional[str] = None) -> List[dict]:
    """Search Google Shopping via SerpApi and return normalized product rows.
    Args:
        q: query text, e.g. "iphone 15 case"
        num: max results (default 20)
        location: optional location string to localize pricing (e.g., "Philadelphia, Pennsylvania, United States")
    """
    return google_shopping_search(q=q, num=num, location=location)