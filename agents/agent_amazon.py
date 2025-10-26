# agents/agent_amazon.py
from typing import List, Optional
from langchain_core.tools import tool
from agents.amazon_tools import amazon_search

@tool
def amazon_products(
    q: str,
    page: int = 1,
    max_pages: int = 1,
    amazon_domain: str = "amazon.com",
    gl: str = "us",
) -> List[dict]:
    """Search Amazon via SerpApi and return normalized product rows."""
    return amazon_search(q=q, amazon_domain=amazon_domain, gl=gl, page=page, max_pages=max_pages)
