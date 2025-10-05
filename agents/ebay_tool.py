# ebay_tool.py
from __future__ import annotations

import base64
import os
import logging
import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Load environment once (safe if imported multiple times)
load_dotenv(override=False)

# ---------------------------- Endpoints & config ---------------------------- #
EBAY_TOKEN_URL_PROD = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_SEARCH_URL_PROD = "https://api.ebay.com/buy/browse/v1/item_summary/search"

EBAY_TOKEN_URL_SB = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_SEARCH_URL_SB = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"

DEFAULT_MARKETPLACE = os.getenv("EBAY_MARKETPLACE", "EBAY_US")


def _get_endpoints(sandbox: bool):
    return (
        (EBAY_TOKEN_URL_SB, EBAY_BROWSE_SEARCH_URL_SB)
        if sandbox
        else (EBAY_TOKEN_URL_PROD, EBAY_BROWSE_SEARCH_URL_PROD)
    )


# ------------------------------- Schemas ----------------------------------- #
class EbaySearchInput(BaseModel):
    """Input schema for the tool (great for LangChain/OpenAI tools)."""

    query: str = Field(..., description="Product search query, e.g. 'iphone 15 case'")
    zip_code: str = Field(
        ..., description="Destination ZIP/postal code, e.g. '19406' (improves shipping estimates)."
    )
    country: str = Field(
        "US", description="2-letter ISO country code for delivery (e.g., 'US', 'CA')."
    )
    limit: int = Field(
        50, ge=1, le=200, description="How many results to request from eBay API (1–200)."
    )
    max_items: int = Field(
        15, ge=1, le=100, description="How many cheapest items to return after sorting."
    )
    fixed_price_only: bool = Field(
        False, description="If true, exclude auctions when metadata is available."
    )
    sandbox: bool = Field(
        False, description="Use eBay Sandbox endpoints (requires sandbox creds/listings)."
    )

    @validator("query")
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must be non-empty")
        return v.strip()


class EbayItem(BaseModel):
    title: Optional[str]
    item_id: Optional[str]
    condition: Optional[str]
    seller: Optional[str]
    price: Optional[float]
    shipping: Optional[float]
    total: float
    url: Optional[str]
    location: Optional[str]
    est_delivery_min: Optional[str]
    est_delivery_max: Optional[str]
    buying_options: Optional[List[str]] = None  # e.g., ["FIXED_PRICE", "AUCTION"]


class EbaySearchOutput(BaseModel):
    items: List[EbayItem]


# ------------------------------ HTTP helpers -------------------------------- #
def _get_basic_auth_header() -> str:
    client_id = os.getenv("EBAY_CLIENT_ID")
    client_secret = os.getenv("EBAY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET environment variables."
        )
    b64 = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return f"Basic {b64}"


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError, requests.Timeout)),
)
def _request_token(token_url: str) -> str:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": _get_basic_auth_header(),
    }
    data = {
        "grant_type": "client_credentials",
        # Basic scope works for Browse search; include more if you call other APIs.
        "scope": "https://api.ebay.com/oauth/api_scope",
    }
    resp = requests.post(token_url, headers=headers, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def _headers(zip_code: str, country: str, token: str) -> Dict[str, str]:
    # End-user context enables accurate shipping (required for total-cost sort).
    ctx = urllib.parse.quote(f"country={country},zip={zip_code}")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "X-EBAY-C-MARKETPLACE-ID": DEFAULT_MARKETPLACE,
        "X-EBAY-C-ENDUSERCTX": f"contextualLocation={ctx}",
    }


def _total_cost(item: Dict[str, Any]) -> float:
    price = float(item.get("price", {}).get("value", 0.0))
    ship = 0.0
    for opt in item.get("shippingOptions", []) or []:
        cost = opt.get("shippingCost", {})
        if "value" in cost:
            ship = float(cost["value"])
            break
    return price + ship


# ------------------------------- Public API --------------------------------- #
def search_ebay_cheapest_tool(
    query: str,
    zip_code: str,
    country: str = "US",
    limit: int = 50,
    max_items: int = 15,
    fixed_price_only: bool = False,
    sandbox: bool = False,
) -> Dict[str, Any]:
    """
    Tool: Search eBay and return up to `max_items` cheapest items by (price + shipping).

    Returns:
        Dict with key "items" mapping to a list of item dicts (see EbayItem fields).
    """
    token_url, search_url = _get_endpoints(sandbox)
    token = _request_token(token_url)
    headers = _headers(zip_code, country, token)

    params = {
        "q": query,
        # eBay 'price' sort is (price + shipping) when ENDUSERCTX includes location
        "sort": "price",
        "limit": str(limit),
        # Filter by delivery country to avoid cross-site surprises
        "filter": f"deliveryCountry:{country}",
        # Request extra metadata that often includes buyingOptions
        # (safe to omit; not all sites/listings return it)
        "fieldgroups": "BUYING_OPTIONS",
    }

    resp = requests.get(search_url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items_raw = data.get("itemSummaries", []) or []

    results: List[Dict[str, Any]] = []
    for it in items_raw:
        price_val = float((it.get("price") or {}).get("value") or 0.0)
        ship_val = None
        est_min = est_max = None
        if it.get("shippingOptions"):
            so = it["shippingOptions"][0]
            if (so.get("shippingCost") or {}).get("value") is not None:
                ship_val = float(so["shippingCost"]["value"])
            est_min = so.get("minEstimatedDeliveryDate")
            est_max = so.get("maxEstimatedDeliveryDate")

        row = {
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
            "buying_options": it.get("buyingOptions"),
        }

        # Optionally exclude auctions when we have the metadata
        if fixed_price_only:
            bo = (row.get("buying_options") or [])
            if bo and ("FIXED_PRICE" not in bo):
                continue

        results.append(row)

    # Defensive sort by total even though the API sorts by price+shipping already.
    results.sort(key=lambda r: r["total"])
    items_model = [EbayItem(**r) for r in results[:max_items]]
    return EbaySearchOutput(items=items_model).dict()


# ------------------- Optional: LangChain StructuredTool --------------------- #
try:
    from langchain.tools import StructuredTool  # langchain==0.2+
    ebay_structured_tool = StructuredTool.from_function(
        name="search_ebay_cheapest",
        description=(
            "Search eBay for a product and return the cheapest results by total cost "
            "(price + shipping) using the buyer's ZIP & country for accurate shipping."
        ),
        func=search_ebay_cheapest_tool,
        args_schema=EbaySearchInput,
        return_direct=False,
    )
except Exception:
    # Allow importing the module even if LangChain isn't installed.
    ebay_structured_tool = None


# ------------------- Optional: OpenAI tool/function spec -------------------- #
OPENAI_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_ebay_cheapest",
        "description": "Search eBay and return up to max_items cheapest listings by price + shipping.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Product search query."},
                "zip_code": {"type": "string", "description": "Destination ZIP/postal code."},
                "country": {"type": "string", "default": "US", "description": "2-letter ISO code."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                "max_items": {"type": "integer", "minimum": 1, "maximum": 100, "default": 15},
                "fixed_price_only": {"type": "boolean", "default": False},
                "sandbox": {"type": "boolean", "default": False},
            },
            "required": ["query", "zip_code"],
            "additionalProperties": False,
        },
    },
}

import logging
from copy import deepcopy

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

__all__ = [
    # ...existing exports...
    "log_vendor_call",
]
