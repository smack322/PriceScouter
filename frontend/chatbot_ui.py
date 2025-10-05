# streamlit_ui.py
# Streamlit UI for product search with optional SerpApi + Keepa + eBay agents
# pip install streamlit google-search-results pandas python-dotenv

import sys
import os
import time
import pandas as pd
import asyncio
import streamlit as st
from dotenv import load_dotenv

# Ensure package imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Agents (these should expose .ainvoke({...}) that return list[dict])
from agents.client import run_agents
from agents.agent_keepa import keepa_search
from agents.agent_ebay import ebay_search

# Optional SerpApi import (fallback path)
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except Exception:
    SERPAPI_AVAILABLE = False

# Optional serp_tools (preferred path)
try:
    from agents import serp_tools  # adjust import path if your serp_tools.py lives elsewhere
    SERP_TOOLS_AVAILABLE = True
except Exception:
    SERP_TOOLS_AVAILABLE = False

load_dotenv()

import re
import json
import numpy as np

_MAX_LEN = 256

# ----------------------------
# Input sanitization helpers
# ----------------------------
def _collapse_spaces(s: str) -> str:
    return " ".join(s.split())

def _strip_emoji_ascii_only(s: str) -> str:
    # Simple: remove non-ASCII (which strips 🍎 and others)
    return "".join(ch for ch in s if ord(ch) < 128)

def _remove_dangerous_tokens(s: str) -> str:
    # strip <script>...</script> and any other tags
    s = re.sub(r"(?is)<\s*script.*?>.*?<\s*/\s*script\s*>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    # neutralize classic SQL-ish bits
    s = s.replace("--", " ").replace("'", " ").replace(";", " ")
    s = re.sub(r"\bOR\b\s*1\s*=\s*1\b", " ", s, flags=re.IGNORECASE)
    return _collapse_spaces(s)

def _looks_dangerous(original: str) -> bool:
    low = original.lower()
    return (
        "<script" in low
        or re.search(r"\b(or)\b\s*1\s*=\s*1", low) is not None
        or "--" in original
        or "'" in original
        or ";" in original
    )

def sanitize_input_fn(user_input) -> dict:
    """
    Returns:
      {
        "error": None | "INPUT_EMPTY" | "INPUT_TOO_LONG" | "DANGEROUS",
        "message": str,
        "safe_input": str,
        "forwarded": bool
      }
    """
    s = "" if user_input is None else str(user_input)

    # empty
    if not s.strip():
        return {
            "error": "INPUT_EMPTY",
            "message": "User-facing message: Please enter a product or keyword.",
            "safe_input": "",
            "forwarded": False,
        }

    # base normalization
    base = _collapse_spaces(s.lower())
    no_emoji = _strip_emoji_ascii_only(base)
    cleaned = _remove_dangerous_tokens(no_emoji)

    # too long?
    too_long = len(cleaned) > _MAX_LEN
    if too_long:
        cleaned = cleaned[:_MAX_LEN]

    # dangerous?
    if _looks_dangerous(s):
        return {
            "error": "DANGEROUS",
            "message": "User-facing message: Your query contained unsafe content and was sanitized.",
            "safe_input": cleaned,
            "forwarded": False,
        }

    if too_long:
        return {
            "error": "INPUT_TOO_LONG",
            "message": "User-facing message: Your query was long; it has been truncated.",
            "safe_input": cleaned,
            "forwarded": False,
        }

    # happy path
    return {
        "error": None,
        "message": "",
        "safe_input": cleaned,
        "forwarded": True,
    }

# keep a thin wrapper if other code imports sanitize_input
def sanitize_input(s: str) -> str:
    return s.strip()

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="PriceScouter – Product Search", page_icon="🛒", layout="wide")

# ----------------------------
# Sidebar controls / settings
# ----------------------------
st.sidebar.title("Settings")
provider = st.sidebar.selectbox(
    "Data source",
    ["Mock data", "SerpApi – Google Shopping", "Keepa - Amazon", "eBay", "All"],
    help="Mock data requires no keys. SerpApi calls Google's Shopping results."
)

default_key = os.getenv("SERPAPI_API_KEY", "")
api_key = st.sidebar.text_input(
    "SerpApi API Key",
    value=default_key,
    type="password",
    help="Optional. Required only if using SerpApi."
)

location = st.sidebar.text_input(
    "Location (optional)",
    value="United States",
    help="Examples: 'United States', 'Philadelphia, Pennsylvania, United States'"
)

max_results = st.sidebar.slider("Max results", 5, 50, 20, step=5)

# Keep a persistent ZIP for eBay / All provider
zip_ebay = st.sidebar.text_input("Ship to ZIP (eBay/All)", value="19406")

st.sidebar.caption(
    "Tip: Leave provider as **Mock data** while designing the UI. "
    "Switch to SerpApi/Keepa/eBay when you're ready to test live calls."
)

# ----------------------------
# Provider helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def mock_search(q: str, n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "title": f"{q.title()} – Example Product {i}",
                "price": f"${9.99 + i:.2f}",
                "source": f"Vendor {((i - 1) % 5) + 1}",
                "rating": round(3.5 + (i % 5) * 0.3, 1),
                "link": f"https://example.com/{q.replace(' ', '-')}/{i}",
                "product_link": f"https://shopping.example.com/{q.replace(' ', '-')}/{i}",
            }
        )
    return pd.DataFrame(rows)

def _to_builtin(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return o

@st.cache_data(show_spinner=False)
def serpapi_search(q: str, key: str, loc: str | None, n: int) -> pd.DataFrame:
    """
    Preferred: use serp_tools.google_shopping_search() (new schema; no 'link').
    Fallback: direct SerpApi request (legacy fields including 'link' when present).
    """
    # Preferred path — serp_tools (your new module)
    if SERP_TOOLS_AVAILABLE:
        # serp_tools reads key from env; set it temporarily if user provided one
        if key:
            # Respect either variable name used inside serp_tools
            os.environ["SERPAPI_API_KEY"] = key
            os.environ["SERP_API_KEY"] = key
        raw_list = serp_tools.google_shopping_search(q=q, hl="en", gl="us", num=n, location=loc)
        df = pd.DataFrame([json.loads(json.dumps(r, default=_to_builtin)) for r in (raw_list or [])])

        if df.empty:
            return df

        # Normalize display price string expected by the rest of the UI
        if "price" in df.columns and df["price"].dtype != object:
            df["price_display"] = df["price"].apply(lambda x: f"${float(x):.2f}" if pd.notnull(x) else None)
        else:
            df["price_display"] = df.get("price")  # if already a string for some reason

        # Maintain older column names for compatibility
        if "seller" in df.columns and "source" not in df.columns:
            df["source"] = df["seller"]

        # Back-compat for link columns (serp_tools intentionally removed links)
        if "link" not in df.columns:
            df["link"] = None
        if "product_link" not in df.columns:
            df["product_link"] = None

        # For uniform downstream handling
        return df

    # Fallback path — direct SerpApi
    if not SERPAPI_AVAILABLE:
        raise RuntimeError("Neither serp_tools nor 'google-search-results' is available.")
    if not key:
        raise ValueError("Missing SerpApi API key.")

    params = {
        "engine": "google_shopping",
        "q": q,
        "hl": "en",
        "gl": "us",
        "num": str(n),
        "api_key": key,
    }
    if loc:
        params["location"] = loc
    search = GoogleSearch(params)
    results = search.get_dict()
    items = []
    for r in results.get("shopping_results", []) or []:
        items.append(
            {
                "title": r.get("title"),
                "price_display": r.get("price"),     # keep UI-friendly string
                "source": r.get("source") or r.get("seller"),
                "seller": r.get("seller") or r.get("source"),
                "rating": r.get("rating"),
                "link": r.get("link"),
                "product_link": r.get("product_link"),
                "price_str": r.get("price"),
            }
        )
    return pd.DataFrame(items)

@st.cache_data(show_spinner=False)
def keepa_agent_search(q: str, n: int = 20) -> pd.DataFrame:
    # Calls the keepa_search agent (async tool)
    items = asyncio.run(
        keepa_search.ainvoke({
            "keyword": q,
            "domain": "US",
            "max_results": n,
        })
    )
    df = pd.DataFrame(items) if items else pd.DataFrame([])

    # --- Normalize/augment display fields for Keepa results ---

    # Price display: prefer 'price_now', fallback to 'price_new' if needed
    if "price" not in df.columns:
        if "price_now" in df.columns:
            df["price_display"] = df["price_now"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
        elif "price_new" in df.columns:
            df["price_display"] = df["price_new"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
    else:
        # If test data already uses 'price' numeric/string, make a display column too
        if df["price"].dtype != object:
            df["price_display"] = df["price"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
        else:
            df["price_display"] = df["price"]

    # Link fallback from ASIN
    if "link" not in df.columns and "asin" in df.columns:
        df["link"] = df["asin"].apply(lambda asin: f"https://www.amazon.com/dp/{asin}" if pd.notnull(asin) else None)

    # Human-friendly badges
    if "deal_vs_avg90_pct" in df.columns:
        df["deal_badge"] = df["deal_vs_avg90_pct"].apply(
            lambda p: f"-{p:.0f}% vs 90d avg" if pd.notnull(p) and p > 0 else None
        )
    if "amazon_competing" in df.columns:
        df["compete"] = df["amazon_competing"].apply(lambda b: "Amazon competing" if bool(b) else "3P only")
    if "buybox_is_fba" in df.columns:
        df["fulfillment"] = df["buybox_is_fba"].apply(lambda b: "FBA" if bool(b) else "MFN")

    return df

@st.cache_data(show_spinner=False)
def ebay_agent_search(q: str, zip_code: str, n: int = 20) -> pd.DataFrame:
    items = asyncio.run(
        ebay_search.ainvoke({
            "keyword": q,
            "zip_code": zip_code,
            "country": "US",
            "limit": n,
            "max_results": n,
            "fixed_price_only": False,
            "sandbox": False,
        })
    )
    df = pd.DataFrame(items) if items else pd.DataFrame([])
    if "total" in df.columns and "price_display" not in df.columns:
        df["price_display"] = df["total"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
    if "url" in df.columns and "link" not in df.columns:
        df["link"] = df["url"]
    return df

# ----------------------------
# Normalization helpers
# ----------------------------
def normalize_price_str_to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None

# ----------------------------
# Header / Search bar
# ----------------------------
st.title("🛒 PriceScouter – Product Search")
st.write("Type a query, choose a data source, and view normalized results.")

with st.form("search_form"):
    q_raw = st.text_input("Search for a product", value="iphone 15 case", placeholder="e.g., 'wireless earbuds'")

    # Sanitize but still allow submission
    sani = sanitize_input_fn(q_raw)
    q = sani["safe_input"] if sani["safe_input"] else q_raw

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Best match",
                "Price (asc)",
                "Price (desc)",
                "Total cost (asc)",     # NEW: only active if column exists
                "Rating (desc)",
                # Keepa-aware sorters (only apply when columns exist)
                "Resellability (desc)",
                "Sales rank (asc)",
                "Deal vs 90d avg (desc)",
            ],
            help="Additional sorters activate when those columns exist."
        )
    with col2:
        min_price = st.number_input("Min $", min_value=0.0, value=0.0, step=1.0)
    with col3:
        max_price = st.number_input("Max $", min_value=0.0, value=0.0, step=1.0, help="0 = no max")
    submitted = st.form_submit_button("Search", use_container_width=True)

# User-facing sanitation messages
if submitted and sani["error"]:
    st.warning(sani["message"])

# ----------------------------
# Execute search
# ----------------------------
if submitted and q.strip():
    st.info(f"Searching **{provider}** for: _{q}_")
    with st.spinner("Fetching results..."):
        time.sleep(0.2)  # tiny delay so spinner shows
        try:
            if provider == "Mock data":
                df = mock_search(q, n=max_results)

            elif provider == "SerpApi – Google Shopping":
                df = serpapi_search(q, api_key, location, n=max_results)

            elif provider == "Keepa - Amazon":
                df = keepa_agent_search(q, n=max_results)

            elif provider == "eBay":
                df = ebay_agent_search(q, zip_ebay, n=max_results)

            elif provider == "All":  # Agents (Keepa + Google + eBay) orchestrated by your client agent
                rows = asyncio.run(
                    run_agents(
                        q,
                        zip_code=zip_ebay,
                        country="US",
                        max_price=(max_price or None),
                        top_n=max_results,
                    )
                )
                rows = json.loads(json.dumps(rows, default=lambda o: _to_builtin(o)))
                df = pd.DataFrame(rows) if rows else pd.DataFrame([])

                # Normalize some common fields for visual parity
                if "total" in df.columns and "price_display" not in df.columns:
                    df["price_display"] = df["total"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
                if "url" in df.columns and "link" not in df.columns:
                    df["link"] = df["url"]

            # --- Sorting/Filtering/Display (provider-agnostic) ---

            # Numeric price for filters/sorting if present (prefer price_display; else price_str; else price)
            price_source_col = "price_display" if "price_display" in df.columns else ("price_str" if "price_str" in df.columns else "price")
            if price_source_col in df.columns:
                df["price_value"] = df[price_source_col].apply(normalize_price_str_to_float)
            else:
                df["price_value"] = None

            # If serp_tools provided total_cost (numeric), mirror a display string for consistency
            if "total_cost" in df.columns and "total_cost_value" not in df.columns:
                df["total_cost_value"] = pd.to_numeric(df["total_cost"], errors="coerce")
                df["total_cost_display"] = df["total_cost_value"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)

            # Filters (by item price, not total)
            if min_price > 0:
                df = df[df["price_value"].fillna(10**9) >= min_price]
            if max_price > 0:
                df = df[df["price_value"].fillna(0) <= max_price]

            # Sorters
            if sort_by == "Price (asc)":
                df = df.sort_values(by=["price_value", "title"], ascending=[True, True], na_position="last")
            elif sort_by == "Price (desc)":
                df = df.sort_values(by=["price_value", "title"], ascending=[False, True], na_position="last")
            elif sort_by == "Total cost (asc)" and "total_cost_value" in df.columns:
                df = df.sort_values(by=["total_cost_value", "title"], ascending=[True, True], na_position="last")
            elif sort_by == "Rating (desc)" and "rating" in df.columns:
                df = df.sort_values(by=["rating", "title"], ascending=[False, True], na_position="last")
            elif sort_by == "Resellability (desc)" and "resellability_score" in df.columns:
                df = df.sort_values(by=["resellability_score", "title"], ascending=[False, True], na_position="last")
            elif sort_by == "Sales rank (asc)" and "sales_rank_now" in df.columns:
                df = df.sort_values(by=["sales_rank_now", "title"], ascending=[True, True], na_position="last")
            elif sort_by == "Deal vs 90d avg (desc)" and "deal_vs_avg90_pct" in df.columns:
                df = df.sort_values(by=["deal_vs_avg90_pct", "title"], ascending=[False, True], na_position="last")
            # else: Best match → leave incoming order

            # Columns to show: base + Keepa-aware + serp_tools-aware extras (only if present)
            base_cols = [
                "title",
                "price_display", "price_str",
                "source", "merchant", "seller", "seller_domain",
                "rating", "reviews_count",
                "link", "product_link", "_source"
            ]
            keepa_cols = [
                "compete", "fulfillment",
                "sales_rank_now", "offer_count_new_now", "offer_count_used_now",
                "deal_badge", "resellability_score",
            ]
            serp_cols = [
                "shipping_str", "shipping",
                "total_cost_display", "total_cost",
                "free_shipping", "in_store_pickup", "fast_delivery",
                "brand_guess", "condition_guess",
                "product_id", "currency_guess",
                "extensions", "delivery",
            ]

            candidate_cols = base_cols + keepa_cols + serp_cols
            seen, show_cols = set(), []
            for c in candidate_cols:
                if c in df.columns and c not in seen:
                    show_cols.append(c)
                    seen.add(c)

            st.caption(f"Found {len(df)} item(s)")
            st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, height=520)

            # Rich preview for quick scanning
            with st.expander("Preview top 5"):
                for _, row in df.head(5).iterrows():
                    title = row.get("title", "")
                    price = row.get("price_display") or row.get("price_str") or row.get("price", "")
                    src   = row.get("source") or row.get("merchant") or row.get("seller") or row.get("_source", "")
                    rating = row.get("rating", "—")
                    link = row.get("link") or row.get("product_link")

                    badges = []
                    if row.get("deal_badge"): badges.append(str(row["deal_badge"]))
                    if pd.notnull(row.get("sales_rank_now")): badges.append(f"Rank: {int(row['sales_rank_now'])}")
                    if pd.notnull(row.get("offer_count_new_now")): badges.append(f"Sellers: {int(row['offer_count_new_now'])}")
                    if pd.notnull(row.get("resellability_score")): badges.append(f"Score: {float(row['resellability_score']):.1f}")
                    if row.get("compete"): badges.append(str(row["compete"]))
                    if row.get("fulfillment"): badges.append(str(row["fulfillment"]))
                    if row.get("free_shipping"): badges.append("Free ship")
                    if row.get("in_store_pickup"): badges.append("Store pickup")
                    if row.get("fast_delivery"): badges.append("Fast delivery")
                    if row.get("total_cost_display"): badges.append(f"Total: {row['total_cost_display']}")
                    chips = "  ·  ".join(badges)

                    st.markdown(
                        f"**{title}**  \n"
                        f"Price: {price}  |  Source: {src}  |  Rating: {rating}  \n"
                        f"{chips}  \n"
                        f"{('[Open](' + str(link) + ')') if link else ''}"
                    )

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.write("👆 Enter a search term and click **Search** to see results. Use **Mock data** first to tweak the UI.")

st.markdown("---")
st.caption("For production, add more sources (eBay Browse, Walmart, etc.), unify schemas, and persist results to your DB.")

__all__ = ["sanitize_input_fn", "normalize_price_str_to_float", "sanitize_input"]
