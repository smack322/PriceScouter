# frontend/chatbot_ui.py
# Streamlit UI for Product Search (Google Shopping via SerpApi or serp_tools, Keepa, eBay, or All)
# AC enforced: remove these from final payload/UI:
#   price_str, source, seller_domain, link, shipping_str, free_shipping,
#   in_store_pickup, fast_delivery, brand_guess, condition_guess, currency_guess

import sys
import os
import re
import time
import json
import asyncio
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import pathlib

# Ensure project root (PriceScouter) is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# from ui_results import render_results

# --- Project import path (assumes this file is in frontend/ and project root is one level up) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Agents (must expose async .ainvoke or equivalent) ---
from agents.client import run_agents
from agents.agent_keepa import keepa_search
from agents.agent_ebay import ebay_search
from agents.agent_amazon import amazon_products

from backend.local_db.vector_db.index_sync import sync_canonical_index
from comparison.canonical_service import search_canonical_with_sync

@st.cache_resource
def ensure_index_synced_once():
    # Runs once per process, avoids repeated rebuilds on every rerun
    source_count, index_count = sync_canonical_index(force=False)
    return source_count, index_count

# Call this early in your app
source_count, index_count = ensure_index_synced_once()
st.sidebar.write(f"Canonical index: {index_count} vectors (source rows: {source_count})")

from comparison.canonical_service import search_canonical_with_sync

query = st.text_input("What product are you looking for?")

if st.button("Search"):
    # your existing multi-vendor flow here...

    # Show canonical / similar products section
    st.subheader("Canonical / Similar Products")
    canonical_results = search_canonical_with_sync(query, k=5)

    for r in canonical_results:
        st.write(f"**{r['title']}**")
        st.write(
            f"Avg price: {r['avg_price']} (min {r['min_price']}, "
            f"max {r['max_price']}) — sellers: {r['seller_count']}, "
            f"listings: {r['total_listings']}"
        )
        if r["representative_url"]:
            st.write(r["representative_url"])
        st.write("---")
# Optional SerpApi import (fallback path)
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except Exception:
    SERPAPI_AVAILABLE = False

# Preferred serp_tools (your helper that returns cleaner schema)
try:
    from agents import serp_tools
    SERP_TOOLS_AVAILABLE = True
except Exception:
    SERP_TOOLS_AVAILABLE = False


try:
    from agents import amazon_tools  # exposes amazon_tools.amazon_search(...)
    AMAZON_TOOLS_AVAILABLE = True
    AMAZON_TOOLS_HAS_AMAZON = hasattr(amazon_tools, "amazon_search")
except Exception:
    AMAZON_TOOLS_AVAILABLE = False
    AMAZON_TOOLS_HAS_AMAZON = False
load_dotenv()

_MAX_LEN = 256

# --------------------------------------------------------------------------------------
# Acceptance Criteria – BANNED columns (must not be in final payload/UI)
# --------------------------------------------------------------------------------------
BANNED_GOOGLE_UI_COLS = {
    "price_str",
    "source",
    "seller_domain",
    "link",
    "shipping_str",
    "free_shipping",
    "in_store_pickup",
    "fast_delivery",
    "brand_guess",
    "condition_guess",
    "currency_guess",
}

# --------------------------------------------------------------------------------------
# Input sanitation
# --------------------------------------------------------------------------------------
def _collapse_spaces(s: str) -> str:
    return " ".join(s.split())

def _strip_emoji_ascii_only(s: str) -> str:
    return "".join(ch for ch in s if ord(ch) < 128)

def _remove_dangerous_tokens(s: str) -> str:
    s = re.sub(r"(?is)<\s*script.*?>.*?<\s*/\s*script\s*>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
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
    s = "" if user_input is None else str(user_input)
    if not s.strip():
        return {"error": "INPUT_EMPTY", "message": "Please enter a product or keyword.", "safe_input": "", "forwarded": False}

    base = _collapse_spaces(s.lower())
    no_emoji = _strip_emoji_ascii_only(base)
    cleaned = _remove_dangerous_tokens(no_emoji)

    too_long = len(cleaned) > _MAX_LEN
    if too_long:
        cleaned = cleaned[:_MAX_LEN]

    if _looks_dangerous(s):
        return {"error": "DANGEROUS", "message": "Your query contained unsafe content and was sanitized.", "safe_input": cleaned, "forwarded": False}
    if too_long:
        return {"error": "INPUT_TOO_LONG", "message": "Your query was long; it has been truncated.", "safe_input": cleaned, "forwarded": False}

    return {"error": None, "message": "", "safe_input": cleaned, "forwarded": True}

def sanitize_input(s: str) -> str:
    return s.strip()

def _to_float(s):
    """Robust parse of a price-like string into float; returns None if not parseable."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    try:
        # take first number; allow comma or dot decimals
        m = re.search(r"(\d+(?:[.,]\d{1,2})?)", str(s))
        if not m:
            return None
        num = m.group(1).replace(",", "")
        return float(num)
    except Exception:
        return None
# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _to_builtin(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return o

def normalize_price_str_to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None

# --------------------------------------------------------------------------------------
# Data providers
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def mock_search(q: str, n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "title": f"{q.title()} – Example Product {i}",
                "price": 9.99 + i,
                "price_display": f"${9.99 + i:.2f}",
                "merchant": f"Vendor {((i - 1) % 5) + 1}",
                "rating": round(3.5 + (i % 5) * 0.3, 1),
                "product_link": f"https://shopping.example.com/{q.replace(' ', '-')}/{i}",
                "_source": "mock",
            }
        )
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def serpapi_search(q: str, key: str, loc: str | None, n: int) -> pd.DataFrame:
    """
    Preferred: serp_tools.google_shopping_search() which may omit raw 'link' fields.
    Fallback: direct SerpApi which returns 'link' and other fields (we'll drop banned ones later).
    """
    # Preferred path — serp_tools
    if SERP_TOOLS_AVAILABLE:
        if key:
            os.environ["SERPAPI_API_KEY"] = key
            os.environ["SERP_API_KEY"] = key
        raw_list = serp_tools.google_shopping_search(q=q, hl="en", gl="us", num=n, location=loc)
        df = pd.DataFrame([json.loads(json.dumps(r, default=_to_builtin)) for r in (raw_list or [])])
        if df.empty:
            return df

        # Price display
        if "price_display" not in df.columns:
            if "price" in df.columns and df["price"].dtype != object:
                df["price_display"] = df["price"].apply(lambda x: f"${float(x):.2f}" if pd.notnull(x) else None)
            else:
                df["price_display"] = df.get("price")

        # Normalize seller
        if "merchant" not in df.columns and "seller" in df.columns:
            df["merchant"] = df["seller"]

        # Ensure product_link exists (serp_tools may already provide)
        if "product_link" not in df.columns:
            df["product_link"] = None

        # Tag source for debugging
        if "_source" not in df.columns:
            df["_source"] = "google"

        return df

    # Fallback — direct SerpApi
    if not SERPAPI_AVAILABLE:
        raise RuntimeError("Neither serp_tools nor 'google-search-results' is available.")
    if not key:
        raise ValueError("Missing SerpApi API key.")

    params = {"engine": "google_shopping", "q": q, "hl": "en", "gl": "us", "num": str(n), "api_key": key}
    if loc:
        params["location"] = loc
    search = GoogleSearch(params)
    results = search.get_dict()
    items = []
    for r in results.get("shopping_results", []) or []:
        items.append(
            {
                "title": r.get("title"),
                "price_display": r.get("price"),
                "price": normalize_price_str_to_float(r.get("price")),
                "merchant": r.get("seller") or r.get("source"),
                "rating": r.get("rating"),
                "product_link": r.get("product_link") or r.get("link"),
                "_source": "google",
                # NOTE: We intentionally do NOT keep any banned columns here; even if SerpApi returns them,
                # we won't display them downstream due to the AC drop step.
            }
        )
    return pd.DataFrame(items)

# Put near serpapi_search()

_ASIN_RE = re.compile(r"/dp/([A-Z0-9]{10})")

def _asin_from_link(link: str | None) -> str | None:
    if not link:
        return None
    m = _ASIN_RE.search(link)
    return m.group(1) if m else None

def _loc_to_amazon_domain_and_gl(loc: str | None) -> tuple[str, str]:
    """Very light mapping; expand as needed."""
    if not loc:
        return "amazon.com", "us"
    s = loc.lower()
    if "united kingdom" in s or "uk" == s.strip():
        return "amazon.co.uk", "uk"
    if "germany" in s or "de" == s.strip():
        return "amazon.de", "de"
    if "france" in s or "fr" == s.strip():
        return "amazon.fr", "fr"
    if "canada" in s or "ca" == s.strip():
        return "amazon.ca", "ca"
    if "spain" in s or "es" == s.strip():
        return "amazon.es", "es"
    if "italy" in s or "it" == s.strip():
        return "amazon.it", "it"
    # default
    return "amazon.com", "us"

@st.cache_data(show_spinner=False)
def amazon_search(q: str, key: str, loc: str | None, n: int) -> pd.DataFrame:
    """
    Preferred: amazon_tools.amazon_search() (normalized rows).
    Fallback: direct SerpApi Amazon engine with pagination until we gather <= n items.
    """
    # ---------- Preferred path: serp_tools ----------
    if AMAZON_TOOLS_AVAILABLE and AMAZON_TOOLS_HAS_AMAZON:
        if key:
            os.environ["SERPAPI_API_KEY"] = key
            os.environ["SERP_API_KEY"] = key
        amazon_domain, gl = _loc_to_amazon_domain_and_gl(loc)
        raw_list = amazon_tools.amazon_search(q=q, amazon_domain=amazon_domain, gl=gl, page=1, max_pages=max(1, (n // 16) + 1))
        df = pd.DataFrame([json.loads(json.dumps(r, default=_to_builtin)) for r in (raw_list or [])])
        if df.empty:
            return df

        # price_display
        if "price_display" not in df.columns:
            if "price" in df.columns and df["price"].dtype != object:
                df["price_display"] = df["price"].apply(lambda x: f"${float(x):.2f}" if pd.notnull(x) else None)
            else:
                df["price_display"] = df.get("price")

        # merchant normalization
        if "merchant" not in df.columns and "seller" in df.columns:
            df["merchant"] = df["seller"]

        # product_link fallback
        if "product_link" not in df.columns:
            df["product_link"] = None

        # 1) take link where present
        if "link" in df.columns:
            df["product_link"] = df["product_link"].where(
                df["product_link"].notna(),
                df["link"]
            )

        # 2) backfill missing with dp/ASIN
        if "asin" in df.columns:
            df["product_link"] = df["product_link"].where(
                df["product_link"].notna(),
                df["asin"].apply(lambda a: f"https://www.amazon.com/dp/{a}" if pd.notnull(a) else None)
            )

        # 3) normalize scheme (add https:// if someone returned a bare host/path)
        df["product_link"] = df["product_link"].apply(
            lambda u: (f"https://{u}" if isinstance(u, str) and u and not u.startswith(("http://", "https://")) else u)
        )

        # source tag
        df["_source"] = "amazon"

        return df.head(n)

    # ---------- Fallback: direct SerpApi ----------
    if not SERPAPI_AVAILABLE:
        raise RuntimeError("Neither amazon_tools nor 'google-search-results' is available.")
    if not key:
        raise ValueError("Missing SerpApi API key.")

    os.environ["SERPAPI_API_KEY"] = key
    os.environ["SERP_API_KEY"] = key

    from serpapi import GoogleSearch  # local import to keep optional

    amazon_domain, gl = _loc_to_amazon_domain_and_gl(loc)
    items: list[dict] = []
    seen: set[str] = set()

    # SerpApi Amazon uses 'page' for pagination; collect until we hit n or run out.
    page = 1
    while len(items) < n and page <= 5:  # hard guardrail
        params = {
            "engine": "amazon",
            "k": q,
            "amazon_domain": amazon_domain,
            "gl": gl,
            "page": str(page),
            "api_key": key,
        }
        results = GoogleSearch(params).get_dict()

        buckets = []
        for key_ in ("product_results", "organic_results", "shopping_results", "results"):
            if isinstance(results.get(key_), list):
                buckets.extend(results[key_])

        if not buckets:
            break

        for r in buckets:
            title = r.get("title")
            link = r.get("link")
            asin = r.get("asin") or _asin_from_link(link)
            dedupe_key = asin or link
            if not title or not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            price_raw = r.get("price")
            if isinstance(price_raw, dict):
                price_display = price_raw.get("raw")
                price_val = price_raw.get("extracted")
                if price_val is None:
                    price_val = normalize_price_str_to_float(price_display)
            else:
                price_display = price_raw
                price_val = normalize_price_str_to_float(price_raw)

            # delivery text often contains shipping hints; keep as-is (UI drops banned columns later if needed)
            delivery = r.get("delivery")
            merchant = r.get("seller") or r.get("source")

            # prefer direct product link; fallback to dp/ASIN
            product_link = link or (f"https://{amazon_domain}/dp/{asin}" if asin else None)

            items.append(
                {
                    "title": title,
                    "price_display": price_display,
                    "price": price_val,
                    "merchant": merchant,
                    "rating": r.get("rating"),
                    "reviews_count": r.get("ratings_total") or r.get("reviews"),
                    "product_link": product_link,
                    "asin": asin,
                    "_source": "amazon",
                    # keep other keys if you want (badge/is_prime/etc.), your AC drop step will remove banned ones
                }
            )
            if len(items) >= n:
                break

        page += 1
    df = pd.DataFrame(items[:n])

    if not df.empty:
        df["_source"] = "amazon"

    return df

@st.cache_data(show_spinner=False)
def keepa_agent_search(q: str, n: int = 20) -> pd.DataFrame:
    items = asyncio.run(
        keepa_search.ainvoke({"keyword": q, "domain": "US", "max_results": n})
    )
    df = pd.DataFrame(items) if items else pd.DataFrame([])

    if df.empty:
        return df

    if "price_display" not in df.columns:
        if "display_price" in df.columns:
            df["price_display"] = df["display_price"]
        elif "price_now" in df.columns and df["price_now"].notna().any():
            df["price_display"] = df["price_now"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
        elif "price_new" in df.columns and df["price_new"].notna().any():
            df["price_display"] = df["price_new"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
        else:
            df["price_display"] = None

    if "link" not in df.columns and "asin" in df.columns:
        df["product_link"] = df["asin"].apply(lambda asin: f"https://www.amazon.com/dp/{asin}" if pd.notnull(asin) else None)
    elif "product_link" not in df.columns and "url" in df.columns:
        df["product_link"] = df["url"]

    if "_source" not in df.columns:
        df["_source"] = "amazon"

    # Helpful badges
    if "deal_vs_avg90_pct" in df.columns:
        df["deal_badge"] = df["deal_vs_avg90_pct"].apply(lambda p: f"-{p:.0f}% vs 90d avg" if pd.notnull(p) and p > 0 else None)
    if "amazon_competing" in df.columns:
        df["compete"] = df["amazon_competing"].apply(lambda b: "Amazon competing" if bool(b) else "3P only")
    if "buybox_is_fba" in df.columns:
        df["fulfillment"] = df["buybox_is_fba"].apply(lambda b: "FBA" if bool(b) else "MFN")

    return df

@st.cache_data(show_spinner=False)
def ebay_agent_search(q: str, zip_code: str, n: int = 20) -> pd.DataFrame:
    items = asyncio.run(
        ebay_search.ainvoke(
            {
                "keyword": q,
                "zip_code": zip_code,
                "country": "US",
                "limit": n,
                "max_results": n,
                "fixed_price_only": False,
                "sandbox": False,
            }
        )
    )
    df = pd.DataFrame(items) if items else pd.DataFrame([])
    if "price_display" not in df.columns:
        if "total" in df.columns:
            df["price_display"] = df["total"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
        else:
            df["price_display"] = None
    if "product_link" not in df.columns and "url" in df.columns:
        df["product_link"] = df["url"]
    if "_source" not in df.columns:
        df["_source"] = "ebay"
    return df

# --------------------------------------------------------------------------------------
# Streamlit page config & sidebar
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="PriceScouter – Chatbot Search UI", page_icon="🤖", layout="wide")
# render_results()

st.sidebar.title("Settings")
provider = st.sidebar.selectbox(
    "Data source",
    ["Mock data", "SerpApi – Google Shopping", "SerpApi – Amazon", "eBay", "All"],
    help="Mock data requires no keys. SerpApi calls Google's Shopping results."
)

default_key = os.getenv("SERPAPI_API_KEY", "")
api_key = st.sidebar.text_input("SerpApi API Key", value=default_key, type="password")
location = st.sidebar.text_input("Location (optional)", value="United States")
max_results = st.sidebar.slider("Max results", 5, 50, 20, step=5)
zip_ebay = st.sidebar.text_input("Ship to ZIP (eBay/All)", value="19406")

st.sidebar.caption("Tip: Start with Mock data while designing. Switch to live providers when ready.")

# --------------------------------------------------------------------------------------
# Header / Search
# --------------------------------------------------------------------------------------
st.title("🤖 PriceScouter – Chatbot UI")
st.write("Search a product, pick a provider, and view normalized results. AC-enforced: no banned Google columns are shown or exported.")

with st.form("search_form"):
    q_raw = st.text_input("Search for a product", value="iphone 15 case", placeholder="e.g., 'wireless earbuds'")
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
                "Total cost (asc)",
                "Rating (desc)",
                "Resellability (desc)",
                "Sales rank (asc)",
                "Deal vs 90d avg (desc)",
            ],
            help="Some sorters activate only when those columns exist."
        )
    with col2:
        min_price = st.number_input("Min $", min_value=0.0, value=0.0, step=1.0)
    with col3:
        max_price = st.number_input("Max $", min_value=0.0, value=0.0, step=1.0, help="0 = no max")

    submitted = st.form_submit_button("Search", use_container_width=True)

if submitted and sani["error"]:
    st.warning(sani["message"])

# --------------------------------------------------------------------------------------
# Execute search
# --------------------------------------------------------------------------------------
if submitted and q.strip():
    st.info(f"Searching **{provider}** for: _{q}_")
    with st.spinner("Fetching results..."):
        time.sleep(0.2)
        try:
            # -------- Provider calls --------
            if provider == "Mock data":
                df = mock_search(q, n=max_results)

            elif provider == "SerpApi – Google Shopping":
                df = serpapi_search(q, api_key, location, n=max_results)

            elif provider == "SerpApi – Amazon":
                df = amazon_search(q, api_key, location, n=max_results)

            elif provider == "eBay":
                df = ebay_agent_search(q, zip_ebay, n=max_results)

            elif provider == "All":
                rows = asyncio.run(
                    run_agents(q, zip_code=zip_ebay, country="US", max_price=(max_price or None), top_n=max_results)
                )
                rows = json.loads(json.dumps(rows, default=lambda o: _to_builtin(o)))
                df = pd.DataFrame(rows) if rows else pd.DataFrame([])
                # Harmonize common fields
                if "price_display" not in df.columns:
                    if "total" in df.columns:
                        df["price_display"] = df["total"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
                    elif "price" in df.columns:
                        df["price_display"] = df["price"].apply(lambda x: f"${float(x):.2f}" if pd.notnull(x) else None)
                    else:
                        df["price_display"] = None
                if "product_link" not in df.columns and "url" in df.columns:
                    df["product_link"] = df["url"]

            # -------- Provider-agnostic normalization --------
            if not df.empty:
                # Ensure product_link is a clean URL or None
                if "product_link" in df.columns:
                    df["product_link"] = pd.Series(df["product_link"]).where(pd.notna(df["product_link"]), None)
                    df["product_link"] = df["product_link"].apply(
                        lambda u: (f"https://{u}" if isinstance(u, str) and u and not u.startswith(("http://", "https://")) else u)
                    )

                # Numeric price for sort/filter
                price_source_col = "price_display" if "price_display" in df.columns else "price"
                if price_source_col in df.columns:
                    df["price_value"] = df[price_source_col].apply(normalize_price_str_to_float)
                else:
                    df["price_value"] = None

                # Optional total cost handling (if present)
                if "total_cost" in df.columns and "total_cost_value" not in df.columns:
                    df["total_cost_value"] = pd.to_numeric(df["total_cost"], errors="coerce")
                    df["total_cost_display"] = df["total_cost_value"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)

                # ----------------- Acceptance Criteria ENFORCEMENT -----------------
                # Drop banned columns so they do not appear in the final payload/UI
                df = df.drop(columns=[c for c in BANNED_GOOGLE_UI_COLS if c in df.columns], errors="ignore")

                # ----------------- Filters -----------------
                if min_price > 0:
                    df = df[df["price_value"].fillna(10**9) >= min_price]
                if max_price > 0:
                    df = df[df["price_value"].fillna(0) <= max_price]

                # ----------------- Sorters -----------------
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
                # else Best match → keep incoming order

            # -------- Display table --------
            # Columns we are willing to show (explicitly excludes banned ones)
            base_cols = [
                "title",
                "price_display",
                "merchant", "seller",          # prefer merchant/seller; 'source' is banned
                "rating", "reviews_count",
                "product_link",
                "_source",
            ]
            keepa_cols = [
                "compete", "fulfillment",
                "sales_rank_now", "offer_count_new_now", "offer_count_used_now",
                "deal_badge", "resellability_score",
            ]
            serp_cols = [
                "shipping",                    # a numeric/parsed field you might compute elsewhere
                "total_cost_display", "total_cost",
                "product_id",
                "extensions", "delivery",
            ]

            candidate_cols = base_cols + keepa_cols + serp_cols
            show_cols, seen = [], set()
            for c in candidate_cols:
                if c in df.columns and c not in seen:
                    show_cols.append(c)
                    seen.add(c)

            # Link column config (ONLY product_link; 'link' is banned)
            col_config = {}
            if "product_link" in df.columns:
                col_config["product_link"] = st.column_config.LinkColumn(
                    label="Product Link",
                    display_text="Open",
                    help="Opens product page in a new tab"
                )

            st.caption(f"Found {len(df)} item(s)")
            st.dataframe(
                df[show_cols] if show_cols else df,
                use_container_width=True,
                height=520,
                column_config=col_config
            )

            # -------- Preview top 5 (schema-safe, with banned fields removed) --------
            with st.expander("Preview top 5"):
                for _, row in df.head(5).iterrows():
                    raw_title = row.get("title") or row.get("name") or row.get("product_title")
                    title = str(raw_title) if pd.notna(raw_title) else "(untitled)"

                    raw_price = row.get("price_display") or row.get("price")
                    price = str(raw_price) if pd.notna(raw_price) else "—"

                    # Prefer merchant/seller; 'source' is banned
                    raw_seller = row.get("merchant") or row.get("seller") or row.get("_source")
                    seller = str(raw_seller) if pd.notna(raw_seller) else "—"

                    rating_val = row.get("rating")
                    rating = "—" if pd.isna(rating_val) else rating_val

                    # Only use product_link (since 'link' is banned)
                    plink = row.get("product_link")
                    link = plink if isinstance(plink, str) and plink else None

                    # Badges (exclude shipping_str/free_shipping/in_store_pickup/fast_delivery per AC)
                    badges = []
                    if isinstance(row.get("deal_badge"), str) and row["deal_badge"]:
                        badges.append(row["deal_badge"])

                    sr = row.get("sales_rank_now")
                    if pd.notna(sr):
                        try: badges.append(f"Rank: {int(sr)}")
                        except Exception: pass

                    ocn = row.get("offer_count_new_now")
                    if pd.notna(ocn):
                        try: badges.append(f"Sellers: {int(ocn)}")
                        except Exception: pass

                    rs = row.get("resellability_score")
                    if pd.notna(rs):
                        try: badges.append(f"Score: {float(rs):.1f}")
                        except Exception: pass

                    if isinstance(row.get("compete"), str) and row["compete"]:
                        badges.append(row["compete"])
                    if isinstance(row.get("fulfillment"), str) and row["fulfillment"]:
                        badges.append(row["fulfillment"])

                    if isinstance(row.get("total_cost_display"), str) and row["total_cost_display"]:
                        badges.append(f"Total: {row['total_cost_display']}")

                    chips = "  ·  ".join(badges)

                    st.markdown(
                        f"**{title}**  \n"
                        f"Price: {price}  |  Seller: {seller}  |  Rating: {rating}  \n"
                        f"{chips}"
                    )
                    if isinstance(link, str) and link:
                        st.link_button("Open", link)
                    else:
                        st.caption("No direct link available for this item.")
                    st.markdown("---")

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.write("👆 Enter a search term and click **Search** to see results. Try **Mock data** first to tweak the UI.")

st.markdown("---")
st.caption("AC enforced: banned Google columns are removed from both the table and preview. Click items via Product Link.")
