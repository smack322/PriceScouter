# app.py
# Streamlit mock UI for product search with optional SerpApi integration
# pip install streamlit google-search-results pandas python-dotenv

import os
import time
import pandas as pd
import asyncio
import streamlit as st
from dotenv import load_dotenv
from agents.client import run_agents

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except Exception:
    SERPAPI_AVAILABLE = False

load_dotenv()

st.set_page_config(page_title="PriceScouter – Mock Search", page_icon="🛒", layout="wide")

# ----------------------------
# Sidebar controls / settings
# ----------------------------
st.sidebar.title("Settings")
provider = st.sidebar.selectbox(
    "Data source",
    ["Mock data", "SerpApi – Google Shopping", "All"],
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

st.sidebar.caption(
    "Tip: Leave provider as **Mock data** while designing the UI. "
    "Switch to SerpApi when you're ready to test live calls."
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def mock_search(q: str, n: int = 20) -> pd.DataFrame:
    # Simple, deterministic mock dataset
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

@st.cache_data(show_spinner=False)
def serpapi_search(q: str, key: str, loc: str | None, n: int) -> pd.DataFrame:
    if not SERPAPI_AVAILABLE:
        raise RuntimeError("Package 'google-search-results' not installed.")
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
    for r in results.get("shopping_results", []):
        items.append(
            {
                "title": r.get("title"),
                "price": r.get("price"),
                "source": r.get("source"),
                "rating": r.get("rating"),
                "link": r.get("link"),
                "product_link": r.get("product_link"),
            }
        )
    return pd.DataFrame(items)

def normalize_price_str_to_float(x: str | None) -> float | None:
    if not x:
        return None
    # handles "$19.99", "$19.99 to $24.99", "$19.99 + tax", etc. (take the first number)
    import re
    m = re.search(r"\d+(?:\.\d+)?", x.replace(",", ""))
    return float(m.group()) if m else None

# ----------------------------
# Header / Search bar
# ----------------------------
st.title("🛒 PriceScouter – Product Search (Mock UI)")
st.write("Type a query, choose a data source, and view normalized results.")

with st.form("search_form"):
    q = st.text_input("Search for a product", value="iphone 15 case", placeholder="e.g., 'wireless earbuds'")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        sort_by = st.selectbox("Sort by", ["Best match", "Price (asc)", "Price (desc)", "Rating (desc)"])
    with col2:
        min_price = st.number_input("Min $", min_value=0.0, value=0.0, step=1.0)
    with col3:
        max_price = st.number_input("Max $", min_value=0.0, value=0.0, step=1.0, help="0 = no max")
    submitted = st.form_submit_button("Search", use_container_width=True)

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

            elif provider == "All":  # Agents (Keepa + Google + eBay)
                # 🔽 This is the line you asked about — it goes right here:
                rows = asyncio.run(
                    run_agents(
                        q,
                        zip_code=st.sidebar.text_input("Ship to ZIP", value="19406", key="zip_agents"),
                        country="US",  # or expose a sidebar input like you do for SerpApi
                        max_price=(max_price or None),
                        top_n=max_results,
                    )
                )
                # Convert to DataFrame; coerce any odd types to builtins
                import json
                import numpy as np
                def _to_builtin(o):
                    if isinstance(o, (np.integer,)): return int(o)
                    if isinstance(o, (np.floating,)): return float(o)
                    if isinstance(o, (np.bool_,)): return bool(o)
                    if isinstance(o, (np.ndarray,)): return o.tolist()
                    return o
                rows = json.loads(json.dumps(rows, default=_to_builtin))
                df = pd.DataFrame(rows) if rows else pd.DataFrame([])

                # Normalize a 'price' string column for display parity with other providers
                if "total" in df.columns and "price" not in df.columns:
                    df["price"] = df["total"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else None)
                if "url" in df.columns and "link" not in df.columns:
                    df["link"] = df["url"]

            # --- from here down, keep your existing normalization/sorting/display ---
            df["price_value"] = df["price"].apply(normalize_price_str_to_float) if "price" in df.columns else None

            if min_price > 0:
                df = df[df["price_value"].fillna(10**9) >= min_price]
            if max_price > 0:
                df = df[df["price_value"].fillna(0) <= max_price]

            if sort_by == "Price (asc)":
                df = df.sort_values(by=["price_value", "title"], ascending=[True, True], na_position="last")
            elif sort_by == "Price (desc)":
                df = df.sort_values(by=["price_value", "title"], ascending=[False, True], na_position="last")
            elif sort_by == "Rating (desc)" and "rating" in df.columns:
                df = df.sort_values(by=["rating", "title"], ascending=[False, True], na_position="last")

            show_cols = [c for c in ["title", "price", "source", "merchant", "seller",
                                    "rating", "link", "product_link", "_source"] if c in df.columns]
            st.caption(f"Found {len(df)} item(s)")
            st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, height=480)

            with st.expander("Preview top 5"):
                for _, row in df.head(5).iterrows():
                    title = row.get("title", "")
                    price = row.get("price", "")
                    src   = row.get("source") or row.get("merchant") or row.get("seller") or row.get("_source", "")
                    rating = row.get("rating", "—")
                    link = row.get("link") or row.get("product_link")
                    st.markdown(
                        f"**{title}**  \n"
                        f"Price: {price}  |  Source: {src}  |  Rating: {rating}  \n"
                        f"{('[Open](' + str(link) + ')') if link else ''}"
                    )
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.write("👆 Enter a search term and click **Search** to see results. Use **Mock data** first to tweak the UI.")

st.markdown("---")
st.caption("This is a mock UI. For production, add more sources (eBay Browse, Walmart, etc.) and unify fields.")

