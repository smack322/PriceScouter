# frontend/ui_results.py
import pandas as pd
import streamlit as st
import sys
import pathlib

# --- ensure project root is on sys.path ---
ROOT = pathlib.Path(__file__).resolve().parents[1]  # parent directory that has /backend and /frontend
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from backend.queries import fetch_canonicals, fetch_variants
from frontend.product_chart import render_product_chart
from frontend.product_filtering import apply_product_filters

from frontend.product_chart import render_product_chart
from backend.chart_adapter import df_to_chart_points

def _money(x):
    return "" if x is None or pd.isna(x) else f"${x:,.2f}"

def _link(text, url):
    return f"[{text}]({url})" if url else text


def render_results():
    st.subheader("Product Results (Canonical)")

    # --- Controls row ---
    c1, c2, c3 = st.columns([2, 1.3, 1.7])

    with c1:
        search_text = st.text_input(
            "Filter products",
            "",
            placeholder="e.g., iPhone 15 case",
        )

    with c2:
        limit = st.number_input("Max rows", 10, 1000, 200, step=10)

    df = fetch_canonicals(limit=limit, q=q if q else None)
    if df.empty:
        st.info("No results yet. Try another search or run a scrape.")
        return

    show = df.assign(
        Min=df["min_price"].map(_money),
        Avg=df["avg_price"].map(_money),
        Max=df["max_price"].map(_money),
    )[["title","Min","Avg","Max","seller_count","total_listings"]]
    show.columns = ["Title","Min","Avg","Max","Sellers","Listings"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("### Details")
    for _, row in df.iterrows():
        header = f"{row.title} — {_money(row.avg_price)} avg • {int(row.seller_count)} sellers • {int(row.total_listings)} listings"
        with st.expander(header):
            variants = fetch_variants(canonical_key=row.canonical_key)
            if not variants:
                st.write("No variant listings.")
                continue
            vdf = pd.DataFrame(variants)
            vdf["Buy"] = vdf.apply(lambda r: _link(r.get("seller") or "Open", r.get("product_url")), axis=1)
            # Select nice display columns if they exist
            cols = [c for c in ["Buy","listing_title","price","shipping","condition","brand","currency","source"] if c in vdf.columns]
            if "price" in vdf.columns:
                vdf["price"] = vdf["price"].map(_money)
            st.dataframe(vdf[cols], use_container_width=True, hide_index=True)
    st.markdown("### Pricing & Listing Patterns")
    render_product_chart(df)

def render_results_table(results_df: pd.DataFrame) -> None:
    chart_points = df_to_chart_points(results_df)

    if not chart_points:
        st.info("No results found.")
        return

    # Convert to DataFrame for display
    table_df = pd.DataFrame(chart_points)

    # Choose the columns you actually want to show in the UI
    display_cols = [
        "label",
        "vendor",
        "avg_price",
        "est_cost",
        "platform_fees",
        "net_profit",
        "roi_pct",
    ]

    available_cols = [c for c in display_cols if c in table_df.columns]

    st.subheader("Product Results (with Profit Estimates)")
    st.dataframe(
        table_df[available_cols].rename(
            columns={
                "label": "Product",
                "vendor": "Vendor",
                "avg_price": "Sale $",
                "est_cost": "Est. Cost $",
                "platform_fees": "Fees $",
                "net_profit": "Net Profit $",
                "roi_pct": "ROI %",
            }
        ),
        use_container_width=True,
    )