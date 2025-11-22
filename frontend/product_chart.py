# frontend/components/product_chart.py

from __future__ import annotations

import textwrap
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.queries import fetch_chart_data_for_search
from backend.telemetry import log_chart_event


REQUIRED_COLS = {"canonical_title", "vendor", "total_price"}


def build_product_chart_data(
    df: pd.DataFrame, max_products: int = 20
) -> pd.DataFrame:
    """
    Normalize raw product results for charting.

    - Groups by (canonical_title, vendor)
    - Computes min/avg/max price and listing count
    - Keeps only top-N products by listing_count for readability
    """
    if df is None:
        raise ValueError("DataFrame is None")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for chart: {sorted(missing)}")

    if df.empty:
        return df.copy()

    grouped = (
        df.groupby(["canonical_title", "vendor"])["total_price"]
        .agg(
            min_price="min",
            avg_price="mean",
            max_price="max",
            listing_count="count",
        )
        .reset_index()
    )

    # Pick top products by total listing count so labels don’t overlap
    product_totals = (
        grouped.groupby("canonical_title")["listing_count"]
        .sum()
        .sort_values(ascending=False)
    )
    top_product_names = product_totals.head(max_products).index

    chart_df = grouped[grouped["canonical_title"].isin(top_product_names)].copy()

    chart_df["product_label"] = chart_df["canonical_title"].apply(
        lambda s: textwrap.shorten(str(s), width=40, placeholder="…")
    )

    return chart_df


def prepare_chart_view(
    df: pd.DataFrame,
    selected_canonical_title: Optional[str] = None,
    max_products: int = 20,
) -> Tuple[pd.DataFrame, str]:
    """
    Return (chart_df, mode) where mode is either 'aggregate' or 'focused'.

    - Aggregate: top-N products from the filtered dataset
    - Focused: vendor breakdown for a single selected product
    """
    chart_df = build_product_chart_data(df, max_products=max_products)

    if not selected_canonical_title:
        return chart_df, "aggregate"

    focused = chart_df[chart_df["canonical_title"] == selected_canonical_title]
    if focused.empty:
        # Fallback: selection not present in filtered set → use aggregate
        return chart_df, "aggregate"

    # Focus mode: only the selected product, vendor vs avg_price
    focused = focused.copy()
    # Use a single, short label; they’re all the same product anyway
    focused["product_label"] = focused["canonical_title"].apply(
        lambda s: textwrap.shorten(str(s), width=40, placeholder="…")
    )
    return focused, "focused"


def summarize_chart(
    chart_df: pd.DataFrame,
    mode: str = "aggregate",
    selected_canonical_title: Optional[str] = None,
) -> str:
    """
    Build a short textual summary for accessibility and quick scanning.
    """
    if chart_df is None or chart_df.empty:
        return "No products to summarize yet."

    if mode == "focused" and selected_canonical_title:
        n_vendors = chart_df["vendor"].nunique()
        low = float(chart_df["min_price"].min())
        high = float(chart_df["max_price"].max())
        return (
            f"Showing vendor breakdown for '{selected_canonical_title}' "
            f"across {n_vendors} vendors; price range: ${low:,.2f}–${high:,.2f}."
        )

    n_products = chart_df["canonical_title"].nunique()
    n_vendors = chart_df["vendor"].nunique()
    lowest_price = float(chart_df["min_price"].min())

    return (
        f"Showing {n_products} products across {n_vendors} vendors; "
        f"lowest observed price: ${lowest_price:,.2f}."
    )


def render_product_chart(
    df: pd.DataFrame,
    selected_canonical_title: Optional[str] = None,
    max_products: int = 20,
) -> None:
    """
    Streamlit chart renderer.

    - Receives the *filtered* product DataFrame
    - Optionally receives a selected product title
    - Chooses aggregate vs focused view automatically
    """
    st.subheader("Price & Listings Chart")

    if df is None:
        st.info("Chart will appear here after you run a search.")
        return

    if df.empty:
        st.info("No results to visualize yet. Try adjusting your search or vendor filters.")
        return

    try:
        chart_df, mode = prepare_chart_view(
            df, selected_canonical_title=selected_canonical_title, max_products=max_products
        )
    except Exception as exc:
        st.error("We couldn't render the chart for these results.")
        st.caption("If this persists, please report it along with your search parameters.")
        st.debug(f"Chart error: {exc}")
        return

    if chart_df.empty:
        st.info("No results to visualize after filtering.")
        return

    aria_label = (
        "Bar chart showing average price per vendor for the selected product."
        if mode == "focused"
        else "Bar chart showing average price per vendor across products."
    )

    st.markdown(
        f"""
        <div role="img"
             aria-label="{aria_label}">
        """,
        unsafe_allow_html=True,
    )

    fig = px.bar(
        chart_df,
        x="product_label",
        y="avg_price",
        color="vendor",
        barmode="group",
        hover_data={
            "canonical_title": True,
            "vendor": True,
            "min_price": ":.2f",
            "avg_price": ":.2f",
            "max_price": ":.2f",
            "listing_count": True,
        },
        labels={
            "product_label": "Product",
            "avg_price": "Average total price (USD)",
            "vendor": "Vendor",
        },
        title=None,
    )

    fig.update_layout(
        margin=dict(t=10, r=10, b=80, l=60),
        legend_title_text="Vendor",
    )
    fig.update_xaxes(tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        summarize_chart(
            chart_df,
            mode=mode,
            selected_canonical_title=selected_canonical_title,
        )
    )
def render_product_price_chart(search_id: int) -> None:
    chart_data = fetch_chart_data_for_search(search_id)

    if not chart_data:
        st.info("No chart data available for this search yet.")
        return

    log_chart_event("chart_rendered", search_id=search_id, count=len(chart_data))

    df = pd.DataFrame(chart_data)

    fig = px.bar(
        df,
        x="label",
        y="avg_price",
        color="vendor",
        hover_data=["min_price", "max_price", "listing_count"],
    )

    # Capture interactions via selection
    selection = st.plotly_chart(fig, use_container_width=True)

    # (Streamlit doesn’t have deep client-side hooks, so we log server-side events)
    # For example, log when the user changes the selected product label via another UI control:
    selected_label = st.selectbox("Highlight product", df["label"].unique())
    if selected_label:
        log_chart_event("product_selection_changed", search_id=search_id, label=selected_label)