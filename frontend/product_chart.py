from __future__ import annotations

import textwrap
from typing import Tuple
import pandas as pd
import plotly.express as px
import streamlit as st


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
        # caller should handle empty separately, but keep behavior explicit
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

    # Short label for x-axis, retain full title for tooltip
    chart_df["product_label"] = chart_df["canonical_title"].apply(
        lambda s: textwrap.shorten(str(s), width=40, placeholder="…")
    )

    # Nice numeric formatting is left to Plotly hover templates
    return chart_df


def summarize_chart(chart_df: pd.DataFrame) -> str:
    """
    Build a short textual summary for accessibility and quick scanning.
    """
    if chart_df is None or chart_df.empty:
        return "No products to summarize yet."

    n_products = chart_df["canonical_title"].nunique()
    n_vendors = chart_df["vendor"].nunique()
    lowest_price = float(chart_df["min_price"].min())

    return (
        f"Showing {n_products} products across {n_vendors} vendors; "
        f"lowest observed price: ${lowest_price:,.2f}."
    )


def render_product_chart(df: pd.DataFrame, max_products: int = 20) -> None:
    """
    Streamlit chart renderer.

    Responsibilities:
    - Handle loading / empty / error states
    - Render grouped bar chart with vendor colors
    - Provide basic accessibility (title + textual summary)
    """
    st.subheader("Price & Listings Chart")

    # Empty / loading state
    if df is None:
        st.info("Chart will appear here after you run a search.")
        return

    if df.empty:
        st.info("No results to visualize yet. Try adjusting your search or vendor filters.")
        return

    try:
        chart_df = build_product_chart_data(df, max_products=max_products)
    except Exception as exc:
        # Error state: data issues or unexpected schema
        st.error("We couldn't render the chart for these results.")
        st.caption("If this persists, please report it along with your search parameters.")
        # Optional: surface details in logs only, not UI
        st.debug(f"Chart error: {exc}")
        return

    if chart_df.empty:
        st.info("No results to visualize after filtering.")
        return

    # Accessible wrapper around the chart
    st.markdown(
        """
        <div role="img"
             aria-label="Bar chart showing average price per vendor for each product.">
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

    # Make it reasonably responsive / readable
    fig.update_layout(
        margin=dict(t=10, r=10, b=80, l=60),
        legend_title_text="Vendor",
    )
    fig.update_xaxes(tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Textual summary for accessibility & quick context
    st.caption(summarize_chart(chart_df))