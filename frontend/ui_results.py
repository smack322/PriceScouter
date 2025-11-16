# frontend/ui_results.py
import pandas as pd
import streamlit as st
from backend.queries import fetch_canonicals, fetch_variants
from frontend.components.product_chart import render_product_chart
from frontend.components.product_filtering import apply_product_filters

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

    # Load base data once
    try:
        df_all = fetch_canonicals(limit=limit, q=None)  # q handled client-side here
    except Exception:
        st.error("Unable to load product results.")
        df_all = pd.DataFrame()

    if df_all.empty:
        st.info("No canonical products found yet. Try running a search.")
        df_filtered = df_all
    else:
        # Derive vendor + price ranges for UI filters
        all_vendors = sorted(df_all["vendor"].dropna().unique().tolist())
        min_price = float(df_all["total_price"].min())
        max_price = float(df_all["total_price"].max())

        with c2:
            selected_vendors = st.multiselect(
                "Vendors",
                options=all_vendors,
                default=all_vendors,
            )

        with c3:
            price_low, price_high = st.slider(
                "Total price range",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
            )

        # Apply filters to get the view dataset
        df_filtered = apply_product_filters(
            df_all,
            search_text=search_text,
            vendors=selected_vendors,
            price_range=(price_low, price_high),
        )

        if df_filtered.empty:
            st.info("No products match your current filters.")
        else:
            st.dataframe(df_filtered)

    # --- Product selection for focused chart view ---
    selected_canonical_title = None
    if not df_filtered.empty:
        product_titles = sorted(df_filtered["canonical_title"].dropna().unique().tolist())
        focus_label = st.selectbox(
            "Focus chart on product (optional)",
            options=["(All products)"] + product_titles,
            index=0,
            help="Choose a product to see a detailed vendor breakdown, "
                 "or leave as '(All products)' for an aggregate view.",
        )
        if focus_label != "(All products)":
            selected_canonical_title = focus_label

    st.divider()

    st.markdown("### Pricing & Listing Patterns")
    render_product_chart(df_filtered, selected_canonical_title=selected_canonical_title)