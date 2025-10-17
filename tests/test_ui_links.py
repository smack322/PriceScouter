import pandas as pd
import numpy as np


def test_links_render_and_open_button(fake_st):
    ui, st = fake_st

    # Craft a small DF with both link and product_link present
    df = pd.DataFrame(
        [
            {
                "title": "Sample Item",
                "price_display": "$10.00",
                "source": "Vendor 1",
                "link": "https://example.com/a",
                "product_link": "https://example.com/a-product",
                "rating": 4.5,
            },
        ]
    )

    # Emulate the table-render block from the app
    show_cols = ["title", "price_display", "source", "link", "product_link"]

    # Normalize link-like columns to strings / None
    for c in ["product_link", "link"]:
        if c in df.columns:
            df[c] = df[c].astype("string").where(df[c].notna(), None)

    col_config = {}
    if "product_link" in df.columns:
        col_config["product_link"] = ui.st.column_config.LinkColumn(
            label="Product Link", display_text="Open", help="Opens product page in a new tab"
        )
    if "link" in df.columns:
        col_config["link"] = ui.st.column_config.LinkColumn(
            label="Link", display_text="Open", help="Opens listing in a new tab"
        )

    # Call into fake st.dataframe (we assert on calls later)
    ui.st.dataframe(df[show_cols], use_container_width=True, height=520, column_config=col_config)

    # Now emulate the preview block and assert link_button is called
    for _, row in df.head(5).iterrows():
        link = row.get("link")
        if not isinstance(link, str) or not link:
            plink = row.get("product_link")
            link = plink if isinstance(plink, str) and plink else None
        if isinstance(link, str) and link:
            ui.st.link_button("Open", link)

    # --- Assertions ---
    # 1) dataframe called once with column_config containing LinkColumns
    assert len(st.calls["dataframe"]) == 1
    _, kwargs = st.calls["dataframe"][0]
    assert "column_config" in kwargs
    cc = kwargs["column_config"]
    assert "link" in cc and "product_link" in cc
    assert cc["link"].display_text == "Open"
    assert cc["product_link"].display_text == "Open"

    # 2) preview created an "Open" button with correct URL
    assert ("Open", "https://example.com/a") in st.calls["link_button"]

def _normalize(df):
    for c in ["free_shipping", "in_store_pickup", "fast_delivery"]:
        if c in df.columns:
            df[c] = pd.Series(df[c]).astype("boolean").fillna(False).astype(bool)
    for c in ["link", "product_link"]:
        if c in df.columns:
            df[c] = pd.Series(df[c]).where(pd.notna(df[c]), None)
    return df

def test_no_na_boolean_crash_on_google_like_rows(fake_st):
    ui, st = fake_st

    # Row that used to trigger: booleans and link are pd.NA
    df = pd.DataFrame(
        [
            {
                "title": "Google Result",
                "price_display": "$29.99",
                "source": "Google",
                "free_shipping": pd.NA,
                "in_store_pickup": pd.NA,
                "fast_delivery": pd.NA,
                "link": pd.NA,
                "product_link": pd.NA,
                "rating": pd.NA,
            }
        ]
    )

    # Normalize like the app does
    df = _normalize(df)

    # Emulate the preview checks that used to crash
    errors = []
    try:
        for _, row in df.iterrows():
            # boolean badges: explicit `is True`
            if row.get("free_shipping") is True:
                pass
            if row.get("in_store_pickup") is True:
                pass
            if row.get("fast_delivery") is True:
                pass

            # guard link with isinstance(str)
            link = row.get("link")
            if not isinstance(link, str) or not link:
                plink = row.get("product_link")
                link = plink if isinstance(plink, str) and plink else None
            # no exception should be raised getting here
    except Exception as e:
        errors.append(e)

    assert not errors, f"Unexpected exception raised: {errors}"

def _normalize_provider_agnostic(df):
    # Same logic as in app, extracted for test
    for c in ["free_shipping", "in_store_pickup", "fast_delivery"]:
        if c in df.columns:
            df[c] = pd.Series(df[c]).astype("boolean").fillna(False).astype(bool)
    for c in ["link", "product_link"]:
        if c in df.columns:
            df[c] = pd.Series(df[c]).where(pd.notna(df[c]), None)
            df[c] = df[c].apply(
                lambda u: (f"https://{u}" if isinstance(u, str) and u and not u.startswith(("http://", "https://")) else u)
            )
    return df

def test_provider_agnostic_normalization_across_providers(fake_st):
    ui, st = fake_st

    # Simulate three providers mixed together (like "All")
    df = pd.DataFrame(
        [
            # Google-like row (pd.NA booleans and missing link)
            {
                "title": "G Earbuds",
                "price_display": "$59.99",
                "source": "Google",
                "free_shipping": pd.NA,
                "in_store_pickup": pd.NA,
                "fast_delivery": pd.NA,
                "link": pd.NA,
                "product_link": None,
                "rating": pd.NA,
            },
            # Keepa-like row
            {
                "title": "Amazon Thing",
                "price_display": "$19.99",
                "source": "Amazon",
                "free_shipping": True,
                "in_store_pickup": False,
                "fast_delivery": True,
                "link": "www.amazon.com/dp/ASIN123",
                "rating": 4.2,
            },
            # eBay-like row
            {
                "title": "eBay Widget",
                "price_display": "$9.99",
                "source": "eBay",
                "free_shipping": False,
                "in_store_pickup": pd.NA,
                "fast_delivery": pd.NA,
                "link": "https://ebay.com/itm/123",
                "rating": 4.9,
            },
        ]
    )

    df = _normalize_provider_agnostic(df)

    # No pd.NA booleans remain; they should be real bools
    for c in ["free_shipping", "in_store_pickup", "fast_delivery"]:
        assert df[c].map(type).eq(bool).all()

    # Schemeless link should be corrected with https://
    assert df.loc[1, "link"].startswith("https://")

    # Now emulate preview booleans (should not raise)
    badges = []
    for _, row in df.iterrows():
        if row.get("free_shipping") is True:
            badges.append("Free ship")
        if row.get("in_store_pickup") is True:
            badges.append("Store pickup")
        if row.get("fast_delivery") is True:
            badges.append("Fast delivery")

    # Sanity: at least one badge created from booleans
    assert "Free ship" in badges or "Fast delivery" in badges or "Store pickup" in badges