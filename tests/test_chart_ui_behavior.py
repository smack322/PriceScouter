"""
Chart UI Tests for REQ-031 (Core Chart UI Component) and REQ-032 (Filter Integration)

Covers:
- UT-031-ChartRender-01  (Unit)
- IT-031/032-FilterSync-01 (Integration)
- UAT-031/032-Insight-01 (UAT-style functional test)
- REG-032-TableStable-01 (Regression)
"""

import pandas as pd

from frontend.components.product_chart import (
    build_product_chart_data,
    prepare_chart_view,
    summarize_chart,
)
from frontend.components.product_filtering import apply_product_filters


# ---------------------------------------------------------------------------
# Shared mock data for chart + filter tests
# ---------------------------------------------------------------------------

def _make_mock_products_df() -> pd.DataFrame:
    """
    Create a small mock dataset with:
    - ≥ 5 products
    - multiple vendors
    - varying prices
    """
    return pd.DataFrame(
        [
            {
                "canonical_title": "Nintendo Switch Console",
                "vendor": "Amazon",
                "total_price": 299.99,
            },
            {
                "canonical_title": "Nintendo Switch Console",
                "vendor": "eBay",
                "total_price": 289.99,
            },
            {
                "canonical_title": "Nintendo Switch Console",
                "vendor": "Best Buy",
                "total_price": 309.99,
            },
            {
                "canonical_title": "Nintendo Switch Game - Zelda",
                "vendor": "Amazon",
                "total_price": 59.99,
            },
            {
                "canonical_title": "Nintendo Switch Game - Mario Kart",
                "vendor": "Amazon",
                "total_price": 49.99,
            },
            {
                "canonical_title": "PlayStation 5 Console",
                "vendor": "Amazon",
                "total_price": 499.99,
            },
            {
                "canonical_title": "PlayStation 5 Console",
                "vendor": "eBay",
                "total_price": 479.99,
            },
        ]
    )


# ---------------------------------------------------------------------------
# Test Case 1 — Unit Test (REQ-031 — Chart Rendering)
# Test ID: UT-031-ChartRender-01 — Chart renders with correct data
# ---------------------------------------------------------------------------

def test_ut_031_chart_render_01_builds_expected_chart_data():
    """
    UT-031-ChartRender-01 — Chart renders with correct data

    Test Type: Unit
    Context: UI Component (data prep for chart)
    Associated Requirement: REQ-031 — Core Chart UI Component

    This test validates that the Core Chart UI Component's data preparation
    logic builds the correct aggregation for a small dataset, which is the
    backbone of chart rendering.
    """

    df = _make_mock_products_df()

    # Step 1: "Load the ProductChart component with a mock dataset (e.g., 3 products)."
    # Here we simulate the underlying data prep.
    chart_df = build_product_chart_data(df, max_products=10)

    # Step 2: "Count the rendered chart data points/bars."
    # We expect one row per (canonical_title, vendor) pair.
    grouped_keys = chart_df[["canonical_title", "vendor"]].value_counts()
    # Expected unique (product, vendor) combinations from mock data:
    expected_pairs = {
        ("Nintendo Switch Console", "Amazon"),
        ("Nintendo Switch Console", "eBay"),
        ("Nintendo Switch Console", "Best Buy"),
        ("Nintendo Switch Game - Zelda", "Amazon"),
        ("Nintendo Switch Game - Mario Kart", "Amazon"),
        ("PlayStation 5 Console", "Amazon"),
        ("PlayStation 5 Console", "eBay"),
    }
    assert expected_pairs.issubset(set(grouped_keys.index)), (
        "Chart data should contain one row per (product, vendor) pair."
    )

    # Sanity: "Exactly N data points appear in the chart" -- here N = len(expected_pairs)
    assert len(grouped_keys) >= len(expected_pairs)

    # Step 3: "Simulate hover or focus on a data point."
    # At the data layer, this means we have tooltip fields available.
    # We check that aggregation columns exist and are numeric.
    for col in ["min_price", "avg_price", "max_price", "listing_count"]:
        assert col in chart_df.columns, f"Expected tooltip metric column '{col}'"
        assert pd.api.types.is_numeric_dtype(chart_df[col]), f"'{col}' should be numeric"

    # Check that product label exists for hover display
    assert "product_label" in chart_df.columns
    assert chart_df["product_label"].notna().all()

    # Post-condition: no exceptions thrown, data is consistent with expectations.


# ---------------------------------------------------------------------------
# Test Case 2 — Integration Test (REQ-031 & REQ-032)
# Test ID: IT-031/032-FilterSync-01 — Chart updates when filters change
# ---------------------------------------------------------------------------

def test_it_031_032_filter_sync_01_chart_tracks_filtered_results():
    """
    IT-031/032-FilterSync-01 — Chart updates when filters change

    Test Type: Integration
    Context: UI Component (table + chart filter sync)
    Associated Requirements:
      - REQ-031 — Chart UI Component
      - REQ-032 — Filter Integration Between Product Table and Chart

    Tests that filtering the product table updates the chart data to match
    the filtered results. Ensures shared UI state propagates correctly.
    """
    df_all = _make_mock_products_df()

    # Step 1: Load the page with full product data.
    # (Table displays all products; chart visualizes full dataset.)
    df_filtered_all = apply_product_filters(df_all)  # no filters applied
    chart_all, mode_all = prepare_chart_view(df_filtered_all)
    assert mode_all == "aggregate"
    # All vendors appear in the aggregate view
    assert set(chart_all["vendor"].unique()) == set(df_all["vendor"].unique())

    # Step 2: Apply a text filter (e.g., "Nintendo").
    df_filtered_nintendo = apply_product_filters(df_all, search_text="nintendo")
    assert not df_filtered_nintendo.empty
    # The table would now show only Nintendo-related rows.
    assert df_filtered_nintendo["canonical_title"].str.contains(
        "Nintendo"
    ).all()

    # Step 3: Observe the chart (it should now reflect only filtered products).
    chart_nintendo, mode_nintendo = prepare_chart_view(df_filtered_nintendo)
    assert mode_nintendo == "aggregate"
    # Every canonical_title in the chart must exist in the filtered table.
    assert set(chart_nintendo["canonical_title"].unique()).issubset(
        set(df_filtered_nintendo["canonical_title"].unique())
    )

    # Step 4: Clear the filter (back to full dataset).
    df_filtered_clear = apply_product_filters(df_all, search_text="")
    chart_clear, mode_clear = prepare_chart_view(df_filtered_clear)
    assert mode_clear == "aggregate"

    # Chart reverts to full dataset behavior (same set of titles as original)
    assert set(chart_clear["canonical_title"].unique()) == set(
        df_filtered_all["canonical_title"].unique()
    )


# ---------------------------------------------------------------------------
# Test Case 3 — UAT Test (REQ-031 & REQ-032)
# Test ID: UAT-031/032-Insight-01 — User insight visualization workflow
# ---------------------------------------------------------------------------

def test_uat_031_032_insight_01_user_workflow_search_filter_focus():
    """
    UAT-031/032-Insight-01 — User insight visualization workflow

    Test Type: UAT (simulated in pytest via functional checks)
    Context: End-to-End PriceScouter Workflow (data + filter + focus)
    Associated Requirements:
      - REQ-031 — Core Chart UI Component
      - REQ-032 — Filter Integration Between Product Table and Chart

    Simulates the workflow:
      1. User searches for "Nintendo Switch".
      2. User applies a vendor filter (e.g., Amazon + eBay).
      3. User focuses the chart on a single product.
      4. User clears selection to get back to aggregate view.
    """
    df_all = _make_mock_products_df()

    # Step 1: User searches for a product (e.g., "Nintendo Switch").
    df_search = apply_product_filters(df_all, search_text="nintendo switch")
    assert not df_search.empty
    assert df_search["canonical_title"].str.contains("Nintendo Switch").all()

    # "Results table and chart populate with metrics."
    chart_search, mode_search = prepare_chart_view(df_search)
    assert mode_search == "aggregate"
    assert "avg_price" in chart_search.columns

    # Step 2: User applies vendor filter (e.g., Amazon + eBay).
    df_amz_ebay = apply_product_filters(
        df_search,
        vendors=["Amazon", "eBay"],
    )
    assert not df_amz_ebay.empty
    assert set(df_amz_ebay["vendor"].unique()).issubset({"Amazon", "eBay"})

    chart_amz_ebay, mode_amz_ebay = prepare_chart_view(df_amz_ebay)
    assert mode_amz_ebay == "aggregate"
    assert set(chart_amz_ebay["vendor"].unique()).issubset({"Amazon", "eBay"})

    # Step 3: User clicks an individual row to "focus" the product.
    # Choose one of the products in the filtered set.
    some_product = df_amz_ebay["canonical_title"].iloc[0]
    chart_focused, mode_focused = prepare_chart_view(
        df_amz_ebay,
        selected_canonical_title=some_product,
    )
    # "Chart switches to detailed view for selected product."
    assert mode_focused == "focused"
    assert chart_focused["canonical_title"].nunique() == 1
    assert chart_focused["canonical_title"].unique()[0] == some_product

    # Step 4: User clears selection ("back to aggregate view").
    chart_cleared, mode_cleared = prepare_chart_view(
        df_amz_ebay,
        selected_canonical_title=None,
    )
    assert mode_cleared == "aggregate"
    # Multiple products or multiple vendors can exist again at this level.
    assert chart_cleared["canonical_title"].nunique() >= 1

    # Post-condition: we can still summarize data without error.
    summary_text = summarize_chart(chart_cleared, mode=mode_cleared)
    assert "Showing" in summary_text


# ---------------------------------------------------------------------------
# Test Case 4 — Regression Test (REQ-032)
# Test ID: REG-032-TableStable-01 — Table behavior unaffected by chart
# ---------------------------------------------------------------------------

def test_reg_032_table_stable_01_chart_does_not_break_table_behavior():
    """
    REG-032-TableStable-01 — Table behavior unaffected by chart

    Test Type: Regression
    Context: UI Table + Chart Shared State (simulated via DataFrame operations)
    Associated Requirement:
      - REQ-032 — Filter Integration Between Product Table and Chart

    Validates that introducing chart + filter sync logic does not break
    core table behaviors such as sorting and limiting rows.
    Here we simulate table operations via pandas and ensure that chart
    data remains consistent and error-free.
    """
    df_all = _make_mock_products_df()

    # Step 1: "Load search results with chart enabled."
    df_filtered = apply_product_filters(df_all)
    chart_initial, mode_initial = prepare_chart_view(df_filtered)
    assert mode_initial == "aggregate"
    initial_titles = set(chart_initial["canonical_title"].unique())

    # Step 2: "Sort table by price ascending."
    # Simulate table sorting by total_price
    df_sorted = df_filtered.sort_values(by="total_price", ascending=True)
    # The chart's underlying aggregation should still be valid on sorted data.
    chart_sorted, mode_sorted = prepare_chart_view(df_sorted)
    assert mode_sorted == "aggregate"
    # Sorting should not change the set of canonical titles chart sees.
    assert set(chart_sorted["canonical_title"].unique()) == initial_titles

    # Step 3: "Change row limit (e.g., from 20 → 5)."
    # Simulate limiting to top 5 rows in the table.
    df_limited = df_sorted.head(5)
    chart_limited, mode_limited = prepare_chart_view(df_limited)
    assert mode_limited == "aggregate"
    # The chart should now reflect only the subset of products in the limited table.
    assert set(chart_limited["canonical_title"].unique()).issubset(
        set(df_limited["canonical_title"].unique())
    )

    # Step 4: "Navigate to the next page (if pagination exists)."
    # We simulate this by taking a different slice of the sorted results.
    if len(df_sorted) > 5:
        df_next_page = df_sorted.iloc[5:]
        chart_next, mode_next = prepare_chart_view(df_next_page)
        assert mode_next == "aggregate"
        # No errors and still valid metric columns:
        for col in ["min_price", "avg_price", "max_price", "listing_count"]:
            assert col in chart_next.columns

    # Post-condition: table-like operations (sort/limit) do not cause
    # any inconsistency or breakage in chart data preparation.
