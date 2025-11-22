# tests/test_chart_adapter.py
import math
import pandas as pd

from backend.chart_adapter import df_to_chart_points
from backend.chart_types import ChartPointDict


def test_chart_mapper_empty_df_returns_empty_list():
    df = pd.DataFrame()
    result = df_to_chart_points(df)
    assert result == []


def test_chart_mapper_single_listing_basic_values():
    df = pd.DataFrame(
        [
            {
                "canonical_title": "iPhone 15 Case",
                "vendor_name": "Amazon",
                "min_price": 10.0,
                "avg_price": 12.5,
                "max_price": 15.0,
                "listing_count": 3,
                "currency": "USD",
            }
        ]
    )

    result = df_to_chart_points(df)
    assert len(result) == 1
    row: ChartPointDict = result[0]
    assert row["label"] == "iPhone 15 Case"
    assert row["vendor"] == "Amazon"
    assert row["min_price"] == 10.0
    assert row["avg_price"] == 12.5
    assert row["max_price"] == 15.0
    assert row["listing_count"] == 3
    assert row["currency"] == "USD"


def test_chart_mapper_handles_missing_prices_and_nans():
    df = pd.DataFrame(
        [
            {
                "canonical_title": "Unknown Price Product",
                "vendor_name": "eBay",
                "min_price": None,
                "avg_price": math.nan,
                "max_price": None,
                "listing_count": 1,
            }
        ]
    )

    result = df_to_chart_points(df)
    row = result[0]
    assert row["min_price"] is None
    assert row["avg_price"] is None
    assert row["max_price"] is None
    assert row["listing_count"] == 1


def test_chart_data_backward_compatible_keys():
    """
    Ensure that the core keys always exist even if new optional fields are added.
    """
    df = pd.DataFrame(
        [
            {
                "canonical_title": "Test Product",
                "vendor_name": "Walmart",
                "min_price": 5.0,
                "avg_price": 7.5,
                "max_price": 9.0,
                "listing_count": 2,
                "currency": "USD",
                "canonical_id": 123,
            }
        ]
    )

    result = df_to_chart_points(df)
    row = result[0]

    # Core keys required by the chart
    for key in ["label", "vendor", "min_price", "avg_price", "max_price", "listing_count"]:
        assert key in row

    # Optional keys may or may not be present, but must not break existing code
    # (We don't assert on them – just ensure accessing core keys still works.)
