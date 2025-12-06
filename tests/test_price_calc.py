# tests/test_req034_profit.py
#
# REQ-034 — Vendor Fee & Reselling Cost Calculations
# This file intentionally contains exactly three tests:
# - One UNIT test
# - One SYSTEM / INTEGRATION test
# - One REGRESSION test

import pandas as pd

from backend.pricing.fees import compute_profit
from backend.chart_adapter import df_to_chart_points


def test_unit_compute_profit_uses_fee_rules_and_costs():
    """
    UNIT (REQ-034):
    Validate that compute_profit applies costs + fees and returns a consistent net profit.
    """
    est_costs, platform_fees, net_profit, roi_pct = compute_profit(
        sale_price=50.0,
        vendor_name="eBay",
        shipping_cost=4.0,
        packaging_cost=1.0,
        other_costs=0.0,
    )

    # base expectations
    assert est_costs == 5.0  # 4 + 1
    assert platform_fees > 0.0

    # formula check: net = sale - est_costs - platform_fees
    expected_net = round(50.0 - est_costs - platform_fees, 2)
    assert net_profit == expected_net

    # ROI should be numeric when costs > 0
    assert roi_pct is not None
    assert isinstance(roi_pct, float)


def test_system_df_to_chart_points_includes_profit_fields():
    """
    SYSTEM / INTEGRATION (REQ-034 + REQ-033):
    DataFrame -> adapter -> ChartPointDict, ensuring profit metrics are injected.
    """
    df = pd.DataFrame(
        [
            {
                "canonical_title": "Test Item",
                "vendor_name": "Amazon",
                "avg_price": 40.0,
                "min_price": 38.0,
                "max_price": 42.0,
                "listing_count": 3,
                "currency": "USD",
                "shipping_cost": 5.0,
                "packaging_cost": 1.0,
                "other_costs": 0.0,
            }
        ]
    )

    points = df_to_chart_points(df)
    assert len(points) == 1

    p = points[0]

    # core shape
    assert p["label"] == "Test Item"
    assert p["vendor"].lower().startswith("amazon")

    # profit fields from REQ-034
    assert "est_cost" in p
    assert "platform_fees" in p
    assert "net_profit" in p

    # numeric sanity checks
    assert p["est_cost"] > 0.0
    assert p["platform_fees"] >= 0.0

    expected_net = round(p["avg_price"] - p["est_cost"] - p["platform_fees"], 2)
    assert p["net_profit"] == expected_net


def test_regression_legacy_rows_without_cost_fields_still_work():
    """
    REGRESSION (REQ-033 + REQ-034):
    Ensure that older result rows (no shipping/packaging/other columns)
    still produce valid chart points and do not break when profit logic is present.
    """
    df = pd.DataFrame(
        [
            {
                "canonical_title": "Legacy Item",
                "vendor_name": "Unknown",
                "avg_price": 20.0,
                "min_price": 19.0,
                "max_price": 21.0,
                "listing_count": 2,
                "currency": "USD",
                # no shipping_cost / packaging_cost / other_costs
            }
        ]
    )

    points = df_to_chart_points(df)
    assert len(points) == 1
    p = points[0]

    # previous fields still present
    assert p["label"] == "Legacy Item"
    assert "avg_price" in p

    # profit fields exist and safely default
    assert "est_cost" in p
    assert "platform_fees" in p
    assert "net_profit" in p

    # profit-related values should not be NaN
    assert p["est_cost"] >= 0.0
    assert p["platform_fees"] >= 0.0
    assert isinstance(p["net_profit"], float)
