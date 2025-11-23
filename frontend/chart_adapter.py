# backend/chart_adapter.py
from __future__ import annotations

from typing import Iterable, List, Optional

import math
import pandas as pd

from .chart_types import ChartPoint, ChartPointDict
from backend.fees import compute_profit

def _safe_float(val: object) -> Optional[float]:
    """Convert values to float or None, handling NaN / missing gracefully."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def df_to_chart_points(df: pd.DataFrame) -> List[ChartPointDict]:
    """
    Map product/variant/listing aggregate DataFrame into the chart data structure.

    Expected columns (minimal):
      - label / canonical_title
      - vendor / vendor_name
      - min_price
      - avg_price
      - max_price
      - listing_count

    Optional columns:
      - currency
      - canonical_id
      - variant_id
      - product_url
    """
    if df is None or df.empty:
        return []

    # Allow some column name flexibility
    label_col = "label" if "label" in df.columns else "canonical_title"
    vendor_col = "vendor" if "vendor" in df.columns else "vendor_name"

    points: List[ChartPointDict] = []

    for _, row in df.iterrows():
        cp = ChartPoint(
            label=str(row.get(label_col, "")),
            vendor=str(row.get(vendor_col, "")),
            min_price=_safe_float(row.get("min_price")),
            avg_price=_safe_float(row.get("avg_price")),
            max_price=_safe_float(row.get("max_price")),
            listing_count=int(row.get("listing_count") or 0),
            currency=str(row.get("currency") or "USD"),
            canonical_id=row.get("canonical_id"),
            variant_id=row.get("variant_id"),
            product_url=row.get("product_url"),
        )
        points.append(cp.to_dict())

    return points
def _pick_sale_price(row: pd.Series) -> float:
    """
    Heuristic for which price to use when computing profit.
    You can adjust this to your domain logic.
    """
    # Prefer avg_price, then min_price, then max_price if available
    for col in ("avg_price", "min_price", "max_price", "price"):
        if col in row and not pd.isna(row[col]):
            return float(row[col])
    return float("nan")


def df_to_chart_points(df: pd.DataFrame) -> List[ChartPointDict]:
    """
    Map a DataFrame of canonical/aggregated product rows into ChartPointDicts
    for chart + table consumption.

    Now extended to include profit-related metrics (REQ-034).
    """
    if df is None or df.empty:
        return []

    points: List[ChartPointDict] = []

    for _, row in df.iterrows():
        label = str(row.get("canonical_title") or row.get("title") or "Unknown Product")
        vendor_name = str(row.get("vendor_name") or row.get("vendor") or "Unknown")

        min_price = float(row["min_price"]) if "min_price" in row and not pd.isna(row["min_price"]) else math.nan
        avg_price = float(row["avg_price"]) if "avg_price" in row and not pd.isna(row["avg_price"]) else math.nan
        max_price = float(row["max_price"]) if "max_price" in row and not pd.isna(row["max_price"]) else math.nan

        sale_price = _pick_sale_price(row)
        if math.isnan(sale_price):
            # If no usable price, skip profit, but still create a point
            est_cost = 0.0
            platform_fees = 0.0
            net_profit = 0.0
            roi_pct = None
        else:
            # If your DF already has shipping/packaging columns, plug them here:
            shipping_cost = float(row["shipping_cost"]) if "shipping_cost" in row and not pd.isna(row["shipping_cost"]) else None
            packaging_cost = float(row["packaging_cost"]) if "packaging_cost" in row and not pd.isna(row["packaging_cost"]) else None
            other_costs = float(row["other_costs"]) if "other_costs" in row and not pd.isna(row["other_costs"]) else None

            est_cost, platform_fees, net_profit, roi_pct = compute_profit(
                sale_price=sale_price,
                vendor_name=vendor_name,
                shipping_cost=shipping_cost,
                packaging_cost=packaging_cost,
                other_costs=other_costs,
                # TODO: in the future you could pass user-specific defaults
            )

        point: ChartPointDict = {
            "label": label,
            "vendor": vendor_name,
            "min_price": min_price,
            "avg_price": avg_price,
            "max_price": max_price,
            "listing_count": int(row.get("listing_count", 0)),
            "currency": str(row.get("currency") or "USD"),

            # ---- NEW profit fields ----
            "est_cost": est_cost,
            "platform_fees": platform_fees,
            "net_profit": net_profit,
        }

        if roi_pct is not None:
            point["roi_pct"] = roi_pct

        # Optional extra metrics bucket
        point["extra_metrics"] = {
            "sale_price_for_profit": sale_price,
        }

        points.append(point)

    return points