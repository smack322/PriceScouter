# backend/chart_adapter.py
from __future__ import annotations

from typing import Iterable, List, Optional

import math
import pandas as pd

from .chart_types import ChartPoint, ChartPointDict


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
