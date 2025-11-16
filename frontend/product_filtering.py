from __future__ import annotations
from typing import Iterable, Optional, Tuple

import pandas as pd


def apply_product_filters(
    df: pd.DataFrame,
    search_text: str = "",
    vendors: Optional[Iterable[str]] = None,
    price_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Apply client-side filters so both table and chart see the same filtered set.
    """
    if df is None or df.empty:
        return df.copy()

    filtered = df.copy()

    # Text search on canonical_title (extend as needed)
    if search_text:
        pattern = search_text.strip().lower()
        filtered = filtered[
            filtered["canonical_title"].str.lower().str.contains(pattern, na=False)
        ]

    # Vendor filter
    if vendors:
        vendors = set(vendors)
        filtered = filtered[filtered["vendor"].isin(vendors)]

    # Price range filter
    if price_range is not None:
        low, high = price_range
        filtered = filtered[
            (filtered["total_price"] >= low) & (filtered["total_price"] <= high)
        ]

    return filtered