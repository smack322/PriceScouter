# backend/chart_types.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, TypedDict


class ChartPointDict(TypedDict, total=False):
    """
    Stable data contract for chart data.

    Required fields (backwards-compatible core):
      - label
      - vendor
      - min_price
      - avg_price
      - max_price
      - listing_count

    Optional fields can be added over time as needed.
    """
    label: str
    vendor: str
    min_price: Optional[float]
    avg_price: Optional[float]
    max_price: Optional[float]
    listing_count: int

    # Optional / forward-compatible fields
    currency: str
    canonical_id: int
    variant_id: int
    product_url: str


@dataclass
class ChartPoint:
    """Strongly-typed representation used in Python code."""
    label: str
    vendor: str
    min_price: Optional[float]
    avg_price: Optional[float]
    max_price: Optional[float]
    listing_count: int

    # Optional / forward-compatible fields
    currency: str = "USD"
    canonical_id: Optional[int] = None
    variant_id: Optional[int] = None
    product_url: Optional[str] = None

    def to_dict(self) -> ChartPointDict:
        """Convert to the JSON-friendly dict that the UI expects."""
        raw = asdict(self)
        # Drop None optional fields so they don't break existing consumers.
        return {k: v for k, v in raw.items() if v is not None}  # type: ignore[return-value]
