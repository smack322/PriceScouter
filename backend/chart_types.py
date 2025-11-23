# backend/chart_types.py

from typing import TypedDict, Optional, Dict


class ChartPointDict(TypedDict, total=False):
    label: str
    vendor: str
    min_price: float
    avg_price: float
    max_price: float
    listing_count: int
    currency: str

    # ---- NEW for REQ-034 ----
    est_cost: float                 # estimated non-platform costs
    platform_fees: float            # total platform + processor fees
    net_profit: float               # sale_price - est_cost - platform_fees
    roi_pct: Optional[float]        # percentage, e.g. 42.3 for 42.3%

    # Future-proof bucket for additional metrics
    extra_metrics: Dict[str, float]
