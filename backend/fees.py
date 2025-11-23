# backend/pricing/fees.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class FeeRule:
    """
    Represents platform-level fee rules for a marketplace/vendor.

    All percentages are expressed as decimals (e.g., 0.13 = 13%).
    """
    platform_percent: float         # % of sale price charged by platform
    platform_fixed: float           # fixed fee per transaction (USD)
    processor_percent: float = 0.0  # e.g. Stripe/PayPal percent cut
    processor_fixed: float = 0.0    # fixed processor fee (USD)


# -------------------------------------------------------------------
# Vendor Fee Rules (configurable / maintainable)
# In the future, you could load this from a config file or database.
# -------------------------------------------------------------------
DEFAULT_FEE_RULES: Dict[str, FeeRule] = {
    "ebay": FeeRule(
        platform_percent=0.13,
        platform_fixed=0.30,
        processor_percent=0.029,
        processor_fixed=0.30,
    ),
    "amazon": FeeRule(
        platform_percent=0.15,
        platform_fixed=0.00,
        processor_percent=0.029,
        processor_fixed=0.30,
    ),
    "mercari": FeeRule(
        platform_percent=0.10,
        platform_fixed=0.00,
        processor_percent=0.029,
        processor_fixed=0.30,
    ),
    "facebook marketplace": FeeRule(
        platform_percent=0.05,
        platform_fixed=0.00,
        processor_percent=0.029,
        processor_fixed=0.30,
    ),
}


def normalize_vendor_name(vendor_name: Optional[str]) -> str:
    if not vendor_name:
        return "unknown"

    v = vendor_name.strip().lower()
    if "ebay" in v:
        return "ebay"
    if "amazon" in v or "amzn" in v:
        return "amazon"
    if "mercari" in v:
        return "mercari"
    if "facebook" in v or "fb" in v:
        return "facebook marketplace"
    return v  # fallback – use as-is; may not have explicit rule


def get_fee_rule(vendor_name: Optional[str]) -> Optional[FeeRule]:
    """
    Look up a vendor's fee rule. Returns None if not configured.
    """
    key = normalize_vendor_name(vendor_name)
    return DEFAULT_FEE_RULES.get(key)


def calculate_platform_fees(
    sale_price: float,
    vendor_name: Optional[str],
) -> float:
    """
    Calculate total platform + payment processor fees for a given sale price.

    Returns a non-negative float. If no rule is configured, returns 0.0.
    """
    if sale_price is None or sale_price <= 0:
        return 0.0

    rule = get_fee_rule(vendor_name)
    if rule is None:
        return 0.0

    platform_fee = sale_price * rule.platform_percent + rule.platform_fixed
    processor_fee = sale_price * rule.processor_percent + rule.processor_fixed

    total_fees = platform_fee + processor_fee
    return round(max(total_fees, 0.0), 2)


def estimate_costs(
    *,
    shipping_cost: Optional[float] = None,
    packaging_cost: Optional[float] = None,
    other_costs: Optional[float] = None,
    default_shipping: float = 4.0,
    default_packaging: float = 1.0,
    default_other: float = 0.0,
) -> float:
    """
    Estimate total non-platform costs associated with fulfilling the sale.
    These can be overridden per-listing or per-user later.
    """
    ship = default_shipping if shipping_cost is None else shipping_cost
    pack = default_packaging if packaging_cost is None else packaging_cost
    other = default_other if other_costs is None else other_costs

    total = ship + pack + other
    return round(max(total, 0.0), 2)


def compute_profit(
    sale_price: Optional[float],
    vendor_name: Optional[str],
    *,
    shipping_cost: Optional[float] = None,
    packaging_cost: Optional[float] = None,
    other_costs: Optional[float] = None,
    default_shipping: float = 4.0,
    default_packaging: float = 1.0,
    default_other: float = 0.0,
) -> Tuple[float, float, Optional[float]]:
    """
    High-level entry point for REQ-034.

    Returns:
        (estimated_costs, platform_fees, net_profit, roi_pct)

    - estimated_costs: shipping + packaging + other
    - platform_fees: platform + payment processor
    - net_profit: sale_price - estimated_costs - platform_fees
    - roi_pct: (net_profit / estimated_costs) * 100, or None if cost == 0
    """
    if sale_price is None or sale_price <= 0:
        # No meaningful profit if no sale price
        return 0.0, 0.0, 0.0, None

    est_costs = estimate_costs(
        shipping_cost=shipping_cost,
        packaging_cost=packaging_cost,
        other_costs=other_costs,
        default_shipping=default_shipping,
        default_packaging=default_packaging,
        default_other=default_other,
    )
    platform_fees = calculate_platform_fees(sale_price, vendor_name)
    net_profit = round(sale_price - est_costs - platform_fees, 2)

    if est_costs <= 0:
        roi_pct = None
    else:
        roi_pct = round((net_profit / est_costs) * 100.0, 1)

    return est_costs, platform_fees, net_profit, roi_pct
