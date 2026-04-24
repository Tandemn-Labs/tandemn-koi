"""Cost projection helpers for soft cost-roofline evaluation."""

from __future__ import annotations

import math
from typing import Optional, Tuple


def project_total_cost(
    cost_per_hour: Optional[float],
    elapsed_hours: Optional[float],
    projected_eta_hours: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (projected_remaining_cost, projected_total_cost)."""
    if not cost_per_hour or cost_per_hour <= 0:
        return None, None

    elapsed = float(elapsed_hours or 0.0)
    eta = projected_eta_hours
    if eta is None:
        return None, None

    if math.isfinite(eta):
        remaining = cost_per_hour * eta
    else:
        remaining = float("inf")
    total = (cost_per_hour * elapsed) + remaining
    return remaining, total


def evaluate_cost_roofline(
    total_cost: Optional[float],
    cost_roofline_usd: Optional[float],
) -> Tuple[Optional[bool], Optional[float]]:
    """Return ``(meets_cost_roofline, cost_overage_usd)``.

    Conventions:
    - ``(None, None)``: no roofline or no total-cost estimate is available yet.
    - ``(True, 0.0)``: total cost is under or exactly at the roofline.
    - ``(False, <finite>)``: total cost is over the roofline by a real amount.
    - ``(False, inf)``: total cost is unbounded/infinite, so overage is also unbounded.
    """
    if total_cost is None or cost_roofline_usd is None:
        return None, None
    if not math.isfinite(total_cost):
        return False, math.inf
    meets = total_cost <= cost_roofline_usd
    return meets, round(max(0.0, total_cost - cost_roofline_usd), 2)
