"""Profit-maximizing price search under merchandising guardrails.

With constant elasticity e < -1, unconstrained profit-optimal price is cost · e / (1 + e), but real
pricing needs constraints, so we search a grid of candidate prices and pick the best one that:
  - stays within ±max_change of the current price (customers notice big jumps)
  - keeps at least `min_margin` gross margin
  - stays within `max_vs_competitor` of the competitor price, when known
  - ends in .99 / .49 (charm pricing) when `charm=True`
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .elasticity import Elasticity


@dataclass(frozen=True)
class Guardrails:
    max_change: float = 0.15
    min_margin: float = 0.20
    max_vs_competitor: float = 0.10
    charm: bool = True


@dataclass
class Recommendation:
    sku: str
    current_price: float
    recommended_price: float
    expected_units_current: float
    expected_units_new: float
    profit_current: float
    profit_new: float
    binding: list[str]
    action: str

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("current_price", "recommended_price", "profit_current", "profit_new"):
            d[k] = round(d[k], 2)
        d["expected_units_current"] = round(self.expected_units_current, 1)
        d["expected_units_new"] = round(self.expected_units_new, 1)
        d["profit_lift_pct"] = round(100 * (self.profit_new - self.profit_current) / self.profit_current, 1) if self.profit_current > 0 else None
        return d


def charm_price(p: float) -> float:
    """Round to the nearest x.49 or x.99 at or below p (never round up past a guardrail)."""
    floor = math.floor(p)
    candidates = [floor - 0.01, floor + 0.49, floor + 0.99]
    below = [c for c in candidates if c <= p + 1e-9 and c > 0]
    return round(max(below), 2) if below else round(p, 2)


def demand_ratio(new_price: float, current_price: float, elasticity: float) -> float:
    return (new_price / current_price) ** elasticity


def recommend(sku: str, current_price: float, cost: float, baseline_units: float, est: Elasticity,
              competitor_price: float | None = None, rails: Guardrails = Guardrails()) -> Recommendation:
    profit_current = (current_price - cost) * baseline_units
    if not est.reliable:
        return Recommendation(sku, current_price, current_price, baseline_units, baseline_units, profit_current, profit_current,
                              ["insufficient-evidence"], "hold")

    # Each bound remembers which guardrail set it, so we can report what limited the recommendation.
    lo, lo_rule = current_price * (1 - rails.max_change), "max-decrease"
    hi, hi_rule = current_price * (1 + rails.max_change), "max-increase"
    floor_margin = cost / (1 - rails.min_margin)
    if floor_margin > lo:
        lo, lo_rule = floor_margin, "min-margin"
    if competitor_price and competitor_price * (1 + rails.max_vs_competitor) < hi:
        hi, hi_rule = competitor_price * (1 + rails.max_vs_competitor), "competitor"
    binding = []
    if lo > hi:
        # Constraints conflict (e.g. cost too close to a low competitor price): protect margin.
        hi, hi_rule = lo, lo_rule
        binding.append("conflict:margin-wins")

    grid = np.linspace(lo, hi, 400)
    if rails.charm:
        charmed = [charm_price(p) for p in grid if charm_price(p) >= lo - 1e-9]
        grid = np.unique(charmed) if charmed else np.array([round(lo, 2)])
    units = baseline_units * demand_ratio(grid, current_price, est.elasticity)
    profit = (grid - cost) * units
    best = int(np.argmax(profit))
    price = float(grid[best])

    # The optimum sits on a guardrail when it's the first or last candidate price.
    if len(grid) > 1 and best == 0:
        binding.append(lo_rule)
    if len(grid) > 1 and best == len(grid) - 1 and hi_rule not in binding:
        binding.append(hi_rule)

    action = "hold" if abs(price - current_price) < 0.01 else ("raise" if price > current_price else "lower")
    return Recommendation(sku, current_price, price, baseline_units, float(units[best]), profit_current, float(profit[best]), binding, action)


def unconstrained_optimum(cost: float, elasticity: float) -> float | None:
    """Closed form for constant elasticity; only defined for elastic demand (e < -1)."""
    if elasticity >= -1:
        return None
    return cost * elasticity / (1 + elasticity)
