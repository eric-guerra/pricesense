"""Glue: fit elasticities on history and produce recommendations for every SKU."""
from __future__ import annotations

import pandas as pd

from .elasticity import Elasticity, estimate_all
from .optimizer import Guardrails, recommend


class PricingService:
    def __init__(self, history: pd.DataFrame, rails: Guardrails = Guardrails(), lookback_days: int = 28):
        self.history = history.copy()
        self.history["date"] = pd.to_datetime(self.history["date"])
        self.rails = rails
        self.lookback_days = lookback_days
        self.elasticities: dict[str, Elasticity] = estimate_all(self.history)

    def snapshot(self, sku: str) -> dict:
        """Current price, cost, competitor price and baseline demand (recent average, promo days excluded)."""
        d = self.history[self.history["sku"] == sku].sort_values("date")
        if d.empty:
            raise KeyError(sku)
        last = d.iloc[-1]
        recent = d[d["date"] > d["date"].max() - pd.Timedelta(days=self.lookback_days)]
        if "promo" in recent:
            recent = recent[recent["promo"] == 0]
        # Normalize recent demand to the current price using the elasticity, so a recent price change
        # doesn't bias the baseline.
        est = self.elasticities[sku]
        e = est.elasticity if est.reliable else 0.0
        baseline = float((recent["units"] * (last["price"] / recent["price"]) ** e).mean()) if len(recent) else float(d["units"].mean())
        return {
            "price": float(last["price"]), "cost": float(last["cost"]), "baseline_units": baseline,
            "competitor_price": float(last["competitor_price"]) if "competitor_price" in d and pd.notna(last.get("competitor_price")) else None,
        }

    def recommend(self, sku: str, rails: Guardrails | None = None) -> dict:
        snap = self.snapshot(sku)
        rec = recommend(sku, snap["price"], snap["cost"], snap["baseline_units"], self.elasticities[sku], snap["competitor_price"], rails or self.rails)
        out = rec.as_dict()
        out["elasticity"] = self.elasticities[sku].as_dict()
        return out

    def recommend_all(self) -> list[dict]:
        return [self.recommend(sku) for sku in sorted(self.elasticities)]

    def simulate(self, sku: str, price: float) -> dict:
        """What-if: expected daily units and profit at an arbitrary price."""
        snap = self.snapshot(sku)
        est = self.elasticities[sku]
        if not est.reliable:
            raise ValueError(f"no reliable elasticity for {sku}: {est.reason}")
        units = snap["baseline_units"] * (price / snap["price"]) ** est.elasticity
        return {"sku": sku, "price": price, "expected_units": round(units, 2), "expected_profit": round((price - snap["cost"]) * units, 2),
                "margin_pct": round(100 * (price - snap["cost"]) / price, 1)}
