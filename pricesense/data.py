"""Sales history: loading and a synthetic generator with known ground-truth elasticities.

Columns: date, sku, price, cost, units, promo (0/1), competitor_price.
The generator makes it possible to *prove* the estimator works: fit on simulated data and check
that the recovered elasticities match the ones we planted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = ["date", "sku", "price", "cost", "units"]


@dataclass(frozen=True)
class SkuSpec:
    sku: str
    base_price: float
    cost: float
    base_units: float
    elasticity: float          # % change in demand per 1% change in price (negative)
    promo_lift: float = 0.25   # +25% units when on promo
    weekend_lift: float = 0.15


DEFAULT_SKUS = [
    SkuSpec("COFFEE-12OZ", 11.99, 5.20, 60, -1.8),
    SkuSpec("MUG-LOGO", 14.50, 4.00, 18, -0.9),
    SkuSpec("GRINDER", 49.99, 21.00, 6, -1.3),
    SkuSpec("FILTERS", 5.99, 1.50, 40, -0.6),
    SkuSpec("KETTLE", 45.00, 19.00, 4, -2.4),
]


def simulate(skus=DEFAULT_SKUS, days: int = 365, seed: int = 7, start: str = "2017-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for spec in skus:
        # Prices wander ±20% around base with occasional promos: the variation the estimator needs.
        price_mult = np.clip(1 + rng.normal(0, 0.08, days).cumsum() * 0.05 + rng.normal(0, 0.05, days), 0.8, 1.2)
        promo = (rng.random(days) < 0.08).astype(int)
        price = np.round(spec.base_price * price_mult * np.where(promo == 1, 0.85, 1.0), 2)
        competitor = np.round(spec.base_price * (1 + rng.normal(0, 0.05, days)), 2)
        weekend = (dates.dayofweek >= 5).astype(int)
        expected = (spec.base_units
                    * (price / spec.base_price) ** spec.elasticity
                    * (1 + spec.promo_lift * promo)
                    * (1 + spec.weekend_lift * weekend))
        units = rng.poisson(expected)
        rows.append(pd.DataFrame({
            "date": dates, "sku": spec.sku, "price": price, "cost": spec.cost, "units": units,
            "promo": promo, "competitor_price": competitor,
        }))
    return pd.concat(rows, ignore_index=True)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if (df["price"] <= 0).any():
        raise ValueError("prices must be positive")
    if (df["units"] < 0).any():
        raise ValueError("units cannot be negative")
