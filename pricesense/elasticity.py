"""Own-price elasticity estimation with a constant-elasticity demand model.

    E[units] = exp(a + e·log(price) + b·promo + c·weekend + d·log(competitor_price))

fit as a Poisson GLM (log link). The coefficient `e` is the elasticity. A Poisson GLM handles
low-volume products and zero-sale days correctly, unlike OLS on log(units + 1), which biases
elasticities toward zero when daily counts are small. Confidence intervals come from a nonparametric bootstrap
over days, which also gives a quality flag: products whose price never moved can't be priced
from data and are marked unreliable instead of producing a confident-looking guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


@dataclass
class Elasticity:
    sku: str
    elasticity: float
    ci_low: float
    ci_high: float
    n_days: int
    price_cv: float          # coefficient of variation of price (how much it moved)
    reliable: bool
    reason: str | None
    model: PoissonRegressor | None
    features: list[str]

    def as_dict(self) -> dict:
        return {
            "sku": self.sku, "elasticity": round(self.elasticity, 3), "ci": [round(self.ci_low, 3), round(self.ci_high, 3)],
            "n_days": self.n_days, "price_cv": round(self.price_cv, 3), "reliable": self.reliable, "reason": self.reason,
        }


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = {"log_price": np.log(df["price"].to_numpy())}
    if "promo" in df:
        cols["promo"] = df["promo"].to_numpy()
    cols["weekend"] = (pd.to_datetime(df["date"]).dt.dayofweek >= 5).astype(int).to_numpy()
    if "competitor_price" in df and df["competitor_price"].notna().all():
        # log(competitor) rather than log(competitor/price): the ratio contains -log(price) and would be
        # collinear with the own-price term, inflating the variance of the elasticity estimate.
        cols["log_competitor"] = np.log(df["competitor_price"].to_numpy())
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


def estimate(df: pd.DataFrame, sku: str, n_boot: int = 200, seed: int = 0, min_days: int = 60, min_price_cv: float = 0.03) -> Elasticity:
    d = df[df["sku"] == sku].sort_values("date")
    if d.empty:
        raise KeyError(sku)
    X, names = design_matrix(d)
    y = d["units"].to_numpy()
    price_cv = float(d["price"].std() / d["price"].mean()) if len(d) > 1 else 0.0

    reason = None
    if len(d) < min_days:
        reason = f"only {len(d)} days of history (need {min_days})"
    elif price_cv < min_price_cv:
        reason = "price barely changed, so demand response can't be measured"
    if reason:
        return Elasticity(sku, float("nan"), float("nan"), float("nan"), len(d), price_cv, False, reason, None, names)

    def fit(Xs, ys):
        return PoissonRegressor(alpha=0.0, max_iter=1000).fit(Xs, ys)

    model = fit(X, y)
    e = float(model.coef_[0])

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        boots.append(fit(X[idx], y[idx]).coef_[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    if e >= 0:
        reliable, reason = False, "estimated demand rises with price (likely confounded by promotions or seasonality)"
    elif hi - lo > 2.0:
        reliable, reason = False, "estimate is too uncertain"
    else:
        reliable = True
    return Elasticity(sku, e, float(lo), float(hi), len(d), price_cv, reliable, reason, model, names)


def estimate_all(df: pd.DataFrame, **kwargs) -> dict[str, Elasticity]:
    return {sku: estimate(df, sku, **kwargs) for sku in sorted(df["sku"].unique())}
