import math

import numpy as np
import pandas as pd
import pytest

from pricesense.app import create_app
from pricesense.data import DEFAULT_SKUS, SkuSpec, simulate, validate
from pricesense.elasticity import estimate
from pricesense.optimizer import Guardrails, charm_price, recommend, unconstrained_optimum
from pricesense.service import PricingService


@pytest.fixture(scope="module")
def history():
    return simulate(days=365, seed=11)


@pytest.fixture(scope="module")
def service(history):
    return PricingService(history)


def test_recovers_planted_elasticities(history):
    for spec in DEFAULT_SKUS:
        est = estimate(history, spec.sku, n_boot=100)
        assert est.reliable, (spec.sku, est.reason)
        assert est.ci_low <= spec.elasticity <= est.ci_high, (spec.sku, est.elasticity, est.ci_low, est.ci_high)
        if spec.base_units >= 15:  # high-volume SKUs give tight estimates
            assert abs(est.elasticity - spec.elasticity) < 0.3, (spec.sku, est.elasticity)


def test_flags_products_whose_price_never_moved():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=120), "sku": "FLAT", "price": 9.99, "cost": 4.0,
                       "units": rng.poisson(20, 120), "promo": 0})
    est = estimate(df, "FLAT")
    assert not est.reliable
    assert "barely changed" in est.reason
    short = estimate(df.head(30), "FLAT")
    assert "only 30 days" in short.reason


def test_charm_pricing_never_rounds_up():
    assert charm_price(12.30) == 11.99
    assert charm_price(12.60) == 12.49
    assert charm_price(12.99) == 12.99
    assert charm_price(0.40) == 0.40


def test_optimizer_matches_closed_form_when_unconstrained(history):
    est = estimate(history, "KETTLE")
    est.elasticity = -2.4
    rails = Guardrails(max_change=0.9, min_margin=0.0, max_vs_competitor=10, charm=False)
    rec = recommend("KETTLE", 45.0, 19.0, 4.0, est, None, rails)
    assert rec.recommended_price == pytest.approx(unconstrained_optimum(19.0, -2.4), rel=0.01)
    assert unconstrained_optimum(10, -0.5) is None, "inelastic demand has no interior optimum"


def test_inelastic_products_raise_up_to_the_guardrail(history):
    est = estimate(history, "FILTERS")
    rec = recommend("FILTERS", 5.99, 1.50, 40, est, competitor_price=None, rails=Guardrails(charm=False))
    assert rec.action == "raise"
    assert rec.recommended_price == pytest.approx(5.99 * 1.15, abs=0.02)
    assert "max-increase" in rec.binding
    assert rec.profit_new > rec.profit_current


def test_competitor_and_margin_guardrails(history):
    est = estimate(history, "FILTERS")
    capped = recommend("FILTERS", 5.99, 1.50, 40, est, competitor_price=6.00, rails=Guardrails(charm=False))
    assert capped.recommended_price <= 6.60 + 1e-9
    assert "competitor" in capped.binding

    elastic = estimate(history, "KETTLE")
    # Optimum p* = cost·e/(1+e) = 22·6/5 = 26.40, below the 20%-margin floor of 27.50, so the floor binds.
    elastic.elasticity = -6.0
    floor = recommend("KETTLE", 30.0, 22.0, 4, elastic, rails=Guardrails(charm=False))
    assert floor.recommended_price >= 22.0 / 0.8 - 0.01, "never below the margin floor"
    assert "min-margin" in floor.binding


def test_unreliable_estimates_hold_price():
    est = estimate(pd.DataFrame({"date": pd.date_range("2017-01-01", periods=10), "sku": "X", "price": 5.0, "cost": 2.0, "units": 3}), "X")
    rec = recommend("X", 5.0, 2.0, 3, est)
    assert rec.action == "hold" and rec.binding == ["insufficient-evidence"]


def test_service_baseline_excludes_promos_and_normalizes_price(service):
    snap = service.snapshot("COFFEE-12OZ")
    assert 30 < snap["baseline_units"] < 120
    recs = service.recommend_all()
    assert len(recs) == len(DEFAULT_SKUS)
    assert all(r["recommended_price"] > 0 for r in recs)


def test_api(service):
    client = create_app(service).test_client()
    assert len(client.get("/api/elasticities").get_json()) == 5
    r = client.get("/api/recommendations/MUG-LOGO?max_change=0.05").get_json()
    assert abs(r["recommended_price"] - r["current_price"]) <= r["current_price"] * 0.05 + 0.01
    assert client.get("/api/recommendations/NOPE").status_code == 404
    assert client.get("/api/recommendations/MUG-LOGO?max_change=abc").status_code == 400
    assert client.get("/api/recommendations/MUG-LOGO?max_change=0.9").status_code == 400
    sim = client.post("/api/simulate", json={"sku": "GRINDER", "price": 44.99}).get_json()
    assert sim["expected_units"] > 0 and math.isfinite(sim["expected_profit"])
    assert client.post("/api/simulate", json={"sku": "GRINDER", "price": -1}).status_code == 400


def test_validation():
    with pytest.raises(ValueError, match="missing columns"):
        validate(pd.DataFrame({"sku": ["a"]}))
    with pytest.raises(ValueError, match="positive"):
        validate(pd.DataFrame({"date": ["2017-01-01"], "sku": ["a"], "price": [0], "cost": [1], "units": [1]}))
