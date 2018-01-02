# PriceSense

Data-driven price recommendations for retail. PriceSense learns each product's **price elasticity** from sales history, then recommends the profit-maximizing price inside merchandising guardrails. It runs as a CLI and a **Flask** API, built with **pandas** and **scikit-learn**.

> Portfolio reference implementation of a 2017 retail analytics project.

## Method

**Demand model** ([elasticity.py](pricesense/elasticity.py)): constant-elasticity demand fit as a **Poisson GLM**:

```
E[units] = exp(a + e·log(price) + b·promo + c·weekend + d·log(competitor_price))
```

`e` is the elasticity. Two modeling choices matter:
- Poisson rather than OLS on `log(units+1)`, which biases elasticities toward zero for slow sellers (a few units a day)
- `log(competitor)` rather than `log(competitor/price)`, which is collinear with own price

**Uncertainty**: bootstrap confidence intervals. A product is marked **unreliable**, and its price held, when it has too little history, its price barely moved, the CI is too wide, or the estimate has the wrong sign.

**Optimizer** ([optimizer.py](pricesense/optimizer.py)): maximizes `(price − cost) × demand(price)` over candidate prices, subject to:
- ±15% max change
- ≥20% gross margin
- ≤10% above the competitor
- charm endings (.49/.99, never rounding up past a limit)

It reports which guardrail **binds**, so merchants know why a price stopped where it did. With no guardrails it reproduces the closed-form optimum `cost·e/(1+e)` (tested).

**Validated on synthetic data with planted elasticities**: the true value falls inside the 95% CI for every SKU, and high-volume SKUs are within ±0.3.

```
$ python -m pricesense.cli report
SKU            elasticity            95% CI     price →      rec  profit lift  binding
COFFEE-12OZ         -1.96    [-2.15, -1.82]      9.74 →    10.49        +0.7%
FILTERS             -0.57    [-0.76, -0.36]      5.06 →     5.49        +7.0%  max-increase
GRINDER             -1.11    [-1.76, -0.47]     53.28 →    55.49        +2.1%  competitor
KETTLE              -0.72    [-1.42, -0.07]     42.63 →    48.99       +14.8%  max-increase
MUG-LOGO            -0.70    [-1.10, -0.25]     15.26 →    16.49        +5.1%  competitor
```

## API

| | |
|---|---|
| `GET /api/elasticities` | per-SKU estimate, CI, reliability and reason |
| `GET /api/recommendations[/<sku>]?max_change=&min_margin=&max_vs_competitor=&charm=` | recommendation with expected units, profit lift and binding guardrails |
| `POST /api/simulate {sku, price}` | what-if: expected units, profit and margin at any price |

## Run

```bash
pip install -r requirements.txt
python -m pricesense.cli simulate-data -o sales.csv
PRICESENSE_DATA=sales.csv python -m pricesense.cli serve
docker build -t pricesense . && docker run -p 5000:5000 pricesense
pytest
```

Input CSV columns: `date, sku, price, cost, units` (required) plus `promo, competitor_price` (optional).
