"""Flask API."""
from __future__ import annotations

import os

from flask import Flask, jsonify, request

from .data import load_csv, simulate
from .optimizer import Guardrails
from .service import PricingService


def create_app(service: PricingService | None = None) -> Flask:
    app = Flask(__name__)
    if service is None:
        path = os.environ.get("PRICESENSE_DATA")
        service = PricingService(load_csv(path) if path else simulate())
    app.config["SERVICE"] = service

    def not_found(sku):
        return jsonify({"error": f"unknown sku {sku}"}), 404

    @app.get("/api/elasticities")
    def elasticities():
        return jsonify([e.as_dict() for e in service.elasticities.values()])

    @app.get("/api/recommendations")
    def recommendations():
        return jsonify(service.recommend_all())

    @app.get("/api/recommendations/<sku>")
    def recommendation(sku):
        if sku not in service.elasticities:
            return not_found(sku)
        try:
            rails = Guardrails(
                max_change=float(request.args.get("max_change", 0.15)),
                min_margin=float(request.args.get("min_margin", 0.20)),
                max_vs_competitor=float(request.args.get("max_vs_competitor", 0.10)),
                charm=request.args.get("charm", "true").lower() != "false",
            )
        except ValueError:
            return jsonify({"error": "guardrail parameters must be numbers"}), 400
        if not (0 < rails.max_change <= 0.5 and 0 <= rails.min_margin < 0.95):
            return jsonify({"error": "max_change must be in (0, 0.5] and min_margin in [0, 0.95)"}), 400
        return jsonify(service.recommend(sku, rails))

    @app.post("/api/simulate")
    def what_if():
        body = request.get_json(silent=True) or {}
        sku, price = body.get("sku"), body.get("price")
        if sku not in service.elasticities:
            return not_found(sku)
        if not isinstance(price, (int, float)) or price <= 0:
            return jsonify({"error": "price must be a positive number"}), 400
        try:
            return jsonify(service.simulate(sku, float(price)))
        except ValueError as e:
            return jsonify({"error": str(e)}), 422

    @app.get("/healthz")
    def health():
        return jsonify({"ok": True, "skus": len(service.elasticities)})

    return app
