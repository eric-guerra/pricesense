"""python -m pricesense.cli {report,simulate-data,serve}"""
from __future__ import annotations

import argparse

from .data import load_csv, simulate
from .service import PricingService


def main(argv=None):
    p = argparse.ArgumentParser(prog="pricesense")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="print elasticities and recommendations")
    r.add_argument("--data", help="sales history CSV (default: simulated)")
    g = sub.add_parser("simulate-data", help="write a synthetic history CSV")
    g.add_argument("-o", "--out", default="sales.csv")
    g.add_argument("--days", type=int, default=365)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=5000)
    args = p.parse_args(argv)

    if args.cmd == "simulate-data":
        simulate(days=args.days).to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    elif args.cmd == "report":
        svc = PricingService(load_csv(args.data) if args.data else simulate())
        print(f"{'SKU':<14}{'elasticity':>11}{'95% CI':>18}  {'price':>8} → {'rec':>8}  {'profit lift':>11}  binding")
        for rec in svc.recommend_all():
            e = rec["elasticity"]
            ci = f"[{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]" if e["reliable"] else e["reason"][:16]
            lift = f"{rec['profit_lift_pct']:+.1f}%" if rec["profit_lift_pct"] is not None else "—"
            print(f"{rec['sku']:<14}{e['elasticity']:>11.2f}{ci:>18}  {rec['current_price']:>8.2f} → {rec['recommended_price']:>8.2f}  {lift:>11}  {','.join(rec['binding'])}")
    elif args.cmd == "serve":
        from .app import create_app
        create_app().run(port=args.port)


if __name__ == "__main__":
    main()
