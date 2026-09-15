"""Reproduce the forecast-accuracy evidence.

Runs the rolling-origin backtest that the dissertation reports (12 expanding-window
folds, 7-day horizon) and writes the results table to ``outputs/``.

Usage::

    python scripts/run_backtest.py
    python scripts/run_backtest.py --folds 6 --skus "Burger Buns" "Truffle Oil"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.data.loader import load_sales  # noqa: E402
from src.evaluation.backtest import run_backtest  # noqa: E402
from src.evaluation.forecast_metrics import improvement_vs_benchmark  # noqa: E402

logger = logging.getLogger("run_backtest")
OUTPUTS = PROJECT_ROOT / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin forecast backtest.")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--skus", nargs="*", default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-ml", action="store_true", help="Skip the gradient-boosting model.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    sales = load_sales(config=config)

    folds = args.folds or config.forecast.backtest.n_folds
    horizon = args.horizon or config.forecast.backtest.horizon_days
    print(
        f"Rolling-origin backtest: {folds} expanding-window folds, {horizon}-day horizon, "
        f"{sales['sku'].nunique()} SKUs, seed {config.forecast.random_seed}"
    )

    result = run_backtest(
        sales, config, n_folds=folds, horizon=horizon, skus=args.skus,
        include_gradient_boosting=not args.no_ml,
    )

    summary = result.summary()
    summary["sMAPE vs SNaive %"] = improvement_vs_benchmark(summary)
    print("\nTable 1 - Forecast accuracy (mean across folds and SKUs)")
    print(summary.to_string())

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(OUTPUTS / "backtest_metrics.csv", index=False)
    summary.to_csv(OUTPUTS / "forecast_accuracy.csv")
    result.predictions.to_csv(OUTPUTS / "backtest_predictions.csv", index=False)

    by_sku = result.summary(by="sku")
    by_sku.to_csv(OUTPUTS / "forecast_accuracy_by_sku.csv")

    best = summary["sMAPE"].idxmin()
    payload = {
        "n_folds": folds,
        "horizon_days": horizon,
        "n_skus": int(sales["sku"].nunique()),
        "random_seed": config.forecast.random_seed,
        "best_model": best,
        "metrics": summary.to_dict(orient="index"),
        "note": (
            "Features are frozen at each forecast origin and rolling statistics are "
            "computed within each SKU, so no fold can see its own evaluation window."
        ),
    }
    (OUTPUTS / "forecast_accuracy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nBest model by sMAPE: {best}")
    print(f"Wrote backtest_metrics.csv, forecast_accuracy.csv/.json and by-SKU breakdown to {OUTPUTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
