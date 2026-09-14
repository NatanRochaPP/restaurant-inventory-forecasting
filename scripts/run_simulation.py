"""Reproduce the operational evidence: manual baseline versus forecast-driven ordering.

Replays the holdout period day by day under both policies, writes the KPI comparison to
``outputs/`` and SQLite, and optionally sweeps both policies across their tuning
parameter so they can be compared at a matched service level - which is the only
defensible way to rank two replenishment policies.

Usage::

    python scripts/run_simulation.py
    python scripts/run_simulation.py --frontier
    python scripts/run_simulation.py --start 2025-07-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.data.loader import build_sku_master, load_sales  # noqa: E402
from src.evaluation.frontier import (  # noqa: E402
    matched_comparison,
    run_frontier,
    service_level_at_matched_waste,
)
from src.evaluation.inventory_metrics import compare_simulations, compute_kpis, kpis_by_sku  # noqa: E402
from src.forecasting.model_selection import ModelSelector  # noqa: E402
from src.forecasting.service import ForecastService  # noqa: E402
from src.inventory.simulator import InventorySimulator  # noqa: E402
from src.inventory.strategies import ForecastDrivenStrategy, ParLevelStrategy  # noqa: E402
from src.persistence.database import Database  # noqa: E402
from src.segmentation import segment_skus, to_segments  # noqa: E402

logger = logging.getLogger("run_simulation")
OUTPUTS = PROJECT_ROOT / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline versus AI inventory simulation.")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--frontier", action="store_true", help="Also sweep the service-level frontier.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-database", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    sales = load_sales(config=config)
    master = build_sku_master(sales, config)
    start = pd.Timestamp(args.start) if args.start else pd.Timestamp(config.simulation.start_date)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(config.simulation.end_date)

    print(f"Holdout replay: {start.date()} to {end.date()} ({(end - start).days + 1} days)")
    print(
        f"Supplier: {config.supplier.lead_time_days}-day lead time, order days "
        f"{config.supplier.order_days}; policy {config.inventory.policy} at "
        f"{config.inventory.service_level:.0%} service level"
    )

    segments = to_segments(segment_skus(sales, master, config, as_of=start))
    choices = ModelSelector(config, "rule").select(sales, master, as_of=start)
    service = ForecastService(sales, config, model_choices=choices)

    baseline = InventorySimulator(
        sales, config, ParLevelStrategy(sales, config), sku_master=master
    ).run(start, end)
    ai = InventorySimulator(
        sales, config, ForecastDrivenStrategy(service, config, segments=segments), sku_master=master
    ).run(start, end)

    baseline_kpis = compute_kpis(baseline.daily, baseline.orders)
    ai_kpis = compute_kpis(ai.daily, ai.orders)
    comparison = compare_simulations(baseline_kpis, ai_kpis, baseline.strategy_name, ai.strategy_name)

    print("\nTable 2 - Operational outcomes over the holdout period")
    print(
        comparison[["metric", baseline.strategy_name, ai.strategy_name, "improvement_pct"]]
        .to_string(index=False)
    )
    print("\nA positive improvement means the forecast-driven policy performed better.")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUTS / "simulation_comparison.csv", index=False)
    baseline.daily.to_csv(OUTPUTS / "simulation_daily_baseline.csv", index=False)
    ai.daily.to_csv(OUTPUTS / "simulation_daily_ai.csv", index=False)
    kpis_by_sku(ai.daily, ai.orders).to_csv(OUTPUTS / "simulation_by_sku_ai.csv", index=False)
    kpis_by_sku(baseline.daily, baseline.orders).to_csv(OUTPUTS / "simulation_by_sku_baseline.csv", index=False)

    payload = {
        "start": str(start.date()),
        "end": str(end.date()),
        "baseline": baseline_kpis,
        "ai": ai_kpis,
        "comparison": comparison.to_dict(orient="records"),
    }

    if args.frontier:
        print("\nSweeping the service-level frontier for both policies…")
        frontier = run_frontier(
            sales, config, forecast_service=service, sku_master=master,
            service_levels=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99],
            buffer_pcts=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
            start=start, end=end,
        )
        frontier.to_csv(OUTPUTS / "frontier.csv", index=False)
        print("\nFrontier (each policy swept across its tuning parameter)")
        print(
            frontier[[
                "strategy", "setting_value", "unit_service_level", "stockout_units",
                "waste_units", "total_cost", "average_inventory_units",
            ]].round(2).to_string(index=False)
        )

        matched = matched_comparison(frontier)
        matched.to_csv(OUTPUTS / "matched_service_level.csv", index=False)
        print("\nTable 3 - Compared at matched service level")
        if matched.empty:
            print("  The policies' service levels do not overlap; widen the sweep.")
        else:
            print(matched.to_string(index=False))
            best = matched.loc[matched["waste_reduction_pct"].idxmax()]
            print(
                f"\n  At a matched service level of {best['matched_service_level_pct']:.2f}%: "
                f"waste {best['waste_reduction_pct']:.1f}% lower, total cost "
                f"{best['cost_reduction_pct']:.1f}% lower, average inventory "
                f"{best['inventory_reduction_pct']:.1f}% lower."
            )
            payload["matched_service_level"] = matched.to_dict(orient="records")

        at_waste = service_level_at_matched_waste(frontier)
        at_waste.to_csv(OUTPUTS / "matched_waste.csv", index=False)
        if not at_waste.empty:
            print("\nTable 4 - Compared at matched waste")
            print(at_waste.to_string(index=False))
            payload["matched_waste"] = at_waste.to_dict(orient="records")

    (OUTPUTS / "simulation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.no_database:
        database = Database.from_config(config)
        database.initialise()
        database.save_sku_master(master)
        for result, kpis in ((baseline, baseline_kpis), (ai, ai_kpis)):
            database.save_simulation(
                run_id=f"{result.strategy_name}-{start:%Y%m%d}-{end:%Y%m%d}",
                strategy=result.strategy_name,
                daily=result.daily,
                orders=result.orders,
                kpis=kpis,
                start=start,
                end=end,
                config=config,
            )
        # The most recent decision date, kept as the working set the dashboard opens on.
        latest_recommendations = [r for r in ai.recommendations if r.as_of == max(x.as_of for x in ai.recommendations)]
        database.save_recommendations(latest_recommendations, run_id=f"ai-{end:%Y%m%d}")
        latest_as_of = max(r.as_of for r in ai.recommendations)
        database.save_forecasts(
            list(service.forecast(latest_as_of).values()),
            run_id=f"ai-{end:%Y%m%d}",
            as_of=latest_as_of,
        )
        print(f"\nDatabase updated: {database.path}")

    print(f"Wrote simulation outputs to {OUTPUTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
