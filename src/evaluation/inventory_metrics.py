"""Operational KPIs and the baseline-versus-AI comparison.

Forecast accuracy is only an intermediate outcome. The research question asks whether
forecast-driven replenishment reduces stockouts and food waste, so these are the numbers
the dissertation reports: service level, waste, and the cost of both.

Every metric is computed from the simulator's per-SKU-day records, so the same code
serves the aggregate comparison, the per-SKU breakdown and the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

Direction = Literal["lower_is_better", "higher_is_better"]


@dataclass(frozen=True)
class MetricSpec:
    """How a KPI should be displayed and compared."""

    key: str
    label: str
    direction: Direction
    unit: str = ""
    precision: int = 1


#: The KPI set reported for every simulation run.
METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("stockout_days", "Stockout days (SKU-days)", "lower_is_better", "days", 0),
    MetricSpec("stockout_units", "Stockout units", "lower_is_better", "units", 0),
    MetricSpec("unit_service_level", "Service level (unit fill rate)", "higher_is_better", "%", 2),
    MetricSpec("day_service_level", "Service level (days without stockout)", "higher_is_better", "%", 2),
    MetricSpec("waste_units", "Waste units", "lower_is_better", "units", 0),
    MetricSpec("waste_share_of_supply", "Waste as share of stock received", "lower_is_better", "%", 2),
    MetricSpec("waste_cost", "Waste cost", "lower_is_better", "GBP", 2),
    MetricSpec("holding_cost", "Holding cost", "lower_is_better", "GBP", 2),
    MetricSpec("ordering_cost", "Ordering cost", "lower_is_better", "GBP", 2),
    MetricSpec("stockout_cost", "Stockout cost", "lower_is_better", "GBP", 2),
    MetricSpec("total_cost", "Total cost", "lower_is_better", "GBP", 2),
    MetricSpec("average_inventory_units", "Average inventory (units)", "lower_is_better", "units", 1),
    MetricSpec("average_inventory_value", "Average inventory value", "lower_is_better", "GBP", 2),
    MetricSpec("orders_placed", "Orders placed", "lower_is_better", "orders", 0),
)

_SPEC_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}


def compute_kpis(daily: pd.DataFrame, orders: pd.DataFrame | None = None) -> dict[str, float]:
    """Aggregate a simulation's daily records into headline KPIs.

    Args:
        daily: Per-SKU-day frame produced by
            :class:`~src.inventory.simulator.InventorySimulator`.
        orders: Optional order frame, used for the order count.

    Returns:
        A mapping of KPI key to value. Service levels are percentages.

    Raises:
        ValueError: If the frame is missing columns the KPIs depend on.
    """
    required = {
        "demand", "served", "stockout_units", "stockout_day", "expired_units",
        "closing_stock", "inventory_value", "received",
        "holding_cost", "waste_cost", "stockout_cost", "ordering_cost",
    }
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Daily simulation frame is missing column(s) {sorted(missing)}.")
    if daily.empty:
        raise ValueError("Cannot compute KPIs from an empty simulation result.")

    total_demand = float(daily["demand"].sum())
    total_served = float(daily["served"].sum())
    total_received = float(daily["received"].sum())
    waste_units = float(daily["expired_units"].sum())
    n_sku_days = int(len(daily))

    return {
        "stockout_days": float(daily["stockout_day"].sum()),
        "stockout_units": float(daily["stockout_units"].sum()),
        "unit_service_level": float(total_served / total_demand * 100) if total_demand > 0 else 100.0,
        "day_service_level": float((1 - daily["stockout_day"].mean()) * 100) if n_sku_days else 100.0,
        "waste_units": waste_units,
        "waste_share_of_supply": float(waste_units / total_received * 100) if total_received > 0 else 0.0,
        "waste_cost": float(daily["waste_cost"].sum()),
        "holding_cost": float(daily["holding_cost"].sum()),
        "ordering_cost": float(daily["ordering_cost"].sum()),
        "stockout_cost": float(daily["stockout_cost"].sum()),
        "total_cost": float(
            daily[["holding_cost", "waste_cost", "stockout_cost", "ordering_cost"]].sum().sum()
        ),
        "average_inventory_units": float(daily["closing_stock"].mean()),
        "average_inventory_value": float(daily["inventory_value"].mean()),
        "orders_placed": float(len(orders)) if orders is not None else float("nan"),
        "total_demand": total_demand,
        "total_received": total_received,
    }


def kpis_by_sku(daily: pd.DataFrame, orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-SKU KPI breakdown, sorted by total cost.

    Identifies which products drive the aggregate result - typically a handful of
    short-shelf-life items dominate waste while a different set drives stockouts.
    """
    rows = []
    for sku, group in daily.groupby("sku", observed=True):
        sku_orders = orders[orders["sku"] == sku] if orders is not None and not orders.empty else None
        rows.append({"sku": str(sku), **compute_kpis(group, sku_orders)})
    return pd.DataFrame(rows).sort_values("total_cost", ascending=False).reset_index(drop=True)


def percentage_improvement(baseline: float, comparison: float, direction: Direction) -> float:
    """Percentage improvement of ``comparison`` over ``baseline``.

    Positive values always mean "better", whichever direction the metric runs in.

    Returns:
        The improvement in percent, or NaN when the baseline is zero and no meaningful
        relative change can be expressed.
    """
    if baseline == 0:
        if comparison == 0:
            return 0.0
        return float("nan")
    if direction == "lower_is_better":
        return float((baseline - comparison) / abs(baseline) * 100)
    return float((comparison - baseline) / abs(baseline) * 100)


def compare_simulations(
    baseline_kpis: dict[str, float],
    ai_kpis: dict[str, float],
    baseline_label: str = "Manual par-level baseline",
    ai_label: str = "AI forecast + inventory policy",
) -> pd.DataFrame:
    """Build the headline baseline-versus-AI comparison table.

    Args:
        baseline_kpis: KPIs from the manual baseline run.
        ai_kpis: KPIs from the forecast-driven run.
        baseline_label: Column name for the baseline.
        ai_label: Column name for the proposed system.

    Returns:
        One row per KPI with both values, the absolute change and the percentage
        improvement (positive = the AI system is better).
    """
    rows = []
    for spec in METRIC_SPECS:
        base = baseline_kpis.get(spec.key, float("nan"))
        ai = ai_kpis.get(spec.key, float("nan"))
        rows.append(
            {
                "metric": spec.label,
                "key": spec.key,
                "unit": spec.unit,
                baseline_label: round(base, spec.precision),
                ai_label: round(ai, spec.precision),
                "absolute_change": round(ai - base, spec.precision),
                "improvement_pct": round(percentage_improvement(base, ai, spec.direction), 1),
                "better_when": spec.direction,
            }
        )
    return pd.DataFrame(rows)


def summarise_headline(comparison: pd.DataFrame, ai_label: str = "AI forecast + inventory policy") -> dict[str, float]:
    """Extract the handful of figures quoted in the dissertation abstract."""
    indexed = comparison.set_index("key")
    keys = ["stockout_units", "waste_units", "unit_service_level", "total_cost", "average_inventory_units"]
    return {
        key: float(indexed.loc[key, "improvement_pct"])
        for key in keys
        if key in indexed.index
    }


def daily_totals(daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-SKU-day records to a site-level daily series for charting."""
    return (
        daily.groupby("date", observed=True)
        .agg(
            demand=("demand", "sum"),
            served=("served", "sum"),
            stockout_units=("stockout_units", "sum"),
            waste_units=("expired_units", "sum"),
            closing_stock=("closing_stock", "sum"),
            inventory_value=("inventory_value", "sum"),
            total_cost=("total_cost", "sum"),
        )
        .reset_index()
    )


def metric_label(key: str) -> str:
    """Display label for a KPI key."""
    spec = _SPEC_BY_KEY.get(key)
    return spec.label if spec else key.replace("_", " ").capitalize()
