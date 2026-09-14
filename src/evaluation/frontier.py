"""Service-level frontier: comparing ordering policies fairly.

Comparing a single AI run against a single manual run is not sound evidence. Any
replenishment policy can trade waste for availability simply by holding more stock, so
two policies observed at different operating points cannot be ranked: the manual baseline
in this project happens to run at a ~99% fill rate because of its flat 20% buffer, while
the AI policy targets the configured 95%.

This module sweeps both policies across their tuning parameter - service level for the
forecast-driven policy, buffer percentage for the manual one - and records where each
lands. That produces a frontier of (service level, waste) pairs per policy, which
supports two defensible claims:

* at matched service level, one policy wastes less;
* at matched waste, one policy serves more demand.

Both are reported by :func:`matched_comparison`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.evaluation.inventory_metrics import compute_kpis
from src.forecasting.service import ForecastService
from src.inventory.simulator import InventorySimulator
from src.inventory.strategies import ForecastDrivenStrategy, ParLevelStrategy
from src.segmentation import SkuSegment

logger = logging.getLogger(__name__)

AI_LABEL = "AI forecast + inventory policy"
BASELINE_LABEL = "Manual par-level baseline"

#: Columns of :func:`matched_comparison`, declared so that an empty result (the policies'
#: ranges do not overlap) still has a usable shape for callers and charts.
MATCHED_COLUMNS: tuple[str, ...] = (
    "reference_setting", "matched_service_level_pct", "matched_ai_service_level_setting",
    "waste_units_baseline", "waste_units_ai", "waste_reduction_pct",
    "total_cost_baseline", "total_cost_ai", "cost_reduction_pct",
    "avg_inventory_baseline", "avg_inventory_ai", "inventory_reduction_pct",
)

#: Columns of :func:`service_level_at_matched_waste`.
MATCHED_WASTE_COLUMNS: tuple[str, ...] = (
    "reference_setting", "matched_waste_units", "service_level_baseline",
    "service_level_ai", "service_level_gain_pp",
)


@dataclass
class FrontierPoint:
    """One simulation run at a given tuning setting."""

    strategy: str
    setting_name: str
    setting_value: float
    kpis: dict[str, float]

    def to_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "setting_name": self.setting_name,
            "setting_value": self.setting_value,
            **{k: round(v, 3) for k, v in self.kpis.items()},
        }


def run_frontier(
    sales: pd.DataFrame,
    config: AppConfig,
    *,
    service_levels: list[float] | None = None,
    buffer_pcts: list[float] | None = None,
    forecast_service: ForecastService | None = None,
    segments: dict[str, SkuSegment] | None = None,
    sku_master: pd.DataFrame | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    differentiate_service_level: bool = False,
) -> pd.DataFrame:
    """Simulate both policies across a range of operating points.

    Args:
        sales: Full sales history.
        config: Application configuration.
        service_levels: Service-level targets to sweep for the AI policy.
        buffer_pcts: Buffer percentages to sweep for the manual baseline.
        forecast_service: Reuse an existing service so forecasts are computed once and
            memoised across every point on the sweep.
        segments: ABC/XYZ segments; when supplied with ``differentiate_service_level``
            the AI policy varies its target by ABC class.
        sku_master: Pre-built SKU master.
        start: Simulation start; defaults to the configured window.
        end: Simulation end; defaults to the configured window.
        differentiate_service_level: Whether to apply the ABC service-level adjustment.
            Disabled during a sweep so that the swept value is the only thing changing.

    Returns:
        One row per (policy, setting) with the full KPI set.
    """
    service_levels = service_levels or [0.90, 0.95, 0.975, 0.99, 0.995]
    buffer_pcts = buffer_pcts or [0.0, 0.10, 0.20, 0.30, 0.50]
    service = forecast_service or ForecastService(sales, config)
    points: list[FrontierPoint] = []

    for buffer_pct in buffer_pcts:
        run_config = config.model_copy(
            update={"baseline": config.baseline.model_copy(update={"buffer_pct": buffer_pct})}
        )
        strategy = ParLevelStrategy(sales, run_config)
        result = InventorySimulator(sales, run_config, strategy, sku_master=sku_master).run(start, end)
        points.append(
            FrontierPoint(BASELINE_LABEL, "buffer_pct", buffer_pct, compute_kpis(result.daily, result.orders))
        )
        logger.info("Frontier: baseline buffer %.0f%% complete", buffer_pct * 100)

    for level in service_levels:
        run_config = config.model_copy(
            update={"inventory": config.inventory.model_copy(update={"service_level": level})}
        )
        strategy = ForecastDrivenStrategy(
            service, run_config, segments=segments if differentiate_service_level else None
        )
        result = InventorySimulator(sales, run_config, strategy, sku_master=sku_master).run(start, end)
        points.append(
            FrontierPoint(AI_LABEL, "service_level", level, compute_kpis(result.daily, result.orders))
        )
        logger.info("Frontier: AI service level %.1f%% complete", level * 100)

    return pd.DataFrame([p.to_row() for p in points])


def _interpolate(frame: pd.DataFrame, x_col: str, y_col: str, x_value: float) -> float:
    """Linearly interpolate ``y`` at ``x_value`` along a monotonically sorted frame.

    Returns NaN when ``x_value`` lies outside the swept range, so that a comparison is
    never reported by extrapolating beyond what was actually simulated.
    """
    ordered = frame.sort_values(x_col)
    xs = ordered[x_col].to_numpy(dtype=float)
    ys = ordered[y_col].to_numpy(dtype=float)
    if len(xs) < 2 or x_value < xs.min() or x_value > xs.max():
        return float("nan")
    return float(np.interp(x_value, xs, ys))


def matched_comparison(
    frontier: pd.DataFrame,
    reference_strategy: str = BASELINE_LABEL,
    comparison_strategy: str = AI_LABEL,
) -> pd.DataFrame:
    """Compare the two policies at matched operating points.

    For each simulated point of the reference (manual) policy, the comparison (AI) policy
    is interpolated along its own frontier to the same service level, and the difference
    in waste, stockouts and cost is reported.

    Args:
        frontier: Output of :func:`run_frontier`.
        reference_strategy: The policy whose operating points define the comparison.
        comparison_strategy: The policy interpolated onto them.

    Returns:
        One row per reference operating point. ``waste_reduction_pct`` is positive when
        the comparison policy wastes less at the same service level. Rows where the
        service level falls outside the comparison policy's swept range are dropped.
    """
    reference = frontier[frontier["strategy"] == reference_strategy]
    comparison = frontier[frontier["strategy"] == comparison_strategy]
    if reference.empty or comparison.empty:
        raise ValueError(
            f"Frontier must contain both '{reference_strategy}' and '{comparison_strategy}'. "
            f"Found: {sorted(frontier['strategy'].unique())}"
        )

    rows = []
    for point in reference.itertuples(index=False):
        service = float(point.unit_service_level)
        matched_waste = _interpolate(comparison, "unit_service_level", "waste_units", service)
        matched_cost = _interpolate(comparison, "unit_service_level", "total_cost", service)
        matched_stock = _interpolate(comparison, "unit_service_level", "average_inventory_units", service)
        matched_setting = _interpolate(comparison, "unit_service_level", "setting_value", service)
        if np.isnan(matched_waste):
            logger.debug("Service level %.2f%% is outside the AI sweep range; skipped.", service)
            continue
        rows.append(
            {
                "reference_setting": point.setting_value,
                "matched_service_level_pct": round(service, 2),
                "matched_ai_service_level_setting": round(matched_setting, 4),
                "waste_units_baseline": round(float(point.waste_units), 1),
                "waste_units_ai": round(matched_waste, 1),
                "waste_reduction_pct": round(
                    (point.waste_units - matched_waste) / point.waste_units * 100, 1
                ) if point.waste_units > 0 else float("nan"),
                "total_cost_baseline": round(float(point.total_cost), 2),
                "total_cost_ai": round(matched_cost, 2),
                "cost_reduction_pct": round(
                    (point.total_cost - matched_cost) / point.total_cost * 100, 1
                ) if point.total_cost > 0 else float("nan"),
                "avg_inventory_baseline": round(float(point.average_inventory_units), 1),
                "avg_inventory_ai": round(matched_stock, 1),
                "inventory_reduction_pct": round(
                    (point.average_inventory_units - matched_stock) / point.average_inventory_units * 100, 1
                ) if point.average_inventory_units > 0 else float("nan"),
            }
        )
    if not rows:
        logger.info(
            "No overlap between the two policies' service levels; widen the sweep to compare them."
        )
        return pd.DataFrame(columns=list(MATCHED_COLUMNS))
    return pd.DataFrame(rows)


def service_level_at_matched_waste(
    frontier: pd.DataFrame,
    reference_strategy: str = BASELINE_LABEL,
    comparison_strategy: str = AI_LABEL,
) -> pd.DataFrame:
    """The mirror comparison: service level achieved at the same level of waste."""
    reference = frontier[frontier["strategy"] == reference_strategy]
    comparison = frontier[frontier["strategy"] == comparison_strategy]
    rows = []
    for point in reference.itertuples(index=False):
        matched_service = _interpolate(comparison, "waste_units", "unit_service_level", float(point.waste_units))
        if np.isnan(matched_service):
            continue
        rows.append(
            {
                "reference_setting": point.setting_value,
                "matched_waste_units": round(float(point.waste_units), 1),
                "service_level_baseline": round(float(point.unit_service_level), 2),
                "service_level_ai": round(matched_service, 2),
                "service_level_gain_pp": round(matched_service - float(point.unit_service_level), 2),
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(MATCHED_WASTE_COLUMNS))
    return pd.DataFrame(rows)
