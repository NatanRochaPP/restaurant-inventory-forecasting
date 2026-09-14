"""ABC/XYZ SKU segmentation.

ABC ranks SKUs by economic importance (annualised purchase value), so that management
attention and service levels can be concentrated where the money is. XYZ ranks them by
how predictable their demand is, using the coefficient of variation of daily demand plus
the share of zero-demand days.

The combined segment drives two decisions in this system:

* which forecasting model is appropriate (see
  :mod:`src.forecasting.model_selection`) - a smooth AX SKU suits a feature-based ML
  model, while an intermittent CZ SKU needs a Croston-family method;
* how much safety stock is justified, since holding cover for a cheap, stable item costs
  very little while holding it for an expensive, erratic one is what drives waste.

Segmentation is always computed from history strictly before ``as_of`` so that it can be
recomputed inside a backtest or simulation without leaking future demand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.data.preprocessing import history_as_of

logger = logging.getLogger(__name__)

ABC_LABELS = ("A", "B", "C")
XYZ_LABELS = ("X", "Y", "Z")


@dataclass(frozen=True)
class SkuSegment:
    """Segmentation outcome for one SKU."""

    sku: str
    abc: str
    xyz: str
    annual_value: float
    value_share: float
    cumulative_value_share: float
    mean_daily_demand: float
    demand_cv: float
    zero_day_share: float
    n_observations: int

    @property
    def segment(self) -> str:
        """Combined label, e.g. ``"AX"``."""
        return f"{self.abc}{self.xyz}"

    @property
    def is_intermittent(self) -> bool:
        """Whether demand is sparse enough to need a Croston-family model."""
        return self.zero_day_share > 0.0 and self.xyz == "Z"

    def describe(self) -> str:
        """One-line manager-facing description of the segment."""
        value = {"A": "high-value", "B": "mid-value", "C": "low-value"}[self.abc]
        stability = {
            "X": "stable, predictable demand",
            "Y": "moderately variable demand",
            "Z": "erratic or intermittent demand",
        }[self.xyz]
        return f"{self.segment}: {value} item with {stability}"


def segment_skus(
    sales: pd.DataFrame,
    sku_master: pd.DataFrame,
    config: AppConfig,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Classify every SKU on the ABC and XYZ dimensions.

    Args:
        sales: Tidy sales history.
        sku_master: Output of :func:`~src.data.loader.build_sku_master`, supplying unit
            costs for the value calculation.
        config: Application configuration (thresholds and lookback window).
        as_of: Decision date. Only demand strictly before this date is used. Defaults to
            the day after the last observation, i.e. the whole dataset.

    Returns:
        One row per SKU with the ABC class, XYZ class, combined segment and the
        statistics behind them, sorted by descending annual value.

    Raises:
        ValueError: If the lookback window contains no observations.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(sales["date"].max()) + pd.Timedelta(days=1)
    history = history_as_of(sales, as_of)
    if history.empty:
        raise ValueError(f"No sales history available before {as_of.date()} to segment SKUs.")

    window_start = as_of - pd.Timedelta(days=config.segmentation.lookback_days)
    window = history[history["date"] >= window_start]
    if window.empty:
        logger.warning(
            "Segmentation lookback window (%s to %s) is empty; using all available history.",
            window_start.date(), as_of.date(),
        )
        window = history

    costs = sku_master.set_index("sku")["unit_cost"]
    stats = (
        window.groupby("sku", observed=True)["units_sold"]
        .agg(
            total_units="sum",
            mean_daily_demand="mean",
            std_daily_demand=lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
            zero_day_share=lambda s: float((s == 0).mean()),
            n_observations="size",
        )
        .reset_index()
    )
    stats["sku"] = stats["sku"].astype(str)
    stats["unit_cost"] = stats["sku"].map(costs).fillna(config.sku_defaults.unit_cost)

    days = max(int((as_of - max(window_start, window["date"].min())).days), 1)
    stats["annual_value"] = stats["total_units"] * stats["unit_cost"] * (365.0 / days)
    stats["demand_cv"] = np.where(
        stats["mean_daily_demand"] > 0,
        stats["std_daily_demand"] / stats["mean_daily_demand"].replace(0, np.nan),
        np.inf,
    )

    stats = stats.sort_values("annual_value", ascending=False).reset_index(drop=True)
    total_value = stats["annual_value"].sum()
    stats["value_share"] = stats["annual_value"] / total_value if total_value > 0 else 0.0
    stats["cumulative_value_share"] = stats["value_share"].cumsum()

    a_cut, b_cut = config.segmentation.abc_thresholds
    # The item that crosses a threshold belongs to the class it completes, so the A class
    # is the smallest set whose cumulative value reaches the threshold.
    prior = stats["cumulative_value_share"].shift(1).fillna(0.0)
    stats["abc"] = np.where(prior < a_cut, "A", np.where(prior < b_cut, "B", "C"))

    x_cut, y_cut = config.segmentation.xyz_thresholds
    stats["xyz"] = np.where(stats["demand_cv"] < x_cut, "X", np.where(stats["demand_cv"] < y_cut, "Y", "Z"))
    stats["segment"] = stats["abc"] + stats["xyz"]
    stats["as_of"] = as_of

    logger.info(
        "Segmented %d SKUs as of %s: %s",
        len(stats), as_of.date(), stats["segment"].value_counts().to_dict(),
    )
    return stats


def to_segments(frame: pd.DataFrame) -> dict[str, SkuSegment]:
    """Convert the segmentation frame into a lookup of :class:`SkuSegment` objects."""
    out: dict[str, SkuSegment] = {}
    for row in frame.itertuples(index=False):
        out[str(row.sku)] = SkuSegment(
            sku=str(row.sku),
            abc=str(row.abc),
            xyz=str(row.xyz),
            annual_value=float(row.annual_value),
            value_share=float(row.value_share),
            cumulative_value_share=float(row.cumulative_value_share),
            mean_daily_demand=float(row.mean_daily_demand),
            demand_cv=float(row.demand_cv),
            zero_day_share=float(row.zero_day_share),
            n_observations=int(row.n_observations),
        )
    return out


def service_level_for_segment(segment: SkuSegment, base_service_level: float) -> float:
    """Differentiate the service-level target by ABC class.

    High-value A items justify tighter availability, while C items are held to a lower
    target because the cover would otherwise be paid for in waste. The adjustment is
    deliberately small so that the headline configured level remains recognisable.

    Args:
        segment: The SKU's segmentation outcome.
        base_service_level: The configured target, e.g. 0.95.

    Returns:
        A service level in (0, 1).
    """
    adjustment = {"A": 0.02, "B": 0.0, "C": -0.03}[segment.abc]
    return float(np.clip(base_service_level + adjustment, 0.50, 0.995))
