"""Forecast accuracy metrics.

The three metrics reported in the AE1 progress report are preserved exactly:

sMAPE
    Symmetric mean absolute percentage error, bounded and defined when demand is zero
    (the zero/zero case contributes 0), which matters for intermittent SKUs.
MASE
    Mean absolute scaled error against an in-sample seasonal naive forecast. Values
    below 1 beat that benchmark.
WAPE
    Weighted absolute percentage error - total absolute error over total demand. This is
    the metric that best reflects operational impact, because it weights busy days most.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def smape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Symmetric mean absolute percentage error, in percent."""
    y = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    _check_lengths(y, f)
    denom = np.abs(y) + np.abs(f)
    ratio = np.divide(2 * np.abs(f - y), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(np.mean(ratio) * 100)


def wape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Weighted absolute percentage error, in percent. NaN when total demand is zero."""
    y = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    _check_lengths(y, f)
    total = np.abs(y).sum()
    return float(np.abs(f - y).sum() / total * 100) if total > 0 else float("nan")


def mase(
    actual: Sequence[float],
    forecast: Sequence[float],
    insample: Sequence[float],
    seasonal_period: int = 7,
) -> float:
    """Mean absolute scaled error against an in-sample seasonal naive benchmark.

    Args:
        actual: Realised demand over the evaluation window.
        forecast: Predicted demand over the same window.
        insample: Training-period demand used to scale the error.
        seasonal_period: Seasonal lag of the benchmark (7 for weekly).

    Returns:
        The scaled error, or NaN when the benchmark denominator is zero (a flat series).
    """
    y = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ins = np.asarray(insample, dtype=float)
    _check_lengths(y, f)
    if len(ins) <= seasonal_period:
        return float("nan")
    denom = float(np.mean(np.abs(ins[seasonal_period:] - ins[:-seasonal_period])))
    if denom <= 0:
        return float("nan")
    return float(np.mean(np.abs(f - y)) / denom)


def bias(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Mean forecast bias in units (positive = over-forecasting)."""
    y = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    _check_lengths(y, f)
    return float(np.mean(f - y))


def evaluate_forecast(
    actual: Sequence[float],
    forecast: Sequence[float],
    insample: Sequence[float],
    seasonal_period: int = 7,
) -> dict[str, float]:
    """Compute all reported accuracy metrics for one forecast window."""
    return {
        "sMAPE": smape(actual, forecast),
        "MASE": mase(actual, forecast, insample, seasonal_period),
        "WAPE": wape(actual, forecast),
        "bias": bias(actual, forecast),
    }


def summarise_metrics(records: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    """Average per-fold, per-SKU metrics into the headline comparison table.

    Args:
        records: Long frame with one row per (fold, SKU, model) and metric columns.
        by: Grouping column, normally ``"model"`` or ``"sku"``.

    Returns:
        Mean of each metric per group, rounded to two decimals.
    """
    metric_cols = [c for c in ("sMAPE", "MASE", "WAPE", "bias") if c in records.columns]
    if not metric_cols:
        raise ValueError(f"No metric columns found in {sorted(records.columns)}.")
    return records.groupby(by, observed=True)[metric_cols].mean().round(2)


def improvement_vs_benchmark(
    table: pd.DataFrame,
    benchmark: str = "Seasonal Naive",
    metric: str = "sMAPE",
) -> pd.Series:
    """Percentage change in ``metric`` relative to a benchmark row (negative = better)."""
    if benchmark not in table.index:
        raise ValueError(f"Benchmark '{benchmark}' not present in {list(table.index)}.")
    base = table.loc[benchmark, metric]
    return ((table[metric] - base) / base * 100).round(1)


def _check_lengths(y: np.ndarray, f: np.ndarray) -> None:
    if y.shape != f.shape:
        raise ValueError(f"Actual and forecast must have the same shape, got {y.shape} and {f.shape}.")
    if y.size == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation window.")
