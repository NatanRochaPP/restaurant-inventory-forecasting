"""Croston's method and the bias-corrected SBA variant for intermittent demand.

Slow-moving restaurant SKUs (truffle oil, vegan cheese, seasonal berries) have many
zero-demand days. Fitting a smooth model to them produces a forecast that is confidently
wrong on both the zero days and the demand days. Croston's method instead smooths two
separate series - the size of non-zero demand, and the interval between demand events -
and forecasts the demand *rate*.

Classic Croston is known to be positively biased; the Syntetos-Boylan Approximation
(SBA) multiplies the rate by ``1 - alpha / 2`` to correct it. Both are provided so the
dissertation can report the comparison.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastResult, UnivariateForecastModel

logger = logging.getLogger(__name__)


class CrostonForecast(UnivariateForecastModel):
    """Croston-family forecast for intermittent demand.

    Args:
        alpha: Smoothing constant applied to both the demand size and interval series.
        variant: ``"classic"`` for original Croston, ``"sba"`` for the bias-corrected
            Syntetos-Boylan Approximation.
        service_level: Coverage of the reported prediction interval.
    """

    def __init__(self, alpha: float = 0.1, variant: str = "sba", service_level: float = 0.95) -> None:
        super().__init__(service_level=service_level)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Croston alpha must be in (0, 1], got {alpha}.")
        if variant not in {"classic", "sba"}:
            raise ValueError(f"Croston variant must be 'classic' or 'sba', got '{variant}'.")
        self.alpha = float(alpha)
        self.variant = variant

    @property
    def name(self) -> str:  # type: ignore[override]
        return "Croston (SBA)" if self.variant == "sba" else "Croston"

    def fit(self, history: pd.DataFrame) -> "CrostonForecast":
        self.sku_, self.series_ = self._extract_series(history)
        values = self.series_.to_numpy(dtype=float)
        nonzero_idx = np.flatnonzero(values > 0)

        if nonzero_idx.size == 0:
            logger.warning("SKU '%s' has no non-zero demand in history; forecasting zero.", self.sku_)
            self.rate_ = 0.0
            self.size_ = 0.0
            self.interval_ = float(len(values))
            self.sigma_ = 0.0
            self._fitted = True
            return self

        sizes = values[nonzero_idx]
        # Interval between demand events; the first event is dated from the series start.
        intervals = np.diff(np.concatenate(([-1], nonzero_idx))).astype(float)

        z = float(sizes[0])       # smoothed demand size
        p = float(intervals[0])   # smoothed inter-demand interval
        a = self.alpha
        for size, interval in zip(sizes[1:], intervals[1:]):
            z += a * (size - z)
            p += a * (interval - p)

        p = max(p, 1.0)
        rate = z / p
        if self.variant == "sba":
            rate *= 1.0 - a / 2.0

        self.size_ = z
        self.interval_ = p
        self.rate_ = float(max(rate, 0.0))
        # Uncertainty comes from the spread of actual daily demand around the flat rate.
        self.sigma_ = self._residual_sigma(values - self.rate_)
        self._fitted = True
        return self

    def predict(self, future: pd.DataFrame) -> ForecastResult:
        self._check_fitted()
        dates = self._future_dates(future)
        point = np.full(len(dates), self.rate_, dtype=float)
        return self._result(
            dates,
            point,
            {
                "variant": self.variant,
                "alpha": self.alpha,
                "smoothed_demand_size": float(self.size_),
                "smoothed_interval_days": float(self.interval_),
                "demand_rate_per_day": self.rate_,
            },
        )


def zero_demand_share(values: pd.Series | np.ndarray) -> float:
    """Share of days with zero demand - the intermittency measure used for routing."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr == 0))
