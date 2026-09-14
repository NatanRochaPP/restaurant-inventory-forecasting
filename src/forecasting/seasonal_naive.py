"""Seasonal naive forecast: repeat the most recent complete weekly cycle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastResult, UnivariateForecastModel


class SeasonalNaiveForecast(UnivariateForecastModel):
    """Forecast day ``t`` with the observation from ``t - seasonal_period``.

    This is the benchmark the dissertation reports improvements against, and the same
    method that defines the MASE denominator.

    Args:
        seasonal_period: Length of the seasonal cycle in days (7 for weekly).
        service_level: Coverage of the reported prediction interval.
    """

    name = "Seasonal Naive"

    def __init__(self, seasonal_period: int = 7, service_level: float = 0.95) -> None:
        super().__init__(service_level=service_level)
        if seasonal_period < 1:
            raise ValueError("seasonal_period must be >= 1.")
        self.seasonal_period = int(seasonal_period)

    def fit(self, history: pd.DataFrame) -> "SeasonalNaiveForecast":
        self.sku_, self.series_ = self._extract_series(history)
        values = self.series_.to_numpy()
        m = self.seasonal_period
        if len(values) < m:
            raise ValueError(
                f"SKU '{self.sku_}' has {len(values)} observations but the seasonal "
                f"naive model needs at least {m}."
            )
        self.last_cycle_ = values[-m:].astype(float)
        self.last_date_ = self.series_.index[-1]
        self.sigma_ = self._residual_sigma(values[m:] - values[:-m])
        self._fitted = True
        return self

    def predict(self, future: pd.DataFrame) -> ForecastResult:
        self._check_fitted()
        dates = self._future_dates(future)
        m = self.seasonal_period
        # Offset so that each forecast date maps onto the same weekday of the last cycle.
        offsets = np.array([(d - self.last_date_).days - 1 for d in dates], dtype=int)
        point = self.last_cycle_[offsets % m]
        return self._result(dates, point, {"seasonal_period": m})
