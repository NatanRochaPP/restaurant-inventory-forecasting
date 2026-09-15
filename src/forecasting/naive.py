"""Naive forecast: carry the last observed value forward."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastResult, UnivariateForecastModel


class NaiveForecast(UnivariateForecastModel):
    """Repeat the most recent observation across the horizon.

    Retained as the weakest reference point in the model comparison.
    """

    name = "Naive"

    def fit(self, history: pd.DataFrame) -> "NaiveForecast":
        self.sku_, self.series_ = self._extract_series(history)
        values = self.series_.to_numpy()
        self.last_value_ = float(values[-1])
        self.sigma_ = self._residual_sigma(np.diff(values))
        self._fitted = True
        return self

    def predict(self, future: pd.DataFrame) -> ForecastResult:
        self._check_fitted()
        dates = self._future_dates(future)
        point = np.full(len(dates), self.last_value_, dtype=float)
        return self._result(dates, point, {"last_value": self.last_value_})
