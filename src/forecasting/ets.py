"""Exponential smoothing (Holt-Winters) forecast."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.forecasting.base import ForecastResult, UnivariateForecastModel

logger = logging.getLogger(__name__)


class ETSForecast(UnivariateForecastModel):
    """Additive Holt-Winters with weekly seasonality.

    Matches the AE1 baseline configuration (additive trend, additive seasonality,
    estimated initialisation). If the optimiser fails - which happens on short or
    highly intermittent series - the model falls back to a seasonal naive forecast and
    records that fact in the result metadata rather than raising.

    Args:
        seasonal_period: Seasonal cycle length in days.
        trend: statsmodels trend specification, or ``None`` to disable.
        service_level: Coverage of the reported prediction interval.
    """

    name = "ETS (Holt-Winters)"

    def __init__(
        self,
        seasonal_period: int = 7,
        trend: str | None = "add",
        service_level: float = 0.95,
    ) -> None:
        super().__init__(service_level=service_level)
        self.seasonal_period = int(seasonal_period)
        self.trend = trend
        self._fallback = False

    def fit(self, history: pd.DataFrame) -> "ETSForecast":
        self.sku_, self.series_ = self._extract_series(history)
        values = self.series_.to_numpy(dtype=float)
        m = self.seasonal_period
        if len(values) < 2 * m:
            raise ValueError(
                f"SKU '{self.sku_}' needs at least {2 * m} observations for a seasonal "
                f"ETS fit but has {len(values)}."
            )
        self._fallback = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_ = ExponentialSmoothing(
                    values,
                    trend=self.trend,
                    seasonal="add",
                    seasonal_periods=m,
                    initialization_method="estimated",
                ).fit()
            self.sigma_ = self._residual_sigma(np.asarray(self.model_.resid, dtype=float))
        except Exception as exc:  # statsmodels raises a variety of optimiser errors
            logger.warning("ETS fit failed for SKU '%s' (%s); falling back to seasonal naive.", self.sku_, exc)
            self._fallback = True
            self.model_ = None
            self.last_cycle_ = values[-m:]
            self.sigma_ = self._residual_sigma(values[m:] - values[:-m])
        self.last_date_ = self.series_.index[-1]
        self._fitted = True
        return self

    def predict(self, future: pd.DataFrame) -> ForecastResult:
        self._check_fitted()
        dates = self._future_dates(future)
        h = len(dates)
        if self._fallback:
            offsets = np.array([(d - self.last_date_).days - 1 for d in dates], dtype=int)
            point = self.last_cycle_[offsets % self.seasonal_period]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                point = np.asarray(self.model_.forecast(h), dtype=float)
        return self._result(dates, point, {"fallback_to_seasonal_naive": self._fallback})
