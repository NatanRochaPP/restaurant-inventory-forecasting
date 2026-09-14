"""Weather inputs for forecasting.

Temperature is *not* known in advance. Using the realised temperature of a future day
as a model input would flatter the results, because a real manager ordering on Monday
only has a weather forecast for Wednesday. :class:`WeatherProvider` therefore degrades
future temperatures with noise that grows with lead time, controlled from
``config.yaml``. Setting ``weather.perfect_foresight: true`` restores the optimistic
behaviour and is useful for quantifying how much of the accuracy depends on weather
information quality.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import WeatherConfig

logger = logging.getLogger(__name__)


class WeatherProvider:
    """Serves observed and forecast temperatures for a date range.

    Args:
        observed: Frame with ``date`` and ``temp_c`` columns (one row per date).
        config: Weather configuration controlling foresight and noise.
    """

    def __init__(self, observed: pd.DataFrame, config: WeatherConfig) -> None:
        if "date" not in observed.columns or "temp_c" not in observed.columns:
            raise ValueError("Weather observations require 'date' and 'temp_c' columns.")
        series = (
            observed[["date", "temp_c"]]
            .drop_duplicates(subset="date")
            .set_index("date")["temp_c"]
            .astype(float)
            .sort_index()
        )
        self._observed = series
        self._config = config
        self._climatology = float(series.mean())

    @property
    def observed(self) -> pd.Series:
        """The realised daily mean temperature series."""
        return self._observed

    def temperature_forecast(self, as_of: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.Series:
        """Return the temperature a manager would have for ``dates`` on ``as_of``.

        Args:
            as_of: The decision date; lead time is measured from here.
            dates: The dates to describe.

        Returns:
            A temperature series indexed by ``dates``. Dates before ``as_of`` return the
            observed value. Later dates return the observed value perturbed by
            lead-time-dependent noise, or climatology if no observation exists.

        Notes:
            The noise is deterministic given ``as_of`` and the configured seed, so
            repeated calls (for example a Streamlit rerun) produce identical values.
        """
        as_of = pd.Timestamp(as_of)
        dates = pd.DatetimeIndex(dates)
        truth = self._observed.reindex(dates)
        truth = truth.fillna(self._climatology)

        if self._config.perfect_foresight:
            return pd.Series(truth.to_numpy(), index=dates, name="temp_c")

        lead = np.maximum((dates - as_of).days.to_numpy(), 0)
        std = self._config.forecast_noise_std_c + self._config.noise_growth_per_day_c * np.maximum(lead - 1, 0)
        std = np.where(lead == 0, 0.0, std)
        rng = np.random.default_rng(self._config.random_seed + int(as_of.toordinal()))
        noisy = truth.to_numpy() + rng.normal(0.0, 1.0, size=len(dates)) * std
        return pd.Series(noisy, index=dates, name="temp_c")

    def historical_mean(self) -> float:
        """Long-run mean temperature, used as the fallback for unknown dates."""
        return self._climatology
