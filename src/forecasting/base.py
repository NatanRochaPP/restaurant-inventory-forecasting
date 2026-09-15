"""Common forecasting interface.

Every model in this package implements :class:`ForecastModel` and returns a
:class:`ForecastResult`, so the inventory, explainability and dashboard layers can
consume any model without knowing which one produced the numbers.

``ForecastResult`` also carries ``sigma`` - the estimated one-day demand forecast error
- because the inventory policy sizes safety stock from forecast *uncertainty*, not from
raw demand variability. Keeping the two together prevents the two layers drifting apart.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def z_score(service_level: float) -> float:
    """Standard-normal quantile for a target service level.

    Args:
        service_level: Probability in (0, 1), e.g. 0.95.

    Raises:
        ValueError: If the service level is outside (0, 1).
    """
    if not 0.0 < service_level < 1.0:
        raise ValueError(f"Service level must be strictly between 0 and 1, got {service_level}.")
    return float(stats.norm.ppf(service_level))


@dataclass(frozen=True)
class ForecastResult:
    """A forecast for one SKU over a contiguous horizon.

    Attributes:
        sku: The product the forecast belongs to.
        model_name: Name of the model that produced it.
        dates: Forecast dates, ascending.
        predicted: Point forecast of daily demand, same length as ``dates``.
        lower: Lower bound of the prediction interval, or ``None``.
        upper: Upper bound of the prediction interval, or ``None``.
        sigma: Estimated standard deviation of the one-day-ahead forecast error.
        metadata: Model-specific diagnostics (fitted parameters, fallbacks used, ...).
    """

    sku: str
    model_name: str
    dates: pd.DatetimeIndex
    predicted: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    sigma: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.predicted):
            raise ValueError(
                f"Forecast for '{self.sku}' has {len(self.predicted)} values for "
                f"{len(self.dates)} dates."
            )
        if np.any(np.asarray(self.predicted) < 0):
            raise ValueError(f"Forecast for '{self.sku}' contains negative demand.")

    @property
    def horizon(self) -> int:
        """Number of days covered."""
        return len(self.dates)

    def total(self, days: int | None = None) -> float:
        """Total expected demand over the first ``days`` of the horizon."""
        n = self.horizon if days is None else min(int(days), self.horizon)
        return float(np.sum(self.predicted[:n]))

    def to_frame(self) -> pd.DataFrame:
        """Return the tidy representation used across the application and database."""
        return pd.DataFrame(
            {
                "sku": self.sku,
                "forecast_date": pd.DatetimeIndex(self.dates),
                "predicted_demand": np.asarray(self.predicted, dtype=float),
                "lower_bound": np.asarray(self.lower, dtype=float) if self.lower is not None else np.nan,
                "upper_bound": np.asarray(self.upper, dtype=float) if self.upper is not None else np.nan,
                "model": self.model_name,
            }
        )


class ForecastModel(ABC):
    """Abstract base class for all forecasting models.

    Subclasses implement :meth:`fit` on historical observations and :meth:`predict` on a
    frame of future rows. Models must never receive realised demand for the dates they
    are predicting.
    """

    #: Display name recorded on every forecast and in the model-selection audit trail.
    name: str = "base"
    #: Whether the model trains across all SKUs at once (a global model).
    is_global: bool = False

    def __init__(self) -> None:
        self._fitted: bool = False

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> "ForecastModel":
        """Fit the model on a tidy history frame (``date``, ``sku``, ``units_sold``)."""

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> ForecastResult:
        """Forecast the rows in ``future`` (``date``, ``sku`` and any features)."""

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predicting.")


class UnivariateForecastModel(ForecastModel):
    """Base class for models fitted to a single SKU's demand series.

    Handles the shared plumbing: extracting a single SKU, estimating the residual
    standard deviation, and turning a point forecast into a prediction interval.

    Args:
        service_level: Coverage of the prediction interval reported to the user.
    """

    def __init__(self, service_level: float = 0.95) -> None:
        super().__init__()
        self.service_level = service_level
        self._z = z_score(service_level)
        self.sku_: str | None = None
        self.series_: pd.Series | None = None
        self.sigma_: float = 0.0

    @staticmethod
    def _extract_series(history: pd.DataFrame) -> tuple[str, pd.Series]:
        """Pull a single SKU's demand series out of a tidy frame."""
        for col in ("date", "units_sold"):
            if col not in history.columns:
                raise ValueError(f"History frame requires a '{col}' column.")
        skus = history["sku"].unique() if "sku" in history.columns else ["unknown"]
        if len(skus) != 1:
            raise ValueError(
                f"Univariate models fit one SKU at a time but received {len(skus)}: "
                f"{sorted(map(str, skus))[:5]}."
            )
        series = (
            history.sort_values("date", kind="stable")
            .set_index("date")["units_sold"]
            .astype(float)
        )
        if series.empty:
            raise ValueError(f"No history supplied for SKU '{skus[0]}'.")
        return str(skus[0]), series

    def _future_dates(self, future: pd.DataFrame) -> pd.DatetimeIndex:
        if "date" not in future.columns:
            raise ValueError("Future frame requires a 'date' column.")
        if "units_sold" in future.columns:
            raise ValueError(
                "Future frame carries 'units_sold'; realised demand must not be visible "
                "when predicting."
            )
        if "sku" in future.columns and future["sku"].nunique() > 1:
            raise ValueError("Univariate models predict one SKU at a time.")
        return pd.DatetimeIndex(future.sort_values("date")["date"].to_numpy())

    def _interval(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric normal interval widening with the square root of lead time."""
        steps = np.arange(1, len(point) + 1, dtype=float)
        width = self._z * self.sigma_ * np.sqrt(steps)
        lower = np.clip(point - width, 0.0, None)
        upper = point + width
        return lower, upper

    def _result(
        self,
        dates: pd.DatetimeIndex,
        point: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> ForecastResult:
        point = np.clip(np.asarray(point, dtype=float), 0.0, None)
        lower, upper = self._interval(point)
        return ForecastResult(
            sku=self.sku_ or "unknown",
            model_name=self.name,
            dates=dates,
            predicted=point,
            lower=lower,
            upper=upper,
            sigma=float(self.sigma_),
            metadata=metadata or {},
        )

    @staticmethod
    def _residual_sigma(residuals: np.ndarray) -> float:
        """Robust-ish standard deviation of one-step-ahead residuals."""
        res = np.asarray(residuals, dtype=float)
        res = res[np.isfinite(res)]
        if res.size < 2:
            return 0.0
        return float(np.std(res, ddof=1))
