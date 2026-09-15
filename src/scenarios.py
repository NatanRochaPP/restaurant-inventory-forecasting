"""What-if scenario definitions.

A scenario is a set of controlled deviations from the observed conditions - warmer
weather, a promotion, a longer supplier lead time, a stricter service level. Scenarios
are applied to a *copy* of the feature frame and to the resulting forecast, and never
touch the trained model or any persisted state, so the dashboard can explore them freely
without corrupting the results the dissertation reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scenario:
    """Controlled overrides applied to a forecast and its order recommendation.

    Attributes:
        label: Name shown in the dashboard and recorded with any saved result.
        demand_uplift_pct: Multiplicative adjustment to the point forecast, in percent.
            Represents a manager's expectation (a local event, a coach party) that the
            model cannot know about.
        temperature_c: Absolute temperature override for the whole horizon.
        temperature_delta_c: Shift applied to the forecast temperature.
        promo: Force the promotion flag on (True) or off (False) across the horizon.
        service_level: Override the inventory service-level target.
        lead_time_days: Override the supplier lead time.
        current_stock: Override the on-hand stock used for the recommendation.
    """

    label: str = "Base case"
    demand_uplift_pct: float = 0.0
    temperature_c: float | None = None
    temperature_delta_c: float = 0.0
    promo: bool | None = None
    service_level: float | None = None
    lead_time_days: int | None = None
    current_stock: float | None = None

    @property
    def is_base_case(self) -> bool:
        """True when the scenario leaves every input unchanged."""
        return (
            self.demand_uplift_pct == 0.0
            and self.temperature_c is None
            and self.temperature_delta_c == 0.0
            and self.promo is None
            and self.service_level is None
            and self.lead_time_days is None
            and self.current_stock is None
        )

    def apply_to_features(self, future: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the future feature frame with weather and promo overrides."""
        if self.temperature_c is None and self.temperature_delta_c == 0.0 and self.promo is None:
            return future
        out = future.copy()
        if self.temperature_c is not None:
            out["temp_c"] = float(self.temperature_c)
        if self.temperature_delta_c:
            out["temp_c"] = out["temp_c"] + float(self.temperature_delta_c)
        if self.promo is not None:
            out["promo"] = int(bool(self.promo))
        return out

    def apply_to_forecast(self, result: ForecastResult) -> ForecastResult:
        """Scale a forecast (and its interval) by the manager's demand uplift."""
        if self.demand_uplift_pct == 0.0:
            return result
        factor = 1.0 + self.demand_uplift_pct / 100.0
        if factor < 0:
            raise ValueError(f"A demand uplift of {self.demand_uplift_pct}% implies negative demand.")
        return replace(
            result,
            predicted=np.clip(np.asarray(result.predicted) * factor, 0.0, None),
            lower=np.clip(np.asarray(result.lower) * factor, 0.0, None) if result.lower is not None else None,
            upper=np.clip(np.asarray(result.upper) * factor, 0.0, None) if result.upper is not None else None,
            sigma=result.sigma * abs(factor),
            metadata={**result.metadata, "scenario": self.label, "demand_uplift_pct": self.demand_uplift_pct},
        )

    def describe(self) -> list[str]:
        """Human-readable list of the changes this scenario makes."""
        changes: list[str] = []
        if self.demand_uplift_pct:
            changes.append(f"demand adjusted by {self.demand_uplift_pct:+.0f}%")
        if self.temperature_c is not None:
            changes.append(f"temperature fixed at {self.temperature_c:.0f}°C")
        if self.temperature_delta_c:
            changes.append(f"temperature shifted by {self.temperature_delta_c:+.0f}°C")
        if self.promo is not None:
            changes.append("promotion forced on" if self.promo else "promotion forced off")
        if self.service_level is not None:
            changes.append(f"service level set to {self.service_level:.0%}")
        if self.lead_time_days is not None:
            plural = "" if self.lead_time_days == 1 else "s"
            changes.append(f"lead time set to {self.lead_time_days} day{plural}")
        if self.current_stock is not None:
            changes.append(f"current stock set to {self.current_stock:.0f} units")
        return changes or ["no changes (base case)"]


BASE_CASE = Scenario()
