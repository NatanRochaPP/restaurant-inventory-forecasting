"""Ordering strategies compared by the simulator.

A strategy answers one question on a review day: *given what was knowable at this moment,
how much should be ordered?* Two are implemented:

:class:`ForecastDrivenStrategy`
    The proposed system - a per-SKU forecast (gradient boosting or Croston, chosen by
    segment) feeding a base-stock or (s, S) policy, with safety stock sized from forecast
    error and an explicit shelf-life cap.

:class:`ParLevelStrategy`
    The manual baseline - a par level from trailing average usage plus a flat percentage
    buffer, with no view of the calendar, weather or promotions.

Both receive the same stock position, the same supplier calendar and the same constraint
machinery, so the difference in simulated waste and stockouts is attributable to the
demand signal rather than to the mechanics around it. Both read history exclusively
through :func:`~src.data.preprocessing.history_as_of`, so neither can see the demand it is
ordering for.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.config import AppConfig, SkuEconomics
from src.data.preprocessing import history_as_of
from src.forecasting.service import ForecastService
from src.inventory.ordering import OrderRecommendation, recommend_order
from src.inventory.policies import InventoryPolicy, ParLevelPolicy, PolicyInputs, build_policy
from src.scenarios import BASE_CASE, Scenario
from src.segmentation import SkuSegment, service_level_for_segment

logger = logging.getLogger(__name__)


class OrderingStrategy(ABC):
    """Base class for the ordering approaches compared in the simulation."""

    #: Label used in results tables and charts.
    name: str = "strategy"

    @abstractmethod
    def recommend(
        self,
        sku: str,
        as_of: pd.Timestamp,
        on_hand: float,
        on_order: float,
        economics: SkuEconomics,
        review_period_days: int,
        lead_time_days: int,
        backorders: float = 0.0,
    ) -> OrderRecommendation:
        """Produce a recommendation for one SKU on one review day."""

    def prepare(self, as_of: pd.Timestamp, skus: list[str]) -> None:
        """Optional hook called once per review day before per-SKU recommendations.

        Used by the forecast-driven strategy to forecast every SKU in a single model
        pass instead of one call per SKU.
        """
        return None


class ForecastDrivenStrategy(OrderingStrategy):
    """Forecast-driven replenishment: the system under evaluation.

    Args:
        service: Forecasting service supplying per-SKU demand forecasts.
        config: Application configuration.
        policy: Inventory policy; built from configuration when omitted.
        segments: Optional ABC/XYZ segments used to differentiate service levels.
        scenario: What-if overrides applied to the forecast.
    """

    name = "AI forecast + inventory policy"

    def __init__(
        self,
        service: ForecastService,
        config: AppConfig,
        policy: InventoryPolicy | None = None,
        segments: dict[str, SkuSegment] | None = None,
        scenario: Scenario = BASE_CASE,
    ) -> None:
        self.service = service
        self.config = config
        self.policy = policy or build_policy(config.inventory.policy, config.inventory.s_S_reorder_multiplier)
        self.segments = segments or {}
        self.scenario = scenario
        self._forecasts: dict[str, object] = {}
        self._forecast_date: pd.Timestamp | None = None

    def prepare(self, as_of: pd.Timestamp, skus: list[str]) -> None:
        """Forecast every SKU for this review day in one pass."""
        as_of = pd.Timestamp(as_of)
        horizon = max(
            self.config.forecast.horizon_days,
            self.config.supplier.lead_time_days + self.config.inventory.review_period_days,
        )
        self._forecasts = self.service.forecast(as_of, skus=skus, horizon=horizon, scenario=self.scenario)
        self._forecast_date = as_of

    def recommend(
        self,
        sku: str,
        as_of: pd.Timestamp,
        on_hand: float,
        on_order: float,
        economics: SkuEconomics,
        review_period_days: int,
        lead_time_days: int,
        backorders: float = 0.0,
    ) -> OrderRecommendation:
        as_of = pd.Timestamp(as_of)
        if self._forecast_date != as_of or sku not in self._forecasts:
            self.prepare(as_of, [sku])
        forecast = self._forecasts[sku]

        service_level = self.config.inventory.service_level
        if self.scenario.service_level is not None:
            service_level = self.scenario.service_level
        elif sku in self.segments:
            service_level = service_level_for_segment(self.segments[sku], service_level)

        inputs = PolicyInputs(
            sku=sku,
            daily_forecast=forecast.predicted,
            sigma_daily=forecast.sigma,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            lead_time_days=lead_time_days,
            review_period_days=review_period_days,
            service_level=service_level,
        )
        return recommend_order(
            inputs,
            economics,
            self.config,
            as_of=as_of,
            policy=self.policy,
            model_name=forecast.model_name,
            metadata={
                "strategy": self.name,
                "segment": self.segments[sku].segment if sku in self.segments else None,
                "forecast_lower": [float(v) for v in (forecast.lower if forecast.lower is not None else [])],
                "forecast_upper": [float(v) for v in (forecast.upper if forecast.upper is not None else [])],
            },
        )


class ParLevelStrategy(OrderingStrategy):
    """Manual par-level replenishment: the baseline representing current practice.

    The par level is the trailing mean daily usage over ``baseline.lookback_days``,
    multiplied by the cover required and inflated by a flat manager buffer. No calendar,
    weather or promotion information is used, and safety stock is not sized from forecast
    error.

    Args:
        sales: Full sales history; sliced at each decision date so the strategy only ever
            sees the past.
        config: Application configuration supplying the baseline parameters.
        apply_shelf_life_cap: Whether to apply the perishability cap. Defaults to the
            configured ``baseline.apply_shelf_life_cap`` (False), because an
            average-usage par level embodies no explicit shelf-life reasoning. This is
            the one deliberate asymmetry between the two policies: it lets the baseline
            buy availability with waste at high buffers, which the forecast-driven policy
            refuses to do. Setting it to True removes the asymmetry for a sensitivity run.
    """

    name = "Manual par-level baseline"

    def __init__(
        self,
        sales: pd.DataFrame,
        config: AppConfig,
        apply_shelf_life_cap: bool | None = None,
    ) -> None:
        self.sales = sales
        self.config = config
        self.apply_shelf_life_cap = (
            config.baseline.apply_shelf_life_cap if apply_shelf_life_cap is None else apply_shelf_life_cap
        )
        self.policy = ParLevelPolicy(config.baseline.buffer_pct, config.baseline.cover_days)
        self._usage_cache: dict[tuple[str, pd.Timestamp], float] = {}

    def average_daily_usage(self, sku: str, as_of: pd.Timestamp) -> float:
        """Mean daily demand over the lookback window, using only past observations."""
        key = (sku, pd.Timestamp(as_of))
        if key in self._usage_cache:
            return self._usage_cache[key]
        history = history_as_of(self.sales, as_of, sku=sku)
        window_start = pd.Timestamp(as_of) - pd.Timedelta(days=self.config.baseline.lookback_days)
        window = history[history["date"] >= window_start]
        if window.empty:
            window = history
        usage = float(window["units_sold"].mean()) if not window.empty else 0.0
        self._usage_cache[key] = usage
        return usage

    def recommend(
        self,
        sku: str,
        as_of: pd.Timestamp,
        on_hand: float,
        on_order: float,
        economics: SkuEconomics,
        review_period_days: int,
        lead_time_days: int,
        backorders: float = 0.0,
    ) -> OrderRecommendation:
        as_of = pd.Timestamp(as_of)
        usage = self.average_daily_usage(sku, as_of)
        horizon = max(self.config.forecast.horizon_days, lead_time_days + review_period_days)
        # The manager's implicit "forecast" is a flat repetition of recent average usage.
        flat_forecast = np.full(horizon, max(usage, 0.0), dtype=float)

        inputs = PolicyInputs(
            sku=sku,
            daily_forecast=flat_forecast,
            sigma_daily=0.0,  # the flat percentage buffer stands in for uncertainty
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            lead_time_days=lead_time_days,
            review_period_days=review_period_days,
            service_level=self.config.inventory.service_level,
        )
        return recommend_order(
            inputs,
            economics,
            self.config,
            as_of=as_of,
            policy=self.policy,
            model_name="Trailing average usage",
            metadata={
                "strategy": self.name,
                "lookback_days": self.config.baseline.lookback_days,
                "average_daily_usage": usage,
            },
            enforce_shelf_life=self.apply_shelf_life_cap,
        )
