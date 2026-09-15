"""Forecasting service - the single entry point used by the simulator and dashboard.

The service owns the pieces that must not be duplicated at call sites:

* slicing history at the decision date (:func:`~src.data.preprocessing.history_as_of`);
* building leakage-free future feature rows;
* routing each SKU to its selected model;
* caching the global gradient-boosting fit and refitting it only on the configured
  cadence rather than every simulated day.

Training and inference stay separate: :meth:`ForecastService.forecast` never mutates a
persisted model artefact, and scenario overrides are applied to copies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.data.preprocessing import history_as_of, horizon_dates
from src.features.calendar import CLOSED_DAYS
from src.features.demand_features import SkuEncoder, build_future_features
from src.features.weather import WeatherProvider
from src.forecasting.base import ForecastResult
from src.forecasting.gradient_boosting import GradientBoostingForecast
from src.forecasting.model_selection import GRADIENT_BOOSTING, ModelChoice, ModelSelector
from src.scenarios import BASE_CASE, Scenario

logger = logging.getLogger(__name__)


@dataclass
class _FitCache:
    """The global model currently held, and the refit anchor it was fitted at."""

    model: GradientBoostingForecast | None = None
    fitted_as_of: pd.Timestamp | None = None


class ForecastService:
    """Produces SKU-level forecasts for a given decision date.

    Args:
        sales: Full tidy sales history. The service slices it per decision date; it never
            exposes rows on or after the decision date to a model.
        config: Application configuration.
        model_choices: Per-SKU model selection. When omitted, rule-based selection is
            performed on first use and reselected if an earlier decision date is asked
            for.
        weather: Optional weather provider; built from the sales data when omitted.
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        config: AppConfig,
        model_choices: dict[str, ModelChoice] | None = None,
        weather: WeatherProvider | None = None,
    ) -> None:
        if sales.empty:
            raise ValueError("ForecastService requires a non-empty sales history.")
        self.sales = sales
        self.config = config
        self.skus = sorted(str(s) for s in sales["sku"].unique())
        self.encoder = SkuEncoder.fit(self.skus)
        self.weather = weather or WeatherProvider(
            sales[["date", "temp_c"]].drop_duplicates("date"), config.weather
        )
        self.promo_calendar = sales[["date", "sku", "promo"]]
        self.selector = ModelSelector(config, strategy="rule")
        self._choices = model_choices
        # A selection supplied by the caller is authoritative. One computed lazily is made
        # at the decision date's refit anchor and remembered with that anchor (see
        # model_choices), so, like the global model, it depends on the date alone.
        self._choices_are_fixed = model_choices is not None
        self._choices_anchor: pd.Timestamp | None = None
        self._cache = _FitCache()
        # Base-case forecasts are memoised by (as_of, horizon). That is valid only because
        # the global model serving a date, and any routing chosen lazily, are fixed by the
        # date itself (see refit_anchor and model_choices), so a memoised forecast is
        # exactly what a fresh service would produce. It lets
        # repeated simulation runs - a service-level sweep, for instance - reuse one set
        # of forecasts instead of refitting for each run.
        self._forecast_cache: dict[tuple[pd.Timestamp, int], dict[str, ForecastResult]] = {}

    # -- model selection ---------------------------------------------------------------

    def model_choices(self, as_of: pd.Timestamp | None = None) -> dict[str, ModelChoice]:
        """Return the per-SKU model selection, computing it on first use.

        A selection passed to the constructor is authoritative and is never recomputed.
        A lazily computed one is made at the refit anchor of the decision date (see
        refit_anchor) and recomputed whenever the anchor changes. Routing is itself a
        decision: made at the anchor, it never uses history the date cannot see, and it
        does not depend on which dates were requested before. Without a date, the
        selection uses all history.
        """
        if self._choices_are_fixed:
            return self._choices  # type: ignore[return-value]
        anchor = self.refit_anchor(as_of) if as_of is not None else None
        if self._choices is None or anchor != self._choices_anchor:
            from src.data.loader import build_sku_master

            master = build_sku_master(self.sales, self.config)
            self._choices = self.selector.select(self.sales, master, as_of=anchor)
            self._choices_anchor = anchor
        return self._choices

    def model_for(self, sku: str, as_of: pd.Timestamp | None = None) -> ModelChoice:
        """The model chosen for one SKU, with its recorded reason."""
        choices = self.model_choices(as_of)
        if sku not in choices:
            raise KeyError(f"No model selected for SKU '{sku}'. Known SKUs: {self.skus[:5]}...")
        return choices[sku]

    # -- forecasting -------------------------------------------------------------------

    def build_future_frame(
        self,
        as_of: pd.Timestamp,
        horizon: int | None = None,
        skus: list[str] | None = None,
        scenario: Scenario = BASE_CASE,
    ) -> pd.DataFrame:
        """Build the leakage-free feature rows for the forecast window."""
        as_of = pd.Timestamp(as_of)
        horizon = horizon or self.config.forecast.horizon_days
        history = history_as_of(self.sales, as_of)
        if history.empty:
            raise ValueError(
                f"No sales history exists before {as_of.date()}; cannot forecast. "
                f"Data starts on {pd.Timestamp(self.sales['date'].min()).date()}."
            )
        future = build_future_features(
            history,
            horizon_dates(as_of, horizon),
            as_of=as_of,
            config=self.config,
            weather=self.weather,
            encoder=self.encoder,
            skus=skus or self.skus,
            promo_calendar=self.promo_calendar,
        )
        return scenario.apply_to_features(future)

    def forecast(
        self,
        as_of: pd.Timestamp,
        skus: list[str] | None = None,
        horizon: int | None = None,
        scenario: Scenario = BASE_CASE,
    ) -> dict[str, ForecastResult]:
        """Forecast demand for every requested SKU.

        Args:
            as_of: Decision date. Only demand before this date is visible to the models.
            skus: SKUs to forecast; defaults to all.
            horizon: Days to forecast; defaults to ``forecast.horizon_days``.
            scenario: Optional what-if overrides applied to features and to the result.

        Returns:
            A mapping of SKU to :class:`~src.forecasting.base.ForecastResult`.
        """
        as_of = pd.Timestamp(as_of)
        horizon = horizon or self.config.forecast.horizon_days
        targets = skus or self.skus

        cache_key = (as_of, horizon)
        if scenario.is_base_case and cache_key in self._forecast_cache:
            cached = self._forecast_cache[cache_key]
            if all(sku in cached for sku in targets):
                return {sku: cached[sku] for sku in targets}

        history = history_as_of(self.sales, as_of)
        future = self.build_future_frame(as_of, horizon, targets, scenario)
        choices = self.model_choices(as_of)

        results: dict[str, ForecastResult] = {}
        gb_skus = [s for s in targets if choices.get(s) and choices[s].model_name == GRADIENT_BOOSTING]
        if gb_skus:
            model = self._global_model(as_of)
            gb_future = future[future["sku"].isin(gb_skus)]
            for res in model.predict_panel(gb_future):
                results[res.sku] = res

        for sku in targets:
            if sku in results:
                continue
            choice = choices.get(sku)
            if choice is None:
                raise KeyError(f"SKU '{sku}' has no model selection.")
            model = self.selector.build(choice)
            if model is None:  # gradient boosting requested for an unlisted SKU
                model = self._global_model(as_of)
                results[sku] = model.predict(future[future["sku"] == sku])
                continue
            sku_history = history[history["sku"] == sku]
            try:
                results[sku] = model.fit(sku_history).predict(future[future["sku"] == sku])
            except Exception as exc:
                logger.warning(
                    "%s failed for '%s' on %s (%s); falling back to seasonal naive.",
                    choice.model_name, sku, as_of.date(), exc,
                )
                from src.forecasting.seasonal_naive import SeasonalNaiveForecast

                fallback = SeasonalNaiveForecast(
                    self.config.forecast.seasonal_period,
                    service_level=self.config.forecast.interval_service_level,
                )
                results[sku] = fallback.fit(sku_history).predict(future[future["sku"] == sku])

        results = {sku: self._apply_closure_rule(res) for sku, res in results.items()}
        if scenario.is_base_case:
            self._forecast_cache.setdefault(cache_key, {}).update(results)
        return {sku: scenario.apply_to_forecast(res) for sku, res in results.items()}

    def _apply_closure_rule(self, result: ForecastResult) -> ForecastResult:
        """Force the forecast to zero on days the site is known to be closed.

        A closed restaurant sells nothing. That is a business fact rather than a
        statistical pattern, and with only one or two closure days in two years of
        history a tree model cannot learn it dependably - it did so on some training
        cut-offs and not others, which would leave a manager ordering fresh stock for
        Christmas Day. The rule is applied after prediction, recorded in the forecast
        metadata, and can be disabled with ``features.zero_demand_on_closed_days``.
        """
        if not self.config.features.zero_demand_on_closed_days:
            return result
        closed = np.array([pd.Timestamp(d) in CLOSED_DAYS for d in result.dates])
        if not closed.any():
            return result
        predicted = np.where(closed, 0.0, result.predicted)
        lower = np.where(closed, 0.0, result.lower) if result.lower is not None else None
        upper = np.where(closed, 0.0, result.upper) if result.upper is not None else None
        return replace(
            result,
            predicted=predicted,
            lower=lower,
            upper=upper,
            metadata={
                **result.metadata,
                "closure_rule_applied": [str(pd.Timestamp(d).date()) for d, c in zip(result.dates, closed) if c],
            },
        )

    def forecast_sku(
        self,
        as_of: pd.Timestamp,
        sku: str,
        horizon: int | None = None,
        scenario: Scenario = BASE_CASE,
    ) -> ForecastResult:
        """Forecast a single SKU."""
        return self.forecast(as_of, [sku], horizon, scenario)[sku]

    def explain_forecast(
        self,
        as_of: pd.Timestamp,
        sku: str,
        horizon: int | None = None,
        scenario: Scenario = BASE_CASE,
    ):
        """Explain a SKU's forecast day by day.

        Returns:
            ``(result, explanations)`` - the forecast and one explanation per horizon
            day for feature-based models, or a single model-level explanation for the
            Croston and naive families.
        """
        from src.explainability.forecast_drivers import explain_horizon

        as_of = pd.Timestamp(as_of)
        horizon = horizon or self.config.forecast.horizon_days
        result = self.forecast_sku(as_of, sku, horizon, scenario)
        history = history_as_of(self.sales, as_of, sku=sku)
        model = (
            self._global_model(as_of)
            if result.model_name == GRADIENT_BOOSTING
            else None
        )
        future = self.build_future_frame(as_of, horizon, [sku], scenario)
        explanations = explain_horizon(result, future, history, self.config, model)
        return result, explanations

    def recent_average_demand(self, as_of: pd.Timestamp, sku: str, days: int = 28) -> float:
        """Trailing mean daily demand, used as the "normal" reference in explanations."""
        history = history_as_of(self.sales, as_of, sku=sku)
        window = history[history["date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=days)]
        if window.empty:
            return float(history["units_sold"].mean()) if not history.empty else 0.0
        return float(window["units_sold"].mean())

    # -- internals ---------------------------------------------------------------------

    def refit_anchor(self, as_of: pd.Timestamp) -> pd.Timestamp:
        """The date the global model serving ``as_of`` is fitted at.

        Refits fall every ``forecast.refit_frequency_days`` days on a grid anchored at
        ``simulation.start_date`` and extended in both directions. The model for a decision
        date is fitted at the latest grid date on or before it, so it depends on that date
        alone and never on which dates were forecast earlier, and it never sees demand the
        decision could not. If the anchor falls within ``forecast.min_history_days`` of the
        start of the data, too little history precedes it to fit on, so the model is fitted
        at ``as_of`` itself, which is equally a function of the date alone.
        """
        as_of = pd.Timestamp(as_of)
        cadence = self.config.forecast.refit_frequency_days
        origin = pd.Timestamp(self.config.simulation.start_date)
        anchor = origin + pd.Timedelta(days=cadence * ((as_of - origin).days // cadence))
        earliest = pd.Timestamp(self.sales["date"].min()) + pd.Timedelta(days=self.config.forecast.min_history_days)
        if anchor < earliest:
            return as_of
        return anchor

    def _global_model(self, as_of: pd.Timestamp) -> GradientBoostingForecast:
        """Return the global model for ``as_of``, fitting it at its refit anchor if needed."""
        anchor = self.refit_anchor(as_of)
        if self._cache.model is None or self._cache.fitted_as_of != anchor:
            logger.debug("Fitting the global model at %s for decision date %s", anchor.date(), pd.Timestamp(as_of).date())
            model = GradientBoostingForecast(self.config, self.encoder).fit(history_as_of(self.sales, anchor))
            self._cache = _FitCache(model=model, fitted_as_of=anchor)
        return self._cache.model  # type: ignore[return-value]

    @property
    def last_fit_date(self) -> pd.Timestamp | None:
        """Refit anchor of the global model currently held."""
        return self._cache.fitted_as_of
