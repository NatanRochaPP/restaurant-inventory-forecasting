"""Rolling-origin backtesting.

Random train/test splitting is invalid for time series, so evaluation uses an expanding
window: for each origin the models see only data strictly before it and are scored on
the following ``horizon`` days. The layout matches the AE1 baseline (12 folds, 7-day
horizon) so the two sets of numbers are comparable.

Every fold rebuilds its features from :func:`~src.data.preprocessing.history_as_of`,
which is what keeps the exercise honest: no fold can see its own evaluation window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.data.preprocessing import actuals_window, history_as_of, horizon_dates, rolling_origins
from src.evaluation.forecast_metrics import evaluate_forecast
from src.features.demand_features import SkuEncoder, build_future_features
from src.features.weather import WeatherProvider
from src.forecasting.base import ForecastModel, ForecastResult
from src.forecasting.croston import CrostonForecast
from src.forecasting.ets import ETSForecast
from src.forecasting.gradient_boosting import GradientBoostingForecast
from src.forecasting.naive import NaiveForecast
from src.forecasting.seasonal_naive import SeasonalNaiveForecast

logger = logging.getLogger(__name__)

#: Factories for the univariate models compared in every backtest.
UnivariateFactory = Callable[[AppConfig], ForecastModel]

DEFAULT_UNIVARIATE_MODELS: dict[str, UnivariateFactory] = {
    "Naive": lambda cfg: NaiveForecast(service_level=cfg.forecast.interval_service_level),
    "Seasonal Naive": lambda cfg: SeasonalNaiveForecast(
        seasonal_period=cfg.forecast.seasonal_period,
        service_level=cfg.forecast.interval_service_level,
    ),
    "ETS (Holt-Winters)": lambda cfg: ETSForecast(
        seasonal_period=cfg.forecast.seasonal_period,
        service_level=cfg.forecast.interval_service_level,
    ),
    "Croston": lambda cfg: CrostonForecast(
        alpha=cfg.forecast.croston.alpha, variant="classic",
        service_level=cfg.forecast.interval_service_level,
    ),
    "Croston (SBA)": lambda cfg: CrostonForecast(
        alpha=cfg.forecast.croston.alpha, variant="sba",
        service_level=cfg.forecast.interval_service_level,
    ),
}


@dataclass
class BacktestResult:
    """Per-fold metrics plus the raw predictions that produced them."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    origins: list[pd.Timestamp] = field(default_factory=list)

    def summary(self, by: str = "model") -> pd.DataFrame:
        """Mean metrics per model (or per SKU), sorted by sMAPE."""
        cols = ["sMAPE", "MASE", "WAPE", "bias", "interval_coverage"]
        cols = [c for c in cols if c in self.metrics.columns]
        return self.metrics.groupby(by, observed=True)[cols].mean().round(2).sort_values("sMAPE")


def run_backtest(
    sales: pd.DataFrame,
    config: AppConfig,
    *,
    n_folds: int | None = None,
    horizon: int | None = None,
    skus: Sequence[str] | None = None,
    include_gradient_boosting: bool = True,
    univariate_models: dict[str, UnivariateFactory] | None = None,
    estimate_sigma: bool = True,
) -> BacktestResult:
    """Run an expanding-window rolling-origin backtest.

    Args:
        sales: Full tidy sales history.
        config: Application configuration.
        n_folds: Number of folds; defaults to ``forecast.backtest.n_folds``.
        horizon: Forecast horizon in days; defaults to ``forecast.backtest.horizon_days``.
        skus: Restrict the evaluation to these SKUs (useful for quick runs).
        include_gradient_boosting: Whether to train the global ML model each fold.
        univariate_models: Override the default per-SKU model set.
        estimate_sigma: Estimate out-of-sample forecast error for the ML model so that
            interval coverage can be reported. Doubles the ML training cost.

    Returns:
        A :class:`BacktestResult` with one metrics row per (fold, SKU, model).
    """
    n_folds = n_folds or config.forecast.backtest.n_folds
    horizon = horizon or config.forecast.backtest.horizon_days
    factories = univariate_models if univariate_models is not None else DEFAULT_UNIVARIATE_MODELS

    all_skus = sorted(str(s) for s in sales["sku"].unique())
    target_skus = [s for s in (skus or all_skus)]
    unknown = set(target_skus) - set(all_skus)
    if unknown:
        raise ValueError(f"Unknown SKU(s) requested for backtest: {sorted(unknown)}")

    encoder = SkuEncoder.fit(all_skus)
    weather = WeatherProvider(sales[["date", "temp_c"]].drop_duplicates("date"), config.weather)
    promo_calendar = sales[["date", "sku", "promo"]]
    dates = pd.DatetimeIndex(sales["date"].unique()).sort_values()
    origins = rolling_origins(dates, n_folds, horizon)

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, origin in enumerate(origins):
        history = history_as_of(sales, origin)
        window = horizon_dates(origin, horizon)
        future = build_future_features(
            history, window, as_of=origin, config=config, weather=weather,
            encoder=encoder, skus=all_skus, promo_calendar=promo_calendar,
        )
        actual_panel = actuals_window(sales, origin, horizon)
        actual_by_sku = {
            str(sku): g.sort_values("date")["units_sold"].to_numpy(dtype=float)
            for sku, g in actual_panel.groupby("sku", observed=True)
        }
        history_by_sku = {
            str(sku): g.sort_values("date")["units_sold"].to_numpy(dtype=float)
            for sku, g in history.groupby("sku", observed=True)
        }

        results: list[ForecastResult] = []

        if include_gradient_boosting:
            gb = GradientBoostingForecast(
                config, encoder, sigma_validation_days=28 if estimate_sigma else 0
            ).fit(history)
            results.extend(r for r in gb.predict_panel(future) if r.sku in target_skus)

        for sku in target_skus:
            sku_history = history[history["sku"] == sku]
            sku_future = future[future["sku"] == sku]
            for label, factory in factories.items():
                model = factory(config)
                try:
                    results.append(model.fit(sku_history).predict(sku_future))
                except Exception as exc:
                    logger.warning("Fold %d: %s failed for '%s': %s", fold, label, sku, exc)

        for res in results:
            actual = actual_by_sku.get(res.sku)
            if actual is None or len(actual) != res.horizon:
                logger.debug("Fold %d: incomplete actuals for '%s'; skipped.", fold, res.sku)
                continue
            metrics = evaluate_forecast(
                actual, res.predicted, history_by_sku[res.sku], config.forecast.seasonal_period
            )
            coverage = float("nan")
            if res.lower is not None and res.upper is not None:
                coverage = float(np.mean((actual >= res.lower) & (actual <= res.upper)))
            metric_rows.append(
                {
                    "fold": fold,
                    "origin": origin,
                    "sku": res.sku,
                    "model": res.model_name,
                    **metrics,
                    "interval_coverage": coverage,
                    "actual_total": float(actual.sum()),
                    "forecast_total": float(res.total()),
                }
            )
            frame = res.to_frame()
            frame["fold"] = fold
            frame["actual"] = actual
            prediction_frames.append(frame)

        logger.info("Backtest fold %d/%d complete (origin %s)", fold + 1, len(origins), origin.date())

    return BacktestResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        origins=origins,
    )
