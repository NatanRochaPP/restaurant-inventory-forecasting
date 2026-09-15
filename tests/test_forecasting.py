"""Tests for the forecasting models and the shared forecast contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import history_as_of, horizon_dates
from src.evaluation.forecast_metrics import mase, smape, wape
from src.features.demand_features import SkuEncoder, build_future_features
from src.features.weather import WeatherProvider
from src.forecasting.base import ForecastResult, z_score
from src.forecasting.croston import CrostonForecast, zero_demand_share
from src.forecasting.ets import ETSForecast
from src.forecasting.gradient_boosting import GradientBoostingForecast
from src.forecasting.naive import NaiveForecast
from src.forecasting.seasonal_naive import SeasonalNaiveForecast


@pytest.fixture
def future_frame(synthetic_panel, config):
    as_of = pd.Timestamp("2025-04-01")
    history = history_as_of(synthetic_panel, as_of)
    weather = WeatherProvider(synthetic_panel[["date", "temp_c"]].drop_duplicates("date"), config.weather)
    encoder = SkuEncoder.fit([str(s) for s in synthetic_panel["sku"].unique()])
    future = build_future_features(
        history, horizon_dates(as_of, config.forecast.horizon_days), as_of=as_of,
        config=config, weather=weather, encoder=encoder,
    )
    return as_of, history, future, encoder


class TestForecastContract:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: NaiveForecast(),
            lambda: SeasonalNaiveForecast(7),
            lambda: ETSForecast(7),
            lambda: CrostonForecast(0.1, "classic"),
            lambda: CrostonForecast(0.1, "sba"),
        ],
    )
    def test_every_model_returns_the_same_structure(self, factory, future_frame):
        _, history, future, _ = future_frame
        sku = "Smooth SKU"
        result = factory().fit(history[history["sku"] == sku]).predict(future[future["sku"] == sku])
        assert isinstance(result, ForecastResult)
        assert result.sku == sku
        assert result.horizon == 7
        assert result.model_name
        frame = result.to_frame()
        assert list(frame.columns) == [
            "sku", "forecast_date", "predicted_demand", "lower_bound", "upper_bound", "model",
        ]
        assert (frame["predicted_demand"] >= 0).all()
        assert (frame["lower_bound"] <= frame["predicted_demand"] + 1e-9).all()
        assert (frame["upper_bound"] >= frame["predicted_demand"] - 1e-9).all()

    def test_predicting_before_fitting_raises(self, future_frame):
        _, _, future, _ = future_frame
        with pytest.raises(RuntimeError, match="must be fitted"):
            NaiveForecast().predict(future[future["sku"] == "Smooth SKU"])

    def test_univariate_models_reject_multi_sku_history(self, future_frame):
        _, history, _, _ = future_frame
        with pytest.raises(ValueError, match="one SKU at a time"):
            NaiveForecast().fit(history)

    def test_forecast_result_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="values for"):
            ForecastResult(
                sku="X", model_name="m",
                dates=pd.date_range("2025-01-01", periods=3),
                predicted=np.array([1.0, 2.0]),
            )

    def test_forecast_result_rejects_negative_demand(self):
        with pytest.raises(ValueError, match="negative demand"):
            ForecastResult(
                sku="X", model_name="m",
                dates=pd.date_range("2025-01-01", periods=2),
                predicted=np.array([1.0, -1.0]),
            )


class TestNaiveModels:
    def test_naive_repeats_the_last_observation(self, future_frame):
        _, history, future, _ = future_frame
        hist = history[history["sku"] == "Smooth SKU"]
        result = NaiveForecast().fit(hist).predict(future[future["sku"] == "Smooth SKU"])
        last = float(hist.sort_values("date")["units_sold"].iloc[-1])
        assert np.allclose(result.predicted, last)

    def test_seasonal_naive_aligns_weekdays(self, future_frame):
        _, history, future, _ = future_frame
        hist = history[history["sku"] == "Smooth SKU"].sort_values("date")
        result = SeasonalNaiveForecast(7).fit(hist).predict(future[future["sku"] == "Smooth SKU"])
        expected = hist["units_sold"].iloc[-7:].to_numpy(dtype=float)
        np.testing.assert_allclose(result.predicted, expected)
        # weekday alignment: forecast day i has the same weekday as its source day
        for i, d in enumerate(result.dates):
            assert d.dayofweek == hist["date"].iloc[-7 + i].dayofweek

    def test_seasonal_naive_needs_a_full_cycle(self, synthetic_panel):
        short = synthetic_panel[synthetic_panel["sku"] == "Smooth SKU"].head(3)
        with pytest.raises(ValueError, match="at least 7"):
            SeasonalNaiveForecast(7).fit(short)


class TestETS:
    def test_ets_produces_a_weekly_shaped_forecast(self, future_frame):
        _, history, future, _ = future_frame
        hist = history[history["sku"] == "Smooth SKU"]
        result = ETSForecast(7).fit(hist).predict(future[future["sku"] == "Smooth SKU"])
        assert result.horizon == 7
        # weekend days (Fri-Sun) should be forecast higher than midweek for this panel
        weekend = [p for d, p in zip(result.dates, result.predicted) if d.dayofweek in (4, 5, 6)]
        midweek = [p for d, p in zip(result.dates, result.predicted) if d.dayofweek in (0, 1, 2)]
        assert np.mean(weekend) > np.mean(midweek)

    def test_short_history_is_rejected(self, synthetic_panel):
        short = synthetic_panel[synthetic_panel["sku"] == "Smooth SKU"].head(10)
        with pytest.raises(ValueError, match="at least 14"):
            ETSForecast(7).fit(short)


class TestCroston:
    def test_croston_forecasts_a_flat_demand_rate(self, future_frame):
        _, history, future, _ = future_frame
        result = CrostonForecast(0.1, "classic").fit(
            history[history["sku"] == "Spiky SKU"]
        ).predict(future[future["sku"] == "Spiky SKU"])
        assert len(set(np.round(result.predicted, 9))) == 1
        assert result.predicted[0] > 0

    def test_sba_variant_is_lower_than_classic(self, future_frame):
        """SBA removes the known positive bias of classic Croston."""
        _, history, future, _ = future_frame
        hist = history[history["sku"] == "Spiky SKU"]
        fut = future[future["sku"] == "Spiky SKU"]
        classic = CrostonForecast(0.2, "classic").fit(hist).predict(fut)
        sba = CrostonForecast(0.2, "sba").fit(hist).predict(fut)
        assert sba.predicted[0] == pytest.approx(classic.predicted[0] * (1 - 0.2 / 2))

    def test_all_zero_history_forecasts_zero(self, synthetic_panel):
        hist = synthetic_panel[synthetic_panel["sku"] == "Spiky SKU"].copy()
        hist["units_sold"] = 0
        model = CrostonForecast(0.1, "sba").fit(hist)
        assert model.rate_ == 0.0

    def test_invalid_parameters_are_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            CrostonForecast(alpha=0.0)
        with pytest.raises(ValueError, match="variant"):
            CrostonForecast(variant="bogus")

    def test_zero_demand_share(self):
        assert zero_demand_share([0, 0, 1, 3]) == pytest.approx(0.5)


class TestGradientBoosting:
    def test_global_model_predicts_every_sku(self, future_frame, config):
        _, history, future, encoder = future_frame
        model = GradientBoostingForecast(config, encoder, sigma_validation_days=0).fit(history)
        results = model.predict_panel(future)
        assert {r.sku for r in results} == set(history["sku"].unique())
        assert all(r.horizon == 7 for r in results)

    def test_predict_rejects_multi_sku_frames(self, future_frame, config):
        _, history, future, encoder = future_frame
        model = GradientBoostingForecast(config, encoder, sigma_validation_days=0).fit(history)
        with pytest.raises(ValueError, match="one SKU at a time"):
            model.predict(future)

    def test_predict_rejects_frames_carrying_actuals(self, future_frame, config):
        _, history, future, encoder = future_frame
        model = GradientBoostingForecast(config, encoder, sigma_validation_days=0).fit(history)
        leaky = future[future["sku"] == "Smooth SKU"].assign(units_sold=1)
        with pytest.raises(ValueError, match="must not be visible"):
            model.predict(leaky)

    def test_missing_features_are_reported(self, future_frame, config):
        _, history, future, encoder = future_frame
        model = GradientBoostingForecast(config, encoder, sigma_validation_days=0).fit(history)
        with pytest.raises(ValueError, match="missing feature column"):
            model.predict(future[future["sku"] == "Smooth SKU"].drop(columns=["lag7"]))

    def test_training_is_reproducible_under_a_fixed_seed(self, future_frame, config):
        _, history, future, encoder = future_frame
        a = GradientBoostingForecast(config, encoder, sigma_validation_days=0).fit(history)
        b = GradientBoostingForecast(config, encoder, sigma_validation_days=0).fit(history)
        np.testing.assert_allclose(
            a.predict_panel(future)[0].predicted, b.predict_panel(future)[0].predicted
        )


class TestMetrics:
    def test_perfect_forecast_scores_zero(self):
        y = [10.0, 12.0, 8.0]
        assert smape(y, y) == pytest.approx(0.0)
        assert wape(y, y) == pytest.approx(0.0)

    def test_smape_handles_zero_demand_days(self):
        assert smape([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)
        assert smape([0.0], [5.0]) == pytest.approx(200.0)

    def test_wape_is_demand_weighted(self):
        assert wape([100.0, 1.0], [90.0, 1.0]) == pytest.approx(10 / 101 * 100)

    def test_mase_scales_by_the_seasonal_naive_benchmark(self):
        rng = np.random.default_rng(0)
        insample = np.tile([10.0, 10, 10, 10, 30, 30, 30], 10) + rng.normal(0, 2, 70)
        actual = [10.0, 10, 10, 10, 30, 30, 30]
        # A perfect forecast scales to zero; a forecast as bad as the benchmark scales to ~1.
        assert mase(actual, actual, insample, 7) == pytest.approx(0.0)
        denom = np.mean(np.abs(insample[7:] - insample[:-7]))
        offset = [a + denom for a in actual]
        assert mase(actual, offset, insample, 7) == pytest.approx(1.0)

    def test_mase_is_undefined_without_a_usable_benchmark(self):
        actual = [10.0, 10, 10, 10, 30, 30, 30]
        assert np.isnan(mase(actual, actual, [1.0, 2.0], 7))          # too little history
        flat = np.tile([10.0, 10, 10, 10, 30, 30, 30], 10)            # zero denominator
        assert np.isnan(mase(actual, actual, flat, 7))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            smape([1.0, 2.0], [1.0])

    def test_z_score_bounds(self):
        assert z_score(0.95) == pytest.approx(1.6449, abs=1e-3)
        with pytest.raises(ValueError):
            z_score(1.5)
