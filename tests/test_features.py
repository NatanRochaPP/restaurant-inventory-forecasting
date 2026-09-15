"""Tests for calendar, weather and demand feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import history_as_of, horizon_dates
from src.features.calendar import (
    BANK_HOLIDAYS,
    calendar_features,
    days_until_next_order,
    is_school_holiday,
    next_order_date,
)
from src.features.demand_features import (
    SkuEncoder,
    build_future_features,
    demand_volatility,
    engineer_features,
    feature_columns,
    prepare_feature_matrix,
)
from src.features.weather import WeatherProvider


class TestCalendarFeatures:
    def test_weekend_flag_covers_friday_to_sunday(self):
        feats = calendar_features(pd.date_range("2025-06-02", periods=7, freq="D"))  # Mon..Sun
        assert feats["weekend"].tolist() == [0, 0, 0, 0, 1, 1, 1]

    def test_bank_holiday_flag(self):
        feats = calendar_features(pd.to_datetime(["2025-05-26", "2025-05-27"]))
        assert feats["bank_holiday"].tolist() == [1, 0]
        assert pd.Timestamp("2025-05-26") in BANK_HOLIDAYS

    def test_school_holiday_windows(self):
        assert is_school_holiday(pd.Timestamp("2025-08-14")) == 1   # summer
        assert is_school_holiday(pd.Timestamp("2025-12-27")) == 1   # Christmas
        assert is_school_holiday(pd.Timestamp("2025-09-17")) == 0   # term time

    def test_next_order_date_respects_supplier_calendar(self):
        # Monday/Wednesday/Friday supplier
        assert next_order_date(pd.Timestamp("2025-06-03"), [0, 2, 4]) == pd.Timestamp("2025-06-04")
        assert next_order_date(pd.Timestamp("2025-06-04"), [0, 2, 4]) == pd.Timestamp("2025-06-04")

    def test_days_until_next_order_is_the_effective_review_period(self):
        # Friday to Monday is a three-day gap the delivery must cover.
        assert days_until_next_order(pd.Timestamp("2025-06-06"), [0, 2, 4]) == 3
        assert days_until_next_order(pd.Timestamp("2025-06-04"), [0, 2, 4]) == 2


class TestDemandFeatures:
    def test_lags_are_computed_within_each_sku(self, synthetic_panel, config):
        feats = engineer_features(synthetic_panel, config)
        for sku, grp in feats.groupby("sku"):
            grp = grp.sort_values("date")
            expected = grp["units_sold"].shift(7)
            pd.testing.assert_series_equal(
                grp["lag7"].astype(float).reset_index(drop=True),
                expected.astype(float).reset_index(drop=True),
                check_names=False,
            )

    def test_rolling_mean_does_not_cross_sku_boundaries(self, synthetic_panel, config):
        """Regression test for the AE1 bug where rolling() ran over the whole frame."""
        feats = engineer_features(synthetic_panel, config)
        for sku, grp in feats.groupby("sku"):
            grp = grp.sort_values("date").reset_index(drop=True)
            # The first 28 rows of every SKU must be undefined, not borrowed from a neighbour.
            assert grp.loc[:27, "roll28"].isna().all()
            manual = grp["units_sold"].shift(1).rolling(28).mean()
            np.testing.assert_allclose(
                grp["roll28"].astype(float).to_numpy(), manual.to_numpy(), rtol=1e-5, equal_nan=True
            )

    def test_rolling_mean_uses_only_prior_days(self, synthetic_panel, config):
        feats = engineer_features(synthetic_panel, config)
        grp = feats[feats["sku"] == "Smooth SKU"].sort_values("date").reset_index(drop=True)
        row = 40
        window = grp.loc[row - 28 : row - 1, "units_sold"]
        assert grp.loc[row, "roll28"] == pytest.approx(window.mean(), rel=1e-5)

    def test_feature_matrix_column_order_is_stable(self, synthetic_panel, config):
        feats = engineer_features(synthetic_panel, config)
        X, y = prepare_feature_matrix(feats, config)
        assert list(X.columns) == feature_columns(config)
        assert y is not None and len(X) == len(y)

    def test_prepare_feature_matrix_reports_missing_columns(self, synthetic_panel, config):
        with pytest.raises(ValueError, match="missing column"):
            prepare_feature_matrix(synthetic_panel, config)

    def test_demand_volatility_ignores_pure_weekly_seasonality(self):
        cycle = np.tile([10.0, 10, 10, 10, 30, 30, 30], 8)
        assert demand_volatility(pd.Series(cycle)) == pytest.approx(0.0, abs=1e-9)
        assert np.std(cycle, ddof=1) > 8  # naive std would massively overstate it


class TestSkuEncoder:
    def test_encoding_is_sorted_and_stable(self):
        enc = SkuEncoder.fit(["Zeta", "Alpha", "Mid"])
        assert enc.mapping == {"Alpha": 0, "Mid": 1, "Zeta": 2}
        assert enc.skus == ["Alpha", "Mid", "Zeta"]

    def test_unknown_sku_raises(self):
        enc = SkuEncoder.fit(["Alpha"])
        with pytest.raises(ValueError, match="not present when the encoder was fitted"):
            enc.transform(pd.Series(["Beta"], dtype="string"))


class TestFutureFeatures:
    @pytest.fixture
    def setup(self, synthetic_panel, config):
        as_of = pd.Timestamp("2025-04-01")
        history = history_as_of(synthetic_panel, as_of)
        weather = WeatherProvider(synthetic_panel[["date", "temp_c"]].drop_duplicates("date"), config.weather)
        encoder = SkuEncoder.fit([str(s) for s in synthetic_panel["sku"].unique()])
        dates = horizon_dates(as_of, config.forecast.horizon_days)
        future = build_future_features(
            history, dates, as_of=as_of, config=config, weather=weather,
            encoder=encoder, promo_calendar=synthetic_panel[["date", "sku", "promo"]],
        )
        return as_of, history, future

    def test_future_frame_has_no_realised_demand(self, setup):
        _, _, future = setup
        assert "units_sold" not in future.columns

    def test_one_row_per_sku_and_day(self, setup, config):
        _, history, future = setup
        assert len(future) == history["sku"].nunique() * config.forecast.horizon_days

    def test_rolling_features_are_frozen_at_the_origin(self, setup):
        """Every horizon day shares the origin's 28-day average - it cannot update."""
        as_of, history, future = setup
        for sku, grp in future.groupby("sku"):
            assert grp["roll28"].nunique() == 1
            expected = history[history["sku"] == sku].sort_values("date")["units_sold"].iloc[-28:].mean()
            assert grp["roll28"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_lag7_matches_observed_history(self, setup):
        as_of, history, future = setup
        grp = future[future["sku"] == "Smooth SKU"].sort_values("date").reset_index(drop=True)
        hist = history[history["sku"] == "Smooth SKU"].set_index("date")["units_sold"]
        for i in range(7):
            target = grp.loc[i, "date"] - pd.Timedelta(days=7)
            assert grp.loc[i, "lag7"] == pytest.approx(float(hist.loc[target]))

    def test_history_containing_the_origin_is_rejected(self, synthetic_panel, config):
        as_of = pd.Timestamp("2025-04-01")
        leaky = synthetic_panel[synthetic_panel["date"] <= as_of]  # includes as_of
        weather = WeatherProvider(synthetic_panel[["date", "temp_c"]].drop_duplicates("date"), config.weather)
        encoder = SkuEncoder.fit([str(s) for s in synthetic_panel["sku"].unique()])
        with pytest.raises(ValueError, match="not available at decision time"):
            build_future_features(
                leaky, horizon_dates(as_of, 7), as_of=as_of, config=config,
                weather=weather, encoder=encoder,
            )


class TestWeatherProvider:
    def test_forecast_degrades_with_lead_time(self, synthetic_panel, config):
        weather = WeatherProvider(synthetic_panel[["date", "temp_c"]].drop_duplicates("date"), config.weather)
        as_of = pd.Timestamp("2025-03-01")
        dates = horizon_dates(as_of, 7)
        forecast = weather.temperature_forecast(as_of, dates)
        truth = synthetic_panel.drop_duplicates("date").set_index("date")["temp_c"].reindex(dates)
        errors = np.abs(forecast.to_numpy() - truth.to_numpy())
        assert errors[0] == pytest.approx(0.0, abs=1e-9)  # today is observed
        assert errors[1:].sum() > 0                        # later days are uncertain

    def test_forecast_is_reproducible(self, synthetic_panel, config):
        weather = WeatherProvider(synthetic_panel[["date", "temp_c"]].drop_duplicates("date"), config.weather)
        dates = horizon_dates(pd.Timestamp("2025-03-01"), 7)
        a = weather.temperature_forecast(pd.Timestamp("2025-03-01"), dates)
        b = weather.temperature_forecast(pd.Timestamp("2025-03-01"), dates)
        pd.testing.assert_series_equal(a, b)
