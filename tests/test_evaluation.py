"""Tests for evaluation: KPIs, the fair-comparison frontier, backtesting and storage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app_services import (
    DashboardContext,
    forecast_accuracy_summary,
    recent_forecast_accuracy,
)
from src.data.loader import build_sku_master
from src.data.preprocessing import rolling_origins
from src.evaluation.backtest import run_backtest
from src.evaluation.frontier import (
    AI_LABEL,
    BASELINE_LABEL,
    MATCHED_COLUMNS,
    MATCHED_WASTE_COLUMNS,
    _interpolate,
    matched_comparison,
    run_frontier,
    service_level_at_matched_waste,
)
from src.evaluation.inventory_metrics import (
    METRIC_SPECS,
    compare_simulations,
    compute_kpis,
    daily_totals,
    kpis_by_sku,
    metric_label,
    percentage_improvement,
    summarise_headline,
)
from src.forecasting.model_selection import ModelSelector
from src.forecasting.service import ForecastService
from src.inventory.simulator import InventorySimulator
from src.inventory.strategies import ParLevelStrategy
from src.persistence.database import Database
from tests.conftest import make_synthetic_panel

SIM_START = pd.Timestamp("2025-03-03")
SIM_END = pd.Timestamp("2025-04-06")


@pytest.fixture
def master(synthetic_panel, config):
    return build_sku_master(synthetic_panel, config)


@pytest.fixture
def simulation(synthetic_panel, config, master):
    simulator = InventorySimulator(
        synthetic_panel, config, ParLevelStrategy(synthetic_panel, config), sku_master=master
    )
    return simulator.run(SIM_START, SIM_END)


class TestInventoryKpis:
    def test_service_level_is_the_unit_fill_rate(self, simulation):
        kpis = compute_kpis(simulation.daily, simulation.orders)
        served = simulation.daily["served"].sum()
        demand = simulation.daily["demand"].sum()
        assert kpis["unit_service_level"] == pytest.approx(served / demand * 100)

    def test_stockout_days_count_sku_days_not_units(self, simulation):
        kpis = compute_kpis(simulation.daily, simulation.orders)
        assert kpis["stockout_days"] == (simulation.daily["stockout_units"] > 0).sum()

    def test_waste_share_is_relative_to_stock_received(self, simulation):
        kpis = compute_kpis(simulation.daily, simulation.orders)
        expected = simulation.daily["expired_units"].sum() / simulation.daily["received"].sum() * 100
        assert kpis["waste_share_of_supply"] == pytest.approx(expected)

    def test_total_cost_is_the_sum_of_its_components(self, simulation):
        kpis = compute_kpis(simulation.daily, simulation.orders)
        assert kpis["total_cost"] == pytest.approx(
            kpis["holding_cost"] + kpis["waste_cost"] + kpis["stockout_cost"] + kpis["ordering_cost"]
        )

    def test_missing_columns_are_reported(self):
        with pytest.raises(ValueError, match="missing column"):
            compute_kpis(pd.DataFrame({"demand": [1.0]}))

    def test_per_sku_breakdown_sums_to_the_total(self, simulation):
        overall = compute_kpis(simulation.daily, simulation.orders)
        per_sku = kpis_by_sku(simulation.daily, simulation.orders)
        assert per_sku["waste_units"].sum() == pytest.approx(overall["waste_units"])
        assert per_sku["total_cost"].sum() == pytest.approx(overall["total_cost"])

    def test_daily_totals_collapse_to_one_row_per_day(self, simulation):
        totals = daily_totals(simulation.daily)
        assert len(totals) == simulation.daily["date"].nunique()
        assert totals["demand"].sum() == pytest.approx(simulation.daily["demand"].sum())

    def test_metric_labels_exist_for_every_spec(self):
        for spec in METRIC_SPECS:
            assert metric_label(spec.key) == spec.label
        assert metric_label("unknown_key") == "Unknown key"


class TestImprovementDirection:
    def test_lower_is_better_metrics(self):
        assert percentage_improvement(100, 80, "lower_is_better") == pytest.approx(20.0)
        assert percentage_improvement(100, 120, "lower_is_better") == pytest.approx(-20.0)

    def test_higher_is_better_metrics(self):
        assert percentage_improvement(90, 95, "higher_is_better") == pytest.approx(5 / 90 * 100)
        assert percentage_improvement(90, 85, "higher_is_better") < 0

    def test_zero_baseline_is_handled(self):
        assert percentage_improvement(0, 0, "lower_is_better") == 0.0
        assert np.isnan(percentage_improvement(0, 5, "lower_is_better"))

    def test_comparison_table_covers_every_metric(self, simulation):
        kpis = compute_kpis(simulation.daily, simulation.orders)
        better = {**kpis, "waste_units": kpis["waste_units"] * 0.5}
        comparison = compare_simulations(kpis, better)
        assert len(comparison) == len(METRIC_SPECS)
        waste = comparison.set_index("key").loc["waste_units"]
        assert waste["improvement_pct"] == pytest.approx(50.0, abs=0.1)

    def test_headline_summary_extracts_the_quoted_figures(self, simulation):
        kpis = compute_kpis(simulation.daily, simulation.orders)
        comparison = compare_simulations(kpis, kpis)
        headline = summarise_headline(comparison)
        assert set(headline) <= {
            "stockout_units", "waste_units", "unit_service_level",
            "total_cost", "average_inventory_units",
        }
        assert all(v == 0 for v in headline.values())


@pytest.fixture(scope="module")
def frontier(config):
    """A small two-point sweep per policy - enough to exercise the interpolation.

    Module-scoped because four simulations are slow; it builds its own panel so it does
    not depend on a function-scoped fixture.
    """
    panel = make_synthetic_panel()
    sku_master = build_sku_master(panel, config)
    service = ForecastService(panel, config)
    return run_frontier(
        panel, config, service_levels=[0.60, 0.90], buffer_pcts=[0.0, 0.30],
        forecast_service=service, sku_master=sku_master, start=SIM_START, end=SIM_END,
    )


class TestFrontier:

    def test_sweep_covers_both_policies_and_settings(self, frontier):
        assert set(frontier["strategy"]) == {BASELINE_LABEL, AI_LABEL}
        assert len(frontier) == 4
        assert {"unit_service_level", "waste_units", "total_cost"} <= set(frontier.columns)

    def test_raising_the_service_level_raises_stock_held(self, frontier):
        ai = frontier[frontier["strategy"] == AI_LABEL].sort_values("setting_value")
        assert ai["average_inventory_units"].is_monotonic_increasing

    def test_raising_the_manager_buffer_raises_stock_held(self, frontier):
        base = frontier[frontier["strategy"] == BASELINE_LABEL].sort_values("setting_value")
        assert base["average_inventory_units"].is_monotonic_increasing

    def test_interpolation_refuses_to_extrapolate(self):
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
        assert _interpolate(frame, "x", "y", 1.5) == pytest.approx(15.0)
        assert np.isnan(_interpolate(frame, "x", "y", 5.0))
        assert np.isnan(_interpolate(frame, "x", "y", 0.5))

    def test_matched_comparison_only_reports_overlapping_points(self, frontier):
        """Points outside the comparison policy's swept range are dropped, not extrapolated."""
        matched = matched_comparison(frontier)
        assert list(matched.columns) == list(MATCHED_COLUMNS)
        ai = frontier[frontier["strategy"] == AI_LABEL]
        low, high = ai["unit_service_level"].min(), ai["unit_service_level"].max()
        for value in matched["matched_service_level_pct"]:
            assert low <= value <= high

    def test_empty_match_still_has_a_usable_shape(self, frontier):
        """A non-overlapping sweep returns no rows but keeps its columns."""
        single_point = frontier[
            (frontier["strategy"] == BASELINE_LABEL) | (frontier["setting_value"] == 0.60)
        ]
        matched = matched_comparison(single_point)
        assert list(matched.columns) == list(MATCHED_COLUMNS)

    def test_matched_waste_comparison_runs(self, frontier):
        result = service_level_at_matched_waste(frontier)
        assert list(result.columns) == list(MATCHED_WASTE_COLUMNS)

    def test_missing_strategy_is_reported(self, frontier):
        with pytest.raises(ValueError, match="must contain both"):
            matched_comparison(frontier[frontier["strategy"] == AI_LABEL])


class TestBacktest:
    def test_rolling_origins_are_ordered_and_spaced_by_the_horizon(self):
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        origins = rolling_origins(dates, n_folds=4, horizon=7)
        assert len(origins) == 4
        assert origins == sorted(origins)
        gaps = {(b - a).days for a, b in zip(origins, origins[1:])}
        assert gaps == {7}
        assert origins[-1] + pd.Timedelta(days=6) == dates[-1]

    def test_too_little_data_is_rejected(self):
        with pytest.raises(ValueError, match="Need at least"):
            rolling_origins(pd.date_range("2025-01-01", periods=10, freq="D"), n_folds=4, horizon=7)

    def test_backtest_scores_every_model_on_every_fold(self, synthetic_panel, config):
        result = run_backtest(synthetic_panel, config, n_folds=2, horizon=7, estimate_sigma=False)
        assert set(result.metrics["fold"]) == {0, 1}
        assert result.metrics["sku"].nunique() == synthetic_panel["sku"].nunique()
        assert "Gradient Boosting" in set(result.metrics["model"])
        assert not result.metrics["sMAPE"].isna().all()

    def test_backtest_predictions_align_with_actuals(self, synthetic_panel, config):
        result = run_backtest(synthetic_panel, config, n_folds=2, horizon=7, estimate_sigma=False)
        assert {"predicted_demand", "actual", "fold"} <= set(result.predictions.columns)
        assert (result.predictions["predicted_demand"] >= 0).all()

    def test_backtest_never_trains_on_its_evaluation_window(self, synthetic_panel, config):
        """Tampering with demand after the last origin must not change earlier folds."""
        result = run_backtest(synthetic_panel, config, n_folds=2, horizon=7, estimate_sigma=False)
        cutoff = max(result.origins)
        tampered = synthetic_panel.copy()
        tampered.loc[tampered["date"] >= cutoff, "units_sold"] = 5000
        after = run_backtest(tampered, config, n_folds=2, horizon=7, estimate_sigma=False)

        first_fold_before = result.predictions[result.predictions["fold"] == 0]
        first_fold_after = after.predictions[after.predictions["fold"] == 0]
        np.testing.assert_allclose(
            first_fold_before["predicted_demand"].to_numpy(),
            first_fold_after["predicted_demand"].to_numpy(),
        )

    def test_unknown_sku_is_rejected(self, synthetic_panel, config):
        with pytest.raises(ValueError, match="Unknown SKU"):
            run_backtest(synthetic_panel, config, n_folds=2, skus=["Nonexistent"])


def _context_for(panel: pd.DataFrame, config) -> DashboardContext:
    """A minimal context: the accuracy replay needs only the sales, config and service."""
    return DashboardContext(
        config=config,
        sales=panel,
        sku_master=build_sku_master(panel, config),
        segmentation=pd.DataFrame(),
        segments={},
        model_choices={},
        service=ForecastService(panel, config),
    )


class TestRecentForecastAccuracy:
    """The Overview page's accuracy replay: it must be a record, not a fit."""

    AS_OF = pd.Timestamp("2025-04-30")

    def test_the_window_covers_the_requested_period_and_stops_before_the_decision_date(
        self, synthetic_panel, config
    ):
        daily = recent_forecast_accuracy(_context_for(synthetic_panel, config), self.AS_OF, 28)

        assert len(daily) == 28
        assert daily["date"].max() < self.AS_OF  # today's demand is not yet observed
        assert daily["date"].min() == self.AS_OF - pd.Timedelta(days=28)
        assert daily["date"].is_monotonic_increasing
        assert not daily["date"].duplicated().any()  # the windows tile, never overlap

    def test_a_days_prediction_cannot_see_its_own_demand(self, synthetic_panel, config):
        """The decisive test: corrupt the last window; no prediction may move.

        Tampering starts at the final origin, so no origin's history contains a tampered
        day - the earlier origins are fitted well before it, and the final origin sees
        only data strictly before itself. Every prediction in the replay must therefore
        be unchanged even though the demand it is scored against is a hundred times
        larger. Note that tampering from an *earlier* origin would legitimately move the
        later predictions, because by then those days are genuinely past.
        """
        horizon = config.forecast.horizon_days
        last_origin = self.AS_OF - pd.Timedelta(days=horizon)

        tampered = synthetic_panel.copy()
        future = tampered["date"] >= last_origin
        tampered.loc[future, "units_sold"] = tampered.loc[future, "units_sold"] * 100

        clean = recent_forecast_accuracy(_context_for(synthetic_panel, config), self.AS_OF, 28)
        leaked = recent_forecast_accuracy(_context_for(tampered, config), self.AS_OF, 28)

        np.testing.assert_allclose(leaked["predicted_units"], clean["predicted_units"])
        # The actuals must genuinely have moved, or the assertion above proves nothing.
        assert leaked["actual_units"].sum() > clean["actual_units"].sum()

    def test_too_little_history_returns_an_empty_frame_rather_than_failing(
        self, synthetic_panel, config
    ):
        daily = recent_forecast_accuracy(
            _context_for(synthetic_panel, config), pd.Timestamp("2025-01-05"), 28
        )
        assert daily.empty
        assert forecast_accuracy_summary(daily) == {}

    def test_the_two_error_scales_are_consistent_with_the_replay(self, synthetic_panel, config):
        daily = recent_forecast_accuracy(_context_for(synthetic_panel, config), self.AS_OF, 28)
        summary = forecast_accuracy_summary(daily)

        assert summary["days"] == len(daily)
        assert summary["total_actual"] == pytest.approx(daily["actual_units"].sum())
        # Per-SKU errors cannot cancel, so they can never be smaller than the net error.
        assert (daily["sku_abs_error_units"] >= daily["abs_error_units"] - 1e-9).all()
        assert summary["sku_wape"] >= summary["net_wape"]
        assert summary["net_mae_units"] == pytest.approx(daily["abs_error_units"].mean())


class TestDatabase:
    @pytest.fixture
    def database(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.initialise()
        return db

    def test_schema_creates_every_table(self, database):
        expected = {
            "skus", "sales", "inventory_snapshots", "forecasts", "recommendations",
            "recommendation_overrides", "simulated_orders", "simulation_runs",
            "simulation_results", "model_metadata",
        }
        assert expected <= set(database.tables())

    def test_initialise_is_idempotent(self, database):
        database.initialise()
        assert "skus" in database.tables()

    def test_sku_master_round_trip(self, database, synthetic_panel, config):
        master = build_sku_master(synthetic_panel, config)
        assert database.save_sku_master(master) == len(master)
        loaded = database.load_table("skus")
        assert set(loaded["sku"]) == set(master["sku"])

    def test_forecasts_round_trip(self, database, synthetic_panel, config):
        service = ForecastService(synthetic_panel, config)
        results = list(service.forecast(pd.Timestamp("2025-04-01")).values())
        as_of = pd.Timestamp("2025-04-01")
        rows = database.save_forecasts(results, run_id="run-1", as_of=as_of)
        assert rows == sum(r.horizon for r in results)
        loaded = database.load_table("forecasts")
        assert set(loaded["model"]) <= {"Gradient Boosting", "Croston (SBA)"}
        assert (loaded["predicted_demand"] >= 0).all()

    def test_simulation_round_trip(self, database, simulation, config):
        kpis = compute_kpis(simulation.daily, simulation.orders)
        database.save_simulation(
            run_id="sim-1", strategy=simulation.strategy_name, daily=simulation.daily,
            orders=simulation.orders, kpis=kpis, start=SIM_START, end=SIM_END, config=config,
        )
        runs = database.load_table("simulation_runs")
        assert len(runs) == 1 and runs["strategy"].iloc[0] == simulation.strategy_name
        results = database.load_table("simulation_results")
        assert set(results["metric"]) >= {"waste_units", "total_cost", "unit_service_level"}
        comparison = database.load_simulation_comparison()
        assert not comparison.empty

    def test_override_is_stored_without_altering_the_recommendation(self, database):
        database.record_override("run-1", pd.Timestamp("2025-04-01"), "Smooth SKU", 120.0, 200.0, "event")
        overrides = database.load_overrides()
        assert len(overrides) == 1
        row = overrides.iloc[0]
        assert row["recommended_order"] == 120.0   # the model's number is preserved
        assert row["override_order"] == 200.0
        assert row["reason"] == "event"

    def test_negative_override_is_rejected(self, database):
        with pytest.raises(ValueError, match="cannot be negative"):
            database.record_override("run-1", pd.Timestamp("2025-04-01"), "X", 10.0, -5.0)

    def test_model_metadata_records_the_reason(self, database, synthetic_panel, config):
        master = build_sku_master(synthetic_panel, config)
        choices = ModelSelector(config, "rule").select(synthetic_panel, master)
        database.save_model_metadata(choices)
        loaded = database.load_table("model_metadata")
        assert len(loaded) == len(choices)
        assert loaded["reason"].str.len().min() > 40

    def test_unknown_table_is_reported(self, database):
        with pytest.raises(ValueError, match="does not exist"):
            database.load_table("not_a_table")
