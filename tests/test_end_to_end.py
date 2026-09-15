"""End-to-end tests over the real dataset.

One test walks the whole pipeline the dissertation describes -
data → features → segmentation → forecast → recommendation → explanation → simulation →
evaluation → storage - and asserts that each stage hands a usable result to the next.

The rest cover the reproducibility and leakage guarantees at the level of the full
system rather than an individual component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app_services import (
    build_recommendations,
    build_context,
    compare_strategies,
    default_as_of,
    overview_metrics,
    stock_positions,
)
from src.data.loader import build_sku_master
from src.data.preprocessing import history_as_of
from src.data.validation import DataValidationError, validate_sales_frame
from src.evaluation.inventory_metrics import compute_kpis
from src.explainability.recommendation_reason import explain_recommendation
from src.features.demand_features import engineer_features, feature_columns
from src.forecasting.model_selection import ModelSelector
from src.forecasting.service import ForecastService
from src.inventory.simulator import InventorySimulator
from src.inventory.strategies import ForecastDrivenStrategy, ParLevelStrategy
from src.persistence.database import Database
from src.segmentation import segment_skus, to_segments

pytestmark = pytest.mark.slow

AS_OF = pd.Timestamp("2025-09-01")


class TestFullPipeline:
    def test_data_to_simulation_result(self, sales, config, tmp_path):
        """Sample data → features → forecast → recommendation → simulation → storage."""
        # 1. Data loads and validates.
        validate_sales_frame(sales, config)
        assert len(sales) == 14_620
        master = build_sku_master(sales, config)
        assert len(master) == 20

        # 2. Features are engineered without leakage.
        history = history_as_of(sales, AS_OF)
        engineered = engineer_features(history, config)
        assert set(feature_columns(config)) <= set(engineered.columns)
        assert engineered["date"].max() < AS_OF

        # 3. SKUs are segmented and routed to a model, each with a recorded reason.
        segmentation = segment_skus(sales, master, config, as_of=AS_OF)
        segments = to_segments(segmentation)
        choices = ModelSelector(config, "rule").select(sales, master, as_of=AS_OF)
        assert len(choices) == 20
        assert {c.model_name for c in choices.values()} == {"Gradient Boosting", "Croston (SBA)"}
        assert all(c.reason for c in choices.values())

        # 4. Every SKU gets a forecast in the shared structure.
        service = ForecastService(sales, config, model_choices=choices)
        forecasts = service.forecast(AS_OF)
        assert len(forecasts) == 20
        for result in forecasts.values():
            assert result.horizon == config.forecast.horizon_days
            assert (result.predicted >= 0).all()
            assert result.to_frame().shape[0] == config.forecast.horizon_days

        # 5. Forecasts become order recommendations with the constraint trail intact.
        context = build_context()
        stock = stock_positions(context, AS_OF)
        recommendations = build_recommendations(context, AS_OF, stock)
        assert len(recommendations) == 20
        for rec in recommendations:
            assert rec.recommended_order_quantity >= 0
            assert rec.inventory_position == pytest.approx(rec.on_hand + rec.on_order)
            if rec.recommended_order_quantity > 0:
                assert rec.order_value > 0

        # 6. Each recommendation can be explained to a manager.
        explanation = explain_recommendation(
            recommendations[0],
            recent_average_daily=service.recent_average_demand(AS_OF, recommendations[0].sku),
            segment=segments.get(recommendations[0].sku),
        )
        assert explanation.headline and explanation.bullets()
        assert "no order is placed automatically" in explanation.full_text().lower()

        # 7. The simulator replays a holdout window under the same policy.
        window_start, window_end = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-31")
        result = InventorySimulator(
            sales, config, ForecastDrivenStrategy(service, config, segments=segments),
            sku_master=master,
        ).run(window_start, window_end)
        assert result.daily["date"].nunique() == 31
        assert len(result.daily) == 31 * 20

        # 8. Evaluation produces the reported KPIs.
        kpis = compute_kpis(result.daily, result.orders)
        assert 0 <= kpis["unit_service_level"] <= 100
        assert kpis["waste_units"] >= 0
        assert kpis["total_cost"] > 0

        # 9. Everything persists to SQLite and reads back.
        database = Database(tmp_path / "e2e.db")
        database.initialise()
        database.save_sku_master(master, segmentation)
        database.save_forecasts(list(forecasts.values()), run_id="e2e", as_of=AS_OF)
        database.save_recommendations(recommendations, run_id="e2e")
        database.save_simulation(
            run_id="e2e-sim", strategy=result.strategy_name, daily=result.daily,
            orders=result.orders, kpis=kpis, start=window_start, end=window_end, config=config,
        )
        database.save_model_metadata(choices)
        assert len(database.load_table("skus")) == 20
        assert len(database.load_table("forecasts")) == 20 * config.forecast.horizon_days
        assert len(database.load_table("recommendations")) == 20
        assert not database.load_simulation_comparison().empty

    def test_overview_metrics_are_coherent(self, config):
        context = build_context()
        as_of = default_as_of(context)
        stock = stock_positions(context, as_of)
        recommendations = build_recommendations(context, as_of, stock)
        metrics = overview_metrics(recommendations, as_of, config)

        assert metrics["total_skus"] == 20
        assert 0 <= metrics["skus_requiring_orders"] <= 20
        assert metrics["total_order_value"] == pytest.approx(
            sum(r.order_value for r in recommendations)
        )
        assert metrics["next_order_date"] > as_of
        assert metrics["projected_waste_cost"] >= 0
        assert metrics["projected_stockout_cost"] >= 0


class TestReproducibility:
    def test_two_identical_runs_give_identical_recommendations(self, sales, config):
        """Seed 42 throughout: the same inputs must produce the same order quantities."""
        def run() -> list[float]:
            service = ForecastService(sales, config)
            results = service.forecast(AS_OF)
            return [float(results[sku].total()) for sku in sorted(results)]

        np.testing.assert_allclose(run(), run())

    def test_simulation_is_deterministic(self, sales, config):
        master = build_sku_master(sales, config)
        start, end = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-14")

        def run() -> pd.DataFrame:
            simulator = InventorySimulator(
                sales, config, ParLevelStrategy(sales, config), sku_master=master
            )
            return simulator.run(start, end).daily

        pd.testing.assert_frame_equal(run(), run())


class TestSystemLevelLeakage:
    def test_holdout_demand_cannot_reach_an_earlier_order(self, sales, config):
        """Corrupt the last month of the holdout; the first month's orders must not move."""
        master = build_sku_master(sales, config)
        start, end = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-30")
        cutoff = pd.Timestamp("2025-11-01")

        tampered = sales.copy()
        tampered.loc[tampered["date"] >= cutoff, "units_sold"] = 0

        def orders(panel: pd.DataFrame) -> pd.DataFrame:
            service = ForecastService(panel, config)
            simulator = InventorySimulator(
                panel, config, ForecastDrivenStrategy(service, config), sku_master=master
            )
            result = simulator.run(start, end)
            return result.orders[result.orders["ordered_on"] < cutoff].reset_index(drop=True)

        pd.testing.assert_frame_equal(orders(sales), orders(tampered))

    def test_validation_rejects_a_corrupted_dataset(self, sales, config):
        duplicated = pd.concat([sales, sales.head(1)], ignore_index=True)
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_sales_frame(duplicated, config)

        negative = sales.copy()
        negative.loc[0, "units_sold"] = -5
        with pytest.raises(DataValidationError, match="Negative demand"):
            validate_sales_frame(negative, config)

        gapped = sales[sales["date"] != pd.Timestamp("2025-06-15")]
        with pytest.raises(DataValidationError, match="Missing calendar days"):
            validate_sales_frame(gapped, config)


class TestResearchEvidence:
    """The comparison the research question turns on, over a short window."""

    def test_both_policies_run_and_are_comparable(self, config):
        context = build_context()
        start, end = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-31")
        results = compare_strategies(context, start, end)

        comparison = results["comparison"].set_index("key")
        for key in ("waste_units", "stockout_units", "unit_service_level", "total_cost"):
            assert key in comparison.index
            assert np.isfinite(comparison.loc[key, "improvement_pct"])

        # Both policies faced the same demand - the comparison is like for like.
        assert results["baseline_kpis"]["total_demand"] == pytest.approx(
            results["ai_kpis"]["total_demand"]
        )
        # The forecast-driven policy should not need more stock to do its job.
        assert (
            results["ai_kpis"]["average_inventory_units"]
            < results["baseline_kpis"]["average_inventory_units"]
        )


class TestNavigationIndependence:
    def test_simulation_result_does_not_depend_on_dashboard_navigation(self):
        """Regression: opening the Overview page before the Simulation page changed its numbers.

        Overview replays recent dates and memoises their forecasts; the simulation then reused
        them although they came from models fitted on a different refit schedule.
        """
        from src.app_services import recent_forecast_accuracy

        fresh = compare_strategies(build_context())["ai_kpis"]
        context = build_context()
        as_of = pd.Timestamp("2025-12-05")
        stock = stock_positions(context, as_of)
        build_recommendations(context, as_of, stock)
        recent_forecast_accuracy(context, as_of, 35)
        after_overview = compare_strategies(context)["ai_kpis"]
        for key in ("waste_units", "stockout_units", "unit_service_level", "total_cost"):
            assert after_overview[key] == pytest.approx(fresh[key]), key
