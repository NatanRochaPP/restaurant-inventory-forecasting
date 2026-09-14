"""Tests for the inventory simulator, the ordering strategies and chronology.

The leakage tests in :class:`TestNoDataLeakage` are the most important in the suite. If
a future observation can reach a past ordering decision, every operational result the
dissertation reports is invalid, so the guarantee is tested by tampering with future
demand and asserting that nothing upstream changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import build_sku_master
from src.evaluation.inventory_metrics import compute_kpis
from src.forecasting.service import ForecastService
from src.inventory.simulator import InventorySimulator, PendingDelivery, SkuState
from src.inventory.strategies import ForecastDrivenStrategy, ParLevelStrategy

SIM_START = pd.Timestamp("2025-03-03")   # a Monday
SIM_END = pd.Timestamp("2025-04-06")


@pytest.fixture
def master(synthetic_panel, config):
    return build_sku_master(synthetic_panel, config)


@pytest.fixture
def baseline_simulator(synthetic_panel, config, master):
    return InventorySimulator(
        synthetic_panel, config, ParLevelStrategy(synthetic_panel, config), sku_master=master
    )


@pytest.fixture
def ai_simulator(synthetic_panel, config, master):
    service = ForecastService(synthetic_panel, config)
    return InventorySimulator(
        synthetic_panel, config, ForecastDrivenStrategy(service, config), sku_master=master
    )


class TestSimulationChronology:
    def test_every_day_in_the_window_is_simulated(self, baseline_simulator, synthetic_panel):
        result = baseline_simulator.run(SIM_START, SIM_END)
        expected = pd.date_range(SIM_START, SIM_END, freq="D")
        assert sorted(result.daily["date"].unique()) == list(expected)
        assert len(result.daily) == len(expected) * synthetic_panel["sku"].nunique()

    def test_orders_are_only_placed_on_supplier_order_days(self, baseline_simulator, config):
        result = baseline_simulator.run(SIM_START, SIM_END)
        placed = pd.DatetimeIndex(result.orders["ordered_on"].unique())
        assert set(placed.dayofweek) <= set(config.supplier.order_days)

    def test_deliveries_arrive_after_the_lead_time(self, baseline_simulator, config):
        result = baseline_simulator.run(SIM_START, SIM_END)
        gap = (result.orders["arrives_on"] - result.orders["ordered_on"]).dt.days
        assert (gap == config.supplier.lead_time_days).all()

    def test_stock_received_matches_orders_that_landed_in_the_window(self, baseline_simulator):
        result = baseline_simulator.run(SIM_START, SIM_END)
        landed = result.orders[result.orders["arrives_on"] <= SIM_END]["quantity"].sum()
        assert result.daily["received"].sum() == pytest.approx(landed)

    def test_stock_conservation_holds_for_every_sku(self, baseline_simulator):
        """Closing stock = opening + received - expired - served, day after day."""
        result = baseline_simulator.run(SIM_START, SIM_END)
        for sku, group in result.daily.groupby("sku"):
            group = group.sort_values("date")
            balance = (
                group["opening_stock"] + group["received"] - group["expired_units"] - group["served"]
            )
            np.testing.assert_allclose(balance.to_numpy(), group["closing_stock"].to_numpy(), atol=1e-6)

    def test_closing_stock_carries_into_the_next_day(self, baseline_simulator):
        result = baseline_simulator.run(SIM_START, SIM_END)
        for sku, group in result.daily.groupby("sku"):
            group = group.sort_values("date")
            np.testing.assert_allclose(
                group["closing_stock"].iloc[:-1].to_numpy(),
                group["opening_stock"].iloc[1:].to_numpy(),
                atol=1e-6,
            )

    def test_available_stock_is_what_the_order_decision_sees(self, baseline_simulator):
        """Deliveries land and expiries are written off before the order is sized."""
        result = baseline_simulator.run(SIM_START, SIM_END)
        expected = result.daily["opening_stock"] + result.daily["received"] - result.daily["expired_units"]
        np.testing.assert_allclose(
            expected.to_numpy(), result.daily["available_stock"].to_numpy(), atol=1e-6
        )

    def test_demand_served_never_exceeds_demand(self, baseline_simulator):
        result = baseline_simulator.run(SIM_START, SIM_END)
        assert (result.daily["served"] <= result.daily["demand"] + 1e-9).all()
        assert (result.daily["stockout_units"] >= -1e-9).all()
        np.testing.assert_allclose(
            (result.daily["served"] + result.daily["stockout_units"]).to_numpy(),
            result.daily["demand"].to_numpy(),
            atol=1e-6,
        )

    def test_window_validation(self, baseline_simulator, synthetic_panel):
        with pytest.raises(ValueError, match="is after end"):
            baseline_simulator.run(SIM_END, SIM_START)
        with pytest.raises(ValueError, match="beyond the last observation"):
            baseline_simulator.run(SIM_START, pd.Timestamp("2030-01-01"))
        with pytest.raises(ValueError, match="must start after"):
            baseline_simulator.run(pd.Timestamp(synthetic_panel["date"].min()), SIM_END)


class TestPerishability:
    def test_stock_expires_and_is_recorded_as_waste(self, ai_simulator):
        result = ai_simulator.run(SIM_START, SIM_END)
        # The intermittent SKU has a 3-day shelf life and will inevitably waste stock.
        spiky = result.daily[result.daily["sku"] == "Spiky SKU"]
        assert spiky["expired_units"].sum() > 0
        assert spiky["waste_cost"].sum() > 0

    def test_expired_stock_never_serves_demand(self, config, synthetic_panel, master):
        """Waste plus served can never exceed what was received plus opening stock."""
        simulator = InventorySimulator(
            synthetic_panel, config, ParLevelStrategy(synthetic_panel, config), sku_master=master
        )
        result = simulator.run(SIM_START, SIM_END)
        for sku, group in result.daily.groupby("sku"):
            group = group.sort_values("date")
            consumed = group["served"].sum() + group["expired_units"].sum()
            supplied = group["received"].sum() + group["opening_stock"].iloc[0] + group["expired_units"].iloc[0]
            assert consumed <= supplied + 1e-6

    def test_stock_age_is_tracked(self, baseline_simulator):
        result = baseline_simulator.run(SIM_START, SIM_END)
        assert (result.daily["mean_age_days"] >= 0).all()
        assert result.daily["mean_age_days"].max() > 0

    def test_no_waste_recorded_for_a_long_life_product(self, synthetic_panel, config):
        durable = synthetic_panel.copy()
        durable["shelf_life_days"] = 365
        master = build_sku_master(durable, config)
        result = InventorySimulator(
            durable, config, ParLevelStrategy(durable, config), sku_master=master
        ).run(SIM_START, SIM_END)
        assert result.daily["expired_units"].sum() == 0


class TestOrderingStrategies:
    def test_par_level_uses_only_trailing_average_usage(self, synthetic_panel, config):
        strategy = ParLevelStrategy(synthetic_panel, config)
        as_of = pd.Timestamp("2025-03-03")
        expected = (
            synthetic_panel[
                (synthetic_panel["sku"] == "Smooth SKU")
                & (synthetic_panel["date"] < as_of)
                & (synthetic_panel["date"] >= as_of - pd.Timedelta(days=config.baseline.lookback_days))
            ]["units_sold"].mean()
        )
        assert strategy.average_daily_usage("Smooth SKU", as_of) == pytest.approx(expected)

    def test_both_strategies_start_from_identical_stock(self, synthetic_panel, config, master):
        service = ForecastService(synthetic_panel, config)
        a = InventorySimulator(synthetic_panel, config, ParLevelStrategy(synthetic_panel, config), sku_master=master)
        b = InventorySimulator(synthetic_panel, config, ForecastDrivenStrategy(service, config), sku_master=master)
        opening_a = a._initial_state(SIM_START)
        opening_b = b._initial_state(SIM_START)
        for sku in opening_a:
            assert opening_a[sku].ledger.on_hand == pytest.approx(opening_b[sku].ledger.on_hand)

    def test_forecast_strategy_records_the_model_used(self, ai_simulator):
        result = ai_simulator.run(SIM_START, SIM_END)
        models = {r.model_name for r in result.recommendations}
        assert "Croston (SBA)" in models or "Gradient Boosting" in models
        assert all(r.policy_name for r in result.recommendations)

    def test_inventory_position_always_includes_stock_in_transit(self, ai_simulator):
        result = ai_simulator.run(SIM_START, SIM_END)
        for rec in result.recommendations:
            assert rec.inventory_position == pytest.approx(rec.on_hand + rec.on_order - rec.backorders)

    def test_stock_in_transit_suppresses_a_second_order(self, synthetic_panel, config, master):
        """With a lead time longer than the review gap, orders overlap in the pipeline.

        Under the default calendar (order Mon/Wed/Fri, two-day lead time) every delivery
        lands exactly on the next order day, so the pipeline is empty at each review. A
        four-day lead time creates genuine overlap and exercises the guard against
        ordering again for demand that is already covered.
        """
        slow = config.model_copy(
            update={"supplier": config.supplier.model_copy(update={"lead_time_days": 4})}
        )
        simulator = InventorySimulator(
            synthetic_panel, slow, ParLevelStrategy(synthetic_panel, slow), sku_master=master
        )
        result = simulator.run(SIM_START, SIM_END)
        in_transit = [r for r in result.recommendations if r.on_order > 0]
        assert in_transit, "expected overlapping deliveries with a four-day lead time"
        for rec in in_transit:
            assert rec.inventory_position > rec.on_hand
            # The order is sized against the position, so in-transit stock reduces it.
            assert rec.raw_order_quantity == pytest.approx(
                max(rec.target_stock - rec.inventory_position, 0.0)
            )


class TestSimulationCosts:
    def test_costs_are_recorded_and_sum_correctly(self, baseline_simulator):
        result = baseline_simulator.run(SIM_START, SIM_END)
        components = result.daily[["holding_cost", "waste_cost", "stockout_cost", "ordering_cost"]].sum(axis=1)
        np.testing.assert_allclose(components.to_numpy(), result.daily["total_cost"].to_numpy(), atol=1e-9)

    def test_ordering_cost_is_charged_once_per_delivery(self, baseline_simulator, config):
        result = baseline_simulator.run(SIM_START, SIM_END)
        order_days = result.orders["ordered_on"].nunique()
        expected = order_days * config.supplier.ordering_cost_per_order
        assert result.daily["ordering_cost"].sum() == pytest.approx(expected)

    def test_kpis_cover_the_reported_metrics(self, baseline_simulator):
        result = baseline_simulator.run(SIM_START, SIM_END)
        kpis = compute_kpis(result.daily, result.orders)
        for key in (
            "stockout_days", "stockout_units", "unit_service_level", "waste_units",
            "waste_cost", "holding_cost", "ordering_cost", "stockout_cost", "total_cost",
            "average_inventory_units",
        ):
            assert key in kpis and np.isfinite(kpis[key])
        assert 0 <= kpis["unit_service_level"] <= 100

    def test_kpis_reject_an_empty_run(self):
        with pytest.raises(ValueError, match="empty simulation result"):
            compute_kpis(pd.DataFrame(columns=[
                "demand", "served", "stockout_units", "stockout_day", "expired_units",
                "closing_stock", "inventory_value", "received",
                "holding_cost", "waste_cost", "stockout_cost", "ordering_cost",
            ]))


class TestNoDataLeakage:
    """The decisive tests: no future information may reach a past decision."""

    def test_recommendations_are_unchanged_when_future_demand_is_tampered_with(
        self, synthetic_panel, config, master
    ):
        """Multiply all demand after the decision date by 100; orders must not move."""
        as_of = pd.Timestamp("2025-03-05")

        def recommendations(panel: pd.DataFrame) -> list[float]:
            service = ForecastService(panel, config)
            strategy = ForecastDrivenStrategy(service, config)
            strategy.prepare(as_of, ["Smooth SKU"])
            rec = strategy.recommend(
                sku="Smooth SKU", as_of=as_of, on_hand=50.0, on_order=0.0,
                economics=config.sku_economics("Smooth SKU", 5),
                review_period_days=2, lead_time_days=2,
            )
            return [rec.recommended_order_quantity, rec.expected_demand_protection, rec.safety_stock]

        tampered = synthetic_panel.copy()
        future = tampered["date"] >= as_of
        tampered.loc[future, "units_sold"] = tampered.loc[future, "units_sold"] * 100

        np.testing.assert_allclose(recommendations(synthetic_panel), recommendations(tampered))

    def test_baseline_strategy_is_also_immune_to_future_demand(self, synthetic_panel, config):
        as_of = pd.Timestamp("2025-03-05")
        tampered = synthetic_panel.copy()
        tampered.loc[tampered["date"] >= as_of, "units_sold"] += 500
        original = ParLevelStrategy(synthetic_panel, config).average_daily_usage("Smooth SKU", as_of)
        leaked = ParLevelStrategy(tampered, config).average_daily_usage("Smooth SKU", as_of)
        assert original == pytest.approx(leaked)

    def test_simulated_orders_before_a_cutoff_are_unchanged_by_later_demand(
        self, synthetic_panel, config, master
    ):
        """Full-simulation version: tamper mid-run, orders before the cutoff must match."""
        cutoff = pd.Timestamp("2025-03-20")
        tampered = synthetic_panel.copy()
        tampered.loc[tampered["date"] >= cutoff, "units_sold"] = 9999

        def orders(panel: pd.DataFrame) -> pd.DataFrame:
            simulator = InventorySimulator(
                panel, config, ParLevelStrategy(panel, config), sku_master=master
            )
            result = simulator.run(SIM_START, SIM_END)
            return result.orders[result.orders["ordered_on"] < cutoff].reset_index(drop=True)

        pd.testing.assert_frame_equal(orders(synthetic_panel), orders(tampered))

    def test_forecast_inputs_never_include_the_forecast_window(self, synthetic_panel, config):
        as_of = pd.Timestamp("2025-03-05")
        service = ForecastService(synthetic_panel, config)
        future = service.build_future_frame(as_of, horizon=7)
        assert "units_sold" not in future.columns
        assert future["date"].min() >= as_of

    def test_a_decision_uses_no_observation_from_its_own_day(self, synthetic_panel, config):
        """Demand on the decision date itself is not yet observable."""
        as_of = pd.Timestamp("2025-03-05")
        from src.data.preprocessing import history_as_of

        history = history_as_of(synthetic_panel, as_of)
        assert history["date"].max() < as_of

    def test_the_order_decision_precedes_the_days_demand(self, ai_simulator):
        """Stock is counted for ordering before any of that day's demand is served.

        If demand were served first, the recommendation would implicitly know how much
        sold today - a one-day leak. The recorded ``available_stock`` is therefore always
        at least what is subsequently served on the same day.
        """
        result = ai_simulator.run(SIM_START, SIM_END)
        review = result.daily[result.daily["is_review_day"] == 1]
        assert (review["available_stock"] + 1e-9 >= review["served"]).all()
        # And the recommendation was computed from exactly that figure.
        by_key = {(r.sku, pd.Timestamp(r.as_of)): r for r in result.recommendations}
        for row in review.itertuples(index=False):
            rec = by_key[(row.sku, pd.Timestamp(row.date))]
            assert rec.on_hand == pytest.approx(row.available_stock)


class TestSimulatorInternals:
    def test_on_order_sums_the_pipeline(self):
        from src.inventory.shelf_life import InventoryLedger

        state = SkuState(ledger=InventoryLedger("X", 5, 1.0))
        assert state.on_order == 0.0
        state.pipeline.append(PendingDelivery("X", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), 40, 1.0))
        state.pipeline.append(PendingDelivery("X", pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04"), 10, 1.0))
        assert state.on_order == 50.0

    def test_unknown_sku_is_rejected(self, synthetic_panel, config, master):
        with pytest.raises(ValueError, match="No economics available"):
            InventorySimulator(
                synthetic_panel, config, ParLevelStrategy(synthetic_panel, config),
                skus=["Ghost SKU"], sku_master=master,
            )
