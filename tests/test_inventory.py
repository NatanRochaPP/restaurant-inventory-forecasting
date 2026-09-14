"""Tests for the inventory policy engine.

These calculations decide how much food a restaurant buys, so they are covered
exhaustively: position arithmetic, safety stock, target levels, every constraint, and
the interaction between constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SkuEconomics
from src.inventory.ordering import (
    apply_order_constraints,
    recommend_order,
    recommendations_to_frame,
    round_to_pack,
)
from src.inventory.policies import (
    BaseStockPolicy,
    PolicyInputs,
    SsPolicy,
    build_policy,
    inventory_position,
)
from src.inventory.safety_stock import (
    expected_demand_over,
    safety_stock,
    safety_stock_with_lead_time_variability,
    stockout_probability,
)
from src.inventory.shelf_life import (
    InventoryLedger,
    shelf_life_order_cap,
    usable_shelf_life_days,
)

AS_OF = pd.Timestamp("2025-07-01")


@pytest.fixture
def economics() -> SkuEconomics:
    return SkuEconomics(
        sku="Test SKU",
        unit_cost=2.0,
        pack_size=10,
        min_order_quantity=10,
        shelf_life_days=5,
        holding_cost_per_unit_per_day=0.01,
        waste_cost_per_unit=2.05,
        stockout_cost_per_unit=4.0,
    )


@pytest.fixture
def flat_inputs() -> PolicyInputs:
    """20 units/day forecast, no uncertainty - makes hand-checking arithmetic easy."""
    return PolicyInputs(
        sku="Test SKU",
        daily_forecast=np.full(7, 20.0),
        sigma_daily=0.0,
        on_hand=30.0,
        on_order=10.0,
        lead_time_days=2,
        review_period_days=1,
        service_level=0.95,
    )


class TestInventoryPosition:
    def test_position_is_on_hand_plus_on_order_minus_backorders(self):
        assert inventory_position(30, 10, 5) == 35.0

    def test_position_defaults_to_no_backorders(self):
        assert inventory_position(30, 10) == 40.0

    def test_stock_in_transit_is_not_ignored(self):
        """The classic double-ordering bug: on-order stock must count."""
        assert inventory_position(0, 100) == 100.0

    @pytest.mark.parametrize("args", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
    def test_negative_components_are_rejected(self, args):
        with pytest.raises(ValueError, match="non-negative"):
            inventory_position(*args)


class TestSafetyStock:
    def test_formula_matches_the_normal_approximation(self):
        # z(0.95) = 1.6449, sigma = 10, protection = 4 days -> 1.6449 * 10 * 2
        assert safety_stock(10.0, 4, 0.95) == pytest.approx(32.897, abs=1e-2)

    def test_no_uncertainty_needs_no_buffer(self):
        assert safety_stock(0.0, 4, 0.95) == 0.0

    def test_higher_service_level_requires_more_stock(self):
        assert safety_stock(10, 4, 0.99) > safety_stock(10, 4, 0.95) > safety_stock(10, 4, 0.80)

    def test_longer_protection_period_requires_more_stock(self):
        assert safety_stock(10, 9, 0.95) == pytest.approx(safety_stock(10, 1, 0.95) * 3)

    def test_invalid_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="Protection period"):
            safety_stock(10, -1, 0.95)
        with pytest.raises(ValueError, match="uncertainty"):
            safety_stock(-1, 4, 0.95)
        with pytest.raises(ValueError, match="Service level"):
            safety_stock(10, 4, 1.0)

    def test_lead_time_variability_adds_buffer(self):
        deterministic = safety_stock_with_lead_time_variability(10, 20, 4, 0.95, 0.0)
        variable = safety_stock_with_lead_time_variability(10, 20, 4, 0.95, 1.0)
        assert deterministic == pytest.approx(safety_stock(10, 4, 0.95))
        assert variable > deterministic

    def test_stockout_probability_responds_to_cover(self):
        assert stockout_probability(100, 80, 10, 4) < 0.2
        assert stockout_probability(60, 80, 10, 4) > 0.8
        # With no uncertainty the answer is deterministic.
        assert stockout_probability(100, 80, 0, 4) == 0.0
        assert stockout_probability(60, 80, 0, 4) == 1.0


class TestExpectedDemand:
    def test_sums_whole_days(self):
        assert expected_demand_over([10, 20, 30], 2) == 30.0

    def test_interpolates_fractional_days(self):
        assert expected_demand_over([10, 20, 30], 2.5) == pytest.approx(45.0)

    def test_extrapolates_beyond_the_horizon_at_the_mean(self):
        # 7 days of 20 units, asked for 10 days -> 140 + 3 * 20
        assert expected_demand_over(np.full(7, 20.0), 10) == pytest.approx(200.0)

    def test_zero_window_is_zero(self):
        assert expected_demand_over([10, 20], 0) == 0.0

    def test_negative_window_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            expected_demand_over([10], -1)


class TestPolicyInputsValidation:
    def test_empty_forecast_is_rejected(self):
        with pytest.raises(ValueError, match="No demand forecast"):
            PolicyInputs(sku="X", daily_forecast=np.array([]), sigma_daily=1.0, on_hand=0)

    def test_negative_forecast_is_rejected(self):
        with pytest.raises(ValueError, match="negative values"):
            PolicyInputs(sku="X", daily_forecast=np.array([1.0, -2.0]), sigma_daily=1.0, on_hand=0)

    def test_invalid_timing_is_rejected(self):
        with pytest.raises(ValueError, match="Lead time"):
            PolicyInputs(sku="X", daily_forecast=np.ones(3), sigma_daily=1, on_hand=0, lead_time_days=-1)
        with pytest.raises(ValueError, match="Review period"):
            PolicyInputs(sku="X", daily_forecast=np.ones(3), sigma_daily=1, on_hand=0, review_period_days=0)

    def test_protection_period_is_lead_time_plus_review(self, flat_inputs):
        assert flat_inputs.protection_period_days == 3


class TestBaseStockPolicy:
    def test_target_is_protection_demand_plus_safety_stock(self, flat_inputs):
        targets = BaseStockPolicy().targets(flat_inputs)
        assert targets.expected_demand_protection == pytest.approx(60.0)  # 3 days x 20
        assert targets.safety_stock == 0.0
        assert targets.target_stock == pytest.approx(60.0)

    def test_order_is_target_minus_position(self, flat_inputs):
        quantity, targets = BaseStockPolicy().raw_order_quantity(flat_inputs)
        assert flat_inputs.position == 40.0
        assert quantity == pytest.approx(20.0)

    def test_no_order_when_position_exceeds_target(self, flat_inputs):
        from dataclasses import replace

        overstocked = replace(flat_inputs, on_hand=200.0)
        quantity, _ = BaseStockPolicy().raw_order_quantity(overstocked)
        assert quantity == 0.0

    def test_uncertainty_increases_the_target(self, flat_inputs):
        from dataclasses import replace

        uncertain = replace(flat_inputs, sigma_daily=8.0)
        assert BaseStockPolicy().targets(uncertain).target_stock > BaseStockPolicy().targets(flat_inputs).target_stock


class TestSsPolicy:
    def test_no_order_above_the_reorder_point(self, flat_inputs):
        from dataclasses import replace

        # Reorder point covers the 2-day lead time = 40 units; position of 60 is above it.
        inputs = replace(flat_inputs, on_hand=50.0, on_order=10.0)
        quantity, targets = SsPolicy().raw_order_quantity(inputs)
        assert targets.reorder_point == pytest.approx(40.0)
        assert targets.should_order is False
        assert quantity == 0.0

    def test_orders_up_to_target_below_the_reorder_point(self, flat_inputs):
        from dataclasses import replace

        inputs = replace(flat_inputs, on_hand=10.0, on_order=0.0)
        quantity, targets = SsPolicy().raw_order_quantity(inputs)
        assert targets.should_order is True
        assert quantity == pytest.approx(targets.target_stock - 10.0)

    def test_reorder_multiplier_shifts_the_trigger(self, flat_inputs):
        base = SsPolicy(1.0).targets(flat_inputs).reorder_point
        cautious = SsPolicy(1.5).targets(flat_inputs).reorder_point
        assert cautious == pytest.approx(base * 1.5)

    def test_invalid_multiplier_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            SsPolicy(0.0)


class TestPolicyFactory:
    def test_builds_configured_policies(self):
        assert isinstance(build_policy("base_stock"), BaseStockPolicy)
        assert isinstance(build_policy("s_S"), SsPolicy)

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown inventory policy"):
            build_policy("magic")


class TestPackRounding:
    def test_nearest_rounds_both_ways(self):
        assert round_to_pack(46, 10, "nearest") == 50.0
        assert round_to_pack(44, 10, "nearest") == 40.0

    def test_up_always_rounds_up(self):
        assert round_to_pack(41, 10, "up") == 50.0
        assert round_to_pack(40, 10, "up") == 40.0

    def test_none_leaves_quantity_untouched(self):
        assert round_to_pack(41.5, 10, "none") == 41.5

    def test_zero_and_negative_quantities_produce_no_order(self):
        assert round_to_pack(0, 10, "up") == 0.0
        assert round_to_pack(-5, 10, "up") == 0.0

    def test_invalid_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="Pack size"):
            round_to_pack(10, 0)
        with pytest.raises(ValueError, match="Unknown pack rounding"):
            round_to_pack(10, 5, "sideways")


class TestShelfLife:
    def test_usable_life_applies_the_configured_fraction(self):
        assert usable_shelf_life_days(10, 0.8) == 8.0
        assert usable_shelf_life_days(1, 0.5) == 1.0  # never below a single day

    def test_invalid_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="Usable fraction"):
            usable_shelf_life_days(10, 0.0)

    def test_cap_reflects_demand_before_expiry(self):
        # 20 units/day, 3-day shelf life, delivery in 2 days, 10 units already held.
        cap = shelf_life_order_cap(np.full(7, 20.0), 3, 1.0, inventory_position=10.0, lead_time_days=2)
        assert cap == pytest.approx(20 * 5 - 10)

    def test_cap_is_never_negative(self):
        cap = shelf_life_order_cap(np.full(7, 1.0), 2, 1.0, inventory_position=500.0, lead_time_days=1)
        assert cap == 0.0


class TestInventoryLedger:
    def test_receiving_and_valuing_stock(self):
        ledger = InventoryLedger("Lettuce", shelf_life_days=4, unit_cost=0.6)
        ledger.receive(50, pd.Timestamp("2025-07-01"))
        assert ledger.on_hand == 50.0
        assert ledger.value == pytest.approx(30.0)

    def test_stock_expires_exactly_after_its_shelf_life(self):
        ledger = InventoryLedger("Lettuce", shelf_life_days=3, unit_cost=1.0)
        ledger.receive(10, pd.Timestamp("2025-07-01"))
        assert ledger.remove_expired(pd.Timestamp("2025-07-03")) == (0.0, 0.0)
        units, cost = ledger.remove_expired(pd.Timestamp("2025-07-04"))
        assert units == 10.0 and cost == pytest.approx(10.0)
        assert ledger.on_hand == 0.0

    def test_expired_stock_cannot_serve_demand(self):
        ledger = InventoryLedger("Lettuce", shelf_life_days=2, unit_cost=1.0)
        ledger.receive(10, pd.Timestamp("2025-07-01"))
        ledger.remove_expired(pd.Timestamp("2025-07-03"))
        served, short = ledger.issue(10)
        assert served == 0.0 and short == 10.0

    def test_issue_is_first_expired_first_out(self):
        ledger = InventoryLedger("Milk", shelf_life_days=6, unit_cost=1.0)
        ledger.receive(10, pd.Timestamp("2025-07-01"))   # expires 07-07
        ledger.receive(10, pd.Timestamp("2025-07-03"))   # expires 07-09
        served, short = ledger.issue(12)
        assert served == 12.0 and short == 0.0
        remaining = ledger.batches
        assert len(remaining) == 1
        assert remaining[0].received_on == pd.Timestamp("2025-07-03")
        assert remaining[0].quantity == 8.0

    def test_shortfall_is_reported_when_stock_runs_out(self):
        ledger = InventoryLedger("Milk", shelf_life_days=6, unit_cost=1.0)
        ledger.receive(5, pd.Timestamp("2025-07-01"))
        served, short = ledger.issue(8)
        assert served == 5.0 and short == 3.0
        assert ledger.on_hand == 0.0

    def test_age_and_expiry_reporting(self):
        ledger = InventoryLedger("Milk", shelf_life_days=6, unit_cost=1.0)
        ledger.receive(10, pd.Timestamp("2025-07-01"))
        ledger.receive(30, pd.Timestamp("2025-07-04"))
        today = pd.Timestamp("2025-07-05")
        assert ledger.mean_age_days(today) == pytest.approx((10 * 4 + 30 * 1) / 40)
        assert ledger.units_expiring_within(today, 2) == 10.0

    def test_invalid_operations_are_rejected(self):
        with pytest.raises(ValueError, match="Shelf life"):
            InventoryLedger("X", 0, 1.0)
        ledger = InventoryLedger("X", 3, 1.0)
        with pytest.raises(ValueError, match="must be positive"):
            ledger.receive(0, pd.Timestamp("2025-07-01"))
        with pytest.raises(ValueError, match="cannot be negative"):
            ledger.issue(-1)


class TestOrderConstraints:
    def test_shelf_life_cap_limits_a_large_order(self, flat_inputs, economics, config):
        from dataclasses import replace

        starving = replace(flat_inputs, on_hand=0.0, on_order=0.0)
        # Raw demand for 14 days would be 280 units but only 7 days can be used.
        quantity, adjustments = apply_order_constraints(280.0, starving, economics, config)
        names = [a.name for a in adjustments]
        assert "shelf_life_cap" in names
        assert quantity <= 20 * (economics.shelf_life_days + starving.lead_time_days)

    def test_max_days_of_supply_guardrail(self, flat_inputs, economics, config):
        from dataclasses import replace

        durable = economics.model_copy(update={"shelf_life_days": 365})
        starving = replace(flat_inputs, on_hand=0.0, on_order=0.0)
        quantity, adjustments = apply_order_constraints(10_000.0, starving, durable, config)
        assert "max_days_of_supply" in [a.name for a in adjustments]
        assert quantity <= config.inventory.max_order_days_of_supply * 20 + economics.pack_size

    def test_pack_rounding_is_applied(self, flat_inputs, economics, config):
        quantity, adjustments = apply_order_constraints(23.0, flat_inputs, economics, config)
        assert quantity % economics.pack_size == 0
        assert "pack_rounding" in [a.name for a in adjustments]

    def test_minimum_order_quantity_is_enforced(self, flat_inputs, config):

        economics = SkuEconomics(
            sku="Test SKU", unit_cost=2.0, pack_size=1, min_order_quantity=25,
            shelf_life_days=30, holding_cost_per_unit_per_day=0.01,
            waste_cost_per_unit=2.0, stockout_cost_per_unit=4.0,
        )
        quantity, adjustments = apply_order_constraints(8.0, flat_inputs, economics, config)
        assert quantity == 25.0
        assert "min_order_quantity" in [a.name for a in adjustments]

    def test_no_order_stays_no_order(self, flat_inputs, economics, config):
        quantity, adjustments = apply_order_constraints(0.0, flat_inputs, economics, config)
        assert quantity == 0.0 and adjustments == []

    def test_every_adjustment_carries_an_explanation(self, flat_inputs, economics, config):
        _, adjustments = apply_order_constraints(280.0, flat_inputs, economics, config)
        for adjustment in adjustments:
            assert len(adjustment.note) > 20
            assert adjustment.before != adjustment.after


class TestRecommendOrder:
    def test_recommendation_reports_every_intermediate_value(self, flat_inputs, economics, config):
        rec = recommend_order(flat_inputs, economics, config, as_of=AS_OF, model_name="Gradient Boosting")
        assert rec.inventory_position == 40.0
        assert rec.protection_period_days == 3
        assert rec.expected_demand_protection == pytest.approx(60.0)
        assert rec.target_stock == pytest.approx(60.0)
        assert rec.raw_order_quantity == pytest.approx(20.0)
        assert rec.recommended_order_quantity == 20.0
        assert rec.order_value == pytest.approx(40.0)
        assert rec.model_name == "Gradient Boosting"

    def test_days_of_cover_uses_on_hand_only(self, flat_inputs, economics, config):
        rec = recommend_order(flat_inputs, economics, config, as_of=AS_OF)
        assert rec.days_of_cover == pytest.approx(30.0 / 20.0)

    def test_overstocked_sku_needs_no_order(self, flat_inputs, economics, config):
        from dataclasses import replace

        rec = recommend_order(replace(flat_inputs, on_hand=500.0), economics, config, as_of=AS_OF)
        assert rec.recommended_order_quantity == 0.0
        assert rec.order_required is False

    def test_risk_flags_respond_to_the_position(self, flat_inputs, economics, config):
        from dataclasses import replace

        exposed = replace(flat_inputs, on_hand=0.0, on_order=0.0, sigma_daily=8.0)
        assert recommend_order(exposed, economics, config, as_of=AS_OF).stockout_risk == "high"
        safe = replace(flat_inputs, on_hand=200.0, sigma_daily=8.0)
        assert recommend_order(safe, economics, config, as_of=AS_OF).stockout_risk == "low"

    def test_overstock_of_a_perishable_flags_waste_risk(self, flat_inputs, economics, config):
        from dataclasses import replace

        # Far more stock than can be sold within a 5-day shelf life.
        rec = recommend_order(replace(flat_inputs, on_hand=400.0), economics, config, as_of=AS_OF)
        assert rec.projected_waste_units > 0
        assert rec.waste_risk == "high"
        assert rec.risk_level == "high"

    def test_ss_policy_band_is_respected(self, flat_inputs, economics, config):
        from dataclasses import replace

        inputs = replace(flat_inputs, on_hand=60.0, on_order=10.0)
        rec = recommend_order(inputs, economics, config, as_of=AS_OF, policy=SsPolicy())
        assert rec.reorder_point is not None
        assert rec.recommended_order_quantity == 0.0

    def test_frame_conversion(self, flat_inputs, economics, config):
        rec = recommend_order(flat_inputs, economics, config, as_of=AS_OF)
        frame = recommendations_to_frame([rec])
        assert list(frame["sku"]) == ["Test SKU"]
        assert "recommended_order" in frame.columns
        assert recommendations_to_frame([]).empty
