"""Repairing an imperfect feed, running the pipeline as one command, and the order sheet.

These three cover what a deployed run needs beyond the research pipeline: data that arrives
damaged must not stop the run, the steps must run in order with a non-zero exit when one
fails, and the kitchen must be able to take the order off the screen.
"""
from __future__ import annotations

import io
import pathlib

import numpy as np
import pandas as pd
import pytest

from scripts.run_all import run, steps
from src.app_services import apply_overrides, order_sheet
from src.data.repair import IMPUTED_COLUMN, repair_sales_frame
from src.data.validation import validate_sales_frame
from src.inventory.ordering import OrderRecommendation


def feed(config, days=28, skus=("Milk", "Bacon")):
    """A small well-formed feed shaped like the real one."""
    dates = pd.date_range("2025-01-06", periods=days, freq="D")
    rows = [
        {
            "date": day,
            "sku": sku,
            "category": "Dairy",
            "shelf_life_days": 5,
            "units_sold": 40 + 10 * (day.dayofweek >= 5) + i,
            "temp_c": 12.0,
            "bank_holiday": 0,
            "school_holiday": 0,
            "weekend": int(day.dayofweek >= 5),
            "dow": day.dayofweek,
            "promo": 0,
        }
        for i, sku in enumerate(skus)
        for day in dates
    ]
    return pd.DataFrame(rows).sort_values(["sku", "date"]).reset_index(drop=True)


class TestRepairingTheFeed:
    def test_a_clean_feed_is_left_alone(self, config):
        original = feed(config)
        repaired, report = repair_sales_frame(original, config)
        assert report.clean and report.summary() == "Sales feed needed no repair."
        pd.testing.assert_frame_equal(
            repaired.drop(columns=[IMPUTED_COLUMN]), original, check_dtype=False
        )

    def test_missing_day_is_filled_so_lags_stay_aligned(self, config):
        original = feed(config)
        gap_day = pd.Timestamp("2025-01-15")
        damaged = original[~((original["sku"] == "Milk") & (original["date"] == gap_day))]

        repaired, report = repair_sales_frame(damaged, config)

        assert report.missing_days_filled == 1
        milk = repaired[repaired["sku"] == "Milk"].set_index("date")
        assert gap_day in milk.index, "the hole must be filled, not dropped: lag7 is positional"
        filled = milk.loc[gap_day]
        assert filled[IMPUTED_COLUMN]
        # Filled from the same weekday recently, so it is a plausible quantity, not zero.
        assert filled["units_sold"] > 0
        # The calendar comes from the other product, which recorded that date.
        assert filled["dow"] == gap_day.dayofweek
        assert not repaired[repaired["sku"] == "Bacon"][IMPUTED_COLUMN].any()

    def test_invalid_and_duplicate_rows_are_dropped_and_counted(self, config):
        original = feed(config)
        repeated, negative = original.iloc[[3]], original.iloc[[5]].assign(units_sold=-7)
        damaged = pd.concat([original, repeated, negative], ignore_index=True)

        repaired, report = repair_sales_frame(damaged, config)

        assert report.duplicate_rows_dropped == 1
        assert report.invalid_rows_dropped == 1
        assert not repaired.duplicated(subset=["date", "sku"]).any()
        assert (repaired["units_sold"] >= 0).all()
        assert set(report.skus_affected) == {repeated["sku"].iloc[0], negative["sku"].iloc[0]}

    def test_null_quantity_is_not_read_as_zero_demand(self, config):
        original = feed(config).astype({"units_sold": "float64"})
        damaged = original.copy()
        damaged.loc[10, "units_sold"] = np.nan
        day, sku = damaged.loc[10, "date"], damaged.loc[10, "sku"]

        repaired, report = repair_sales_frame(damaged, config)

        assert report.invalid_rows_dropped == 1
        restored = repaired[(repaired["sku"] == sku) & (repaired["date"] == day)]
        assert restored[IMPUTED_COLUMN].all(), "the day is refilled by estimate, not left at zero"
        assert float(restored["units_sold"].iloc[0]) > 0

    def test_repaired_feed_passes_validation(self, config):
        original = feed(config)
        damaged = pd.concat([original.drop(index=range(4, 7)), original.iloc[[9]]], ignore_index=True)
        repaired, report = repair_sales_frame(damaged, config)
        assert not report.clean
        validate_sales_frame(repaired.drop(columns=[IMPUTED_COLUMN]), config)

    def test_repair_is_idempotent(self, config):
        damaged = feed(config).drop(index=[11, 12])
        once, first = repair_sales_frame(damaged, config)
        twice, second = repair_sales_frame(once, config)
        assert not first.clean and second.clean
        pd.testing.assert_frame_equal(once, twice, check_dtype=False)

    def test_a_frame_without_quantities_cannot_be_repaired(self, config):
        with pytest.raises(ValueError, match="units_sold"):
            repair_sales_frame(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "sku": ["Milk"]}), config)


class TestRunningThePipeline:
    def test_steps_run_in_dependency_order(self):
        names = [command[-2] if command[-1] == "--frontier" else command[-1] for _, command in steps()]
        assert [n.split("/")[-1] for n in names] == [
            "train_models.py", "run_backtest.py", "run_simulation.py", "generate_report_figures.py"
        ]

    def test_frontier_sweep_is_requested(self):
        simulation = next(command for _, command in steps() if "run_simulation.py" in command[1])
        assert "--frontier" in simulation

    def test_config_is_passed_to_each_step(self):
        planned = steps("config/other.yaml")
        assert all("--config" in command for _, command in planned if "generate_report_figures" not in command[1])

    def test_figures_can_be_skipped(self):
        assert not any("generate_report_figures" in command[1] for _, command in steps(skip_figures=True))

    def test_a_failing_step_stops_the_run_and_exits_non_zero(self):
        log = io.StringIO()
        attempted = []

        def runner(command, _log):
            attempted.append(command[-1])
            return 0 if len(attempted) == 1 else 3

        code = run(steps(), log, runner=runner)

        assert code == 3
        assert len(attempted) == 2, "later steps must not run on stale outputs"
        assert "FAILED at step 2" in log.getvalue()

    def test_a_clean_run_reports_success(self):
        log = io.StringIO()
        assert run(steps(), log, runner=lambda command, _log: 0) == 0
        assert "All steps completed." in log.getvalue()


class TestOrderSheet:
    def recommendation(self, sku, quantity, unit_cost=1.1, pack_size=10, min_order=10):
        return OrderRecommendation(
            sku=sku, as_of=pd.Timestamp("2025-12-05"), policy_name="Base stock (R, S)",
            model_name="Gradient Boosting", on_hand=5.0, on_order=0.0, backorders=0.0,
            inventory_position=5.0, lead_time_days=2, review_period_days=2,
            protection_period_days=4, service_level=0.95, sigma_daily=3.0,
            daily_forecast=np.array([10.0] * 7), expected_demand_protection=40.0,
            expected_demand_horizon=70.0, safety_stock=6.0, target_stock=46.0,
            reorder_point=None, raw_order_quantity=quantity, recommended_order_quantity=quantity,
            adjustments=[], unit_cost=unit_cost, pack_size=pack_size, min_order_quantity=min_order,
            shelf_life_days=5, order_value=quantity * unit_cost, waste_cost_per_unit=unit_cost * 1.05,
            stockout_cost_per_unit=unit_cost * 2, days_of_cover=3.5, stockout_probability=0.1,
            projected_waste_units=0.0, stockout_risk="low", waste_risk="low",
        )

    def test_only_products_with_an_order_appear_largest_first(self):
        recommendations = [
            self.recommendation("Milk", 30.0),
            self.recommendation("Bacon", 0.0),
            self.recommendation("Onions", 50.0),
        ]
        sheet = order_sheet(apply_overrides(recommendations, {}), recommendations, pd.Timestamp("2025-12-05"))
        assert sheet["Product"].tolist() == ["Onions", "Milk"]
        assert sheet["Order (units)"].tolist() == [50, 30]
        assert sheet["Order date"].unique().tolist() == ["2025-12-05"]

    def test_an_override_stays_visible_on_the_sheet(self):
        recommendations = [self.recommendation("Milk", 30.0)]
        view = apply_overrides(recommendations, {"Milk": 80.0})
        sheet = order_sheet(view, recommendations, pd.Timestamp("2025-12-05"))
        row = sheet.iloc[0]
        assert row["Order (units)"] == 80 and row["Recommended (units)"] == 30
        assert row["Overridden"] == "yes"
        assert row["Line value (GBP)"] == pytest.approx(88.0, abs=0.01)

    def test_supplier_terms_travel_with_each_line(self):
        recommendations = [self.recommendation("Milk", 30.0, unit_cost=0.8, pack_size=6, min_order=12)]
        sheet = order_sheet(apply_overrides(recommendations, {}), recommendations, pd.Timestamp("2025-12-05"))
        assert sheet.iloc[0]["Pack size"] == 6 and sheet.iloc[0]["Minimum order"] == 12
        assert sheet.iloc[0]["Unit cost (GBP)"] == pytest.approx(0.8)

    def test_nothing_to_order_gives_an_empty_sheet_not_an_error(self):
        recommendations = [self.recommendation("Milk", 0.0)]
        sheet = order_sheet(apply_overrides(recommendations, {}), recommendations, pd.Timestamp("2025-12-05"))
        assert sheet.empty and "Product" in sheet.columns
        assert sheet.to_csv(index=False).startswith("Order date,Product")


@pytest.mark.skipif(not (pathlib.Path(__file__).resolve().parents[1] / "data" / "sales_data.csv").exists(),
                    reason="project dataset not present")
def test_repair_leaves_the_projects_own_dataset_untouched(config):
    """Eval: turning repair on must not quietly change the data every reported result rests on."""
    from src.data.loader import load_sales

    plain = load_sales(config=config)
    repaired = load_sales(config=config, repair=True)
    pd.testing.assert_frame_equal(repaired.drop(columns=[IMPUTED_COLUMN]), plain, check_dtype=False)
    assert not repaired[IMPUTED_COLUMN].any()
