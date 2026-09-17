"""Money saved by following the AI policy: sign, completeness and agreement with real runs."""
from pathlib import Path

import pandas as pd
import pytest

from src.app_services import MONEY_KEYS, add_cost_saved, money_saved
from src.evaluation.inventory_metrics import compute_kpis

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


def kpis(**costs):
    base = dict.fromkeys(MONEY_KEYS, 0.0)
    base.update(costs)
    return base


def test_saving_is_baseline_minus_ai():
    saved = money_saved(kpis(waste_cost=100.0, total_cost=500.0), kpis(waste_cost=40.0, total_cost=450.0))
    assert saved["waste_cost"] == pytest.approx(60.0)
    assert saved["total_cost"] == pytest.approx(50.0)


def test_extra_cost_is_negative_not_hidden():
    # The AI policy wastes less but runs short more: the stockout line must show the extra cost.
    saved = money_saved(kpis(waste_cost=100.0, stockout_cost=80.0), kpis(waste_cost=40.0, stockout_cost=110.0))
    assert saved["waste_cost"] == pytest.approx(60.0)
    assert saved["stockout_cost"] == pytest.approx(-30.0)


def test_every_cost_measure_is_returned():
    assert set(money_saved(kpis(), kpis())) == set(MONEY_KEYS)


def test_missing_cost_measure_raises():
    incomplete = kpis()
    del incomplete["stockout_cost"]
    with pytest.raises(KeyError, match="stockout_cost"):
        money_saved(incomplete, kpis())


def test_matched_cost_saved_column():
    matched = pd.DataFrame({"total_cost_baseline": [22501.94, 18201.83], "total_cost_ai": [20081.46, 16030.74]})
    out = add_cost_saved(matched)
    assert out["total_cost_saved"].tolist() == pytest.approx([2420.48, 2171.09])
    assert "total_cost_saved" not in matched.columns  # input is not modified


def test_matched_cost_saved_requires_cost_columns():
    with pytest.raises(KeyError):
        add_cost_saved(pd.DataFrame({"total_cost_ai": [1.0]}))


@pytest.mark.skipif(not (OUTPUTS / "simulation_daily_ai.csv").exists(), reason="pipeline outputs not generated")
def test_saving_matches_the_recorded_simulation_runs():
    # Eval against the real six-month replay: the savings must reproduce the published comparison.
    base = compute_kpis(pd.read_csv(OUTPUTS / "simulation_daily_baseline.csv", parse_dates=["date"]))
    ai = compute_kpis(pd.read_csv(OUTPUTS / "simulation_daily_ai.csv", parse_dates=["date"]))
    saved = money_saved(base, ai)
    comparison = pd.read_csv(OUTPUTS / "simulation_comparison.csv").set_index("key")
    labels = [c for c in comparison.columns if c not in {"metric", "unit", "improvement_pct", "direction"}]
    for key in ("waste_cost", "total_cost"):
        recorded = comparison.loc[key, labels[0]] - comparison.loc[key, labels[1]]
        assert saved[key] == pytest.approx(recorded, abs=0.01), key
    assert sum(saved[k] for k in MONEY_KEYS if k != "total_cost") == pytest.approx(saved["total_cost"], abs=0.01)
