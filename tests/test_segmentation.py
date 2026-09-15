"""Tests for ABC/XYZ segmentation and per-SKU model selection."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import build_sku_master
from src.forecasting.croston import CrostonForecast
from src.forecasting.model_selection import GRADIENT_BOOSTING, ModelChoice, ModelSelector
from src.segmentation import segment_skus, service_level_for_segment, to_segments


@pytest.fixture
def master(synthetic_panel, config):
    return build_sku_master(synthetic_panel, config)


class TestSegmentation:
    def test_every_sku_receives_a_segment(self, synthetic_panel, master, config):
        seg = segment_skus(synthetic_panel, master, config)
        assert set(seg["sku"]) == set(synthetic_panel["sku"].unique())
        assert seg["abc"].isin(["A", "B", "C"]).all()
        assert seg["xyz"].isin(["X", "Y", "Z"]).all()
        assert (seg["segment"] == seg["abc"] + seg["xyz"]).all()

    def test_value_shares_sum_to_one_and_are_ordered(self, synthetic_panel, master, config):
        seg = segment_skus(synthetic_panel, master, config)
        assert seg["value_share"].sum() == pytest.approx(1.0)
        assert seg["annual_value"].is_monotonic_decreasing
        assert seg["cumulative_value_share"].iloc[-1] == pytest.approx(1.0)

    def test_intermittent_sku_is_classified_z(self, synthetic_panel, master, config):
        seg = segment_skus(synthetic_panel, master, config).set_index("sku")
        assert seg.loc["Spiky SKU", "xyz"] == "Z"
        assert seg.loc["Smooth SKU", "xyz"] == "X"
        assert seg.loc["Spiky SKU", "zero_day_share"] > 0.3

    def test_segmentation_uses_only_history_before_as_of(self, synthetic_panel, master, config):
        """A demand explosion after the cut-off must not change the segmentation."""
        as_of = pd.Timestamp("2025-03-01")
        baseline = segment_skus(synthetic_panel, master, config, as_of=as_of)
        tampered = synthetic_panel.copy()
        future = tampered["date"] >= as_of
        tampered.loc[future, "units_sold"] = tampered.loc[future, "units_sold"] * 1000
        after = segment_skus(tampered, master, config, as_of=as_of)
        pd.testing.assert_frame_equal(
            baseline.drop(columns=["as_of"]), after.drop(columns=["as_of"])
        )

    def test_empty_history_raises(self, synthetic_panel, master, config):
        with pytest.raises(ValueError, match="No sales history"):
            segment_skus(synthetic_panel, master, config, as_of=pd.Timestamp("2020-01-01"))

    def test_segment_description_is_readable(self, synthetic_panel, master, config):
        segments = to_segments(segment_skus(synthetic_panel, master, config))
        text = segments["Spiky SKU"].describe()
        assert segments["Spiky SKU"].segment in text
        assert "erratic" in text

    def test_service_level_is_differentiated_by_abc_class(self, synthetic_panel, master, config):
        segments = to_segments(segment_skus(synthetic_panel, master, config))
        levels = {s.abc: service_level_for_segment(s, 0.95) for s in segments.values()}
        for abc, expected in {"A": 0.97, "B": 0.95, "C": 0.92}.items():
            if abc in levels:
                assert levels[abc] == pytest.approx(expected)

    def test_service_level_stays_a_valid_probability(self, synthetic_panel, master, config):
        segments = to_segments(segment_skus(synthetic_panel, master, config))
        for seg in segments.values():
            assert 0.0 < service_level_for_segment(seg, 0.99) < 1.0


class TestModelSelection:
    def test_intermittent_skus_route_to_croston(self, synthetic_panel, master, config):
        choices = ModelSelector(config, "rule").select(synthetic_panel, master)
        assert choices["Spiky SKU"].model_name == "Croston (SBA)"
        assert choices["Smooth SKU"].model_name == GRADIENT_BOOSTING

    def test_every_choice_records_a_reason_and_segment(self, synthetic_panel, master, config):
        choices = ModelSelector(config, "rule").select(synthetic_panel, master)
        for choice in choices.values():
            assert len(choice.reason) > 40
            assert choice.segment
            assert choice.strategy == "rule"

    def test_reason_explains_the_intermittency_threshold(self, synthetic_panel, master, config):
        choices = ModelSelector(config, "rule").select(synthetic_panel, master)
        assert "no demand" in choices["Spiky SKU"].reason

    def test_build_returns_the_named_model(self, config):
        selector = ModelSelector(config)
        croston = selector.build(
            ModelChoice("s", "Croston (SBA)", "r", "CZ", "rule", "2025-01-01")
        )
        assert isinstance(croston, CrostonForecast) and croston.variant == "sba"
        # The global model is owned by the service, not built per SKU.
        assert selector.build(ModelChoice("s", GRADIENT_BOOSTING, "r", "AX", "rule", "2025-01-01")) is None

    def test_unknown_model_is_rejected(self, config):
        with pytest.raises(ValueError, match="unknown model"):
            ModelSelector(config).build(ModelChoice("s", "Prophet", "r", "AX", "rule", "2025-01-01"))

    def test_selection_round_trips_through_disk(self, synthetic_panel, master, config, tmp_path):
        choices = ModelSelector(config, "rule").select(synthetic_panel, master)
        path = tmp_path / "registry" / "model_selection.json"
        ModelSelector.save(choices, path)
        loaded = ModelSelector.load(path)
        assert loaded == choices

    def test_missing_registry_gives_an_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_models.py"):
            ModelSelector.load(tmp_path / "nope.json")


def _pinned_to_gb() -> dict[str, ModelChoice]:
    """Route both fixture SKUs to gradient boosting, whatever the decision date."""
    return {
        sku: ModelChoice(sku, GRADIENT_BOOSTING, "pinned by test", "AX", "rule", "2025-01-01")
        for sku in ("Smooth SKU", "Spiky SKU")
    }


class TestForecastService:
    def test_service_uses_the_selected_model_per_sku(self, synthetic_panel, config):
        from src.forecasting.service import ForecastService

        service = ForecastService(synthetic_panel, config)
        results = service.forecast(pd.Timestamp("2025-04-01"))
        assert results["Spiky SKU"].model_name == "Croston (SBA)"
        assert results["Smooth SKU"].model_name == GRADIENT_BOOSTING
        assert all(r.horizon == config.forecast.horizon_days for r in results.values())

    def test_service_refits_only_on_the_configured_cadence(self, synthetic_panel, config):
        from src.forecasting.service import ForecastService

        service = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
        cadence = pd.Timedelta(days=config.forecast.refit_frequency_days)
        anchor = service.refit_anchor(pd.Timestamp("2025-04-02"))
        service.forecast(anchor, ["Smooth SKU"])
        assert service.last_fit_date == anchor
        service.forecast(anchor + cadence - pd.Timedelta(days=1), ["Smooth SKU"])
        assert service.last_fit_date == anchor  # same refit window, same model
        service.forecast(anchor + cadence, ["Smooth SKU"])
        assert service.last_fit_date == anchor + cadence

    def test_moving_backwards_in_time_forces_a_refit(self, synthetic_panel, config):
        """The refit cadence must not keep a model fitted on data the new date cannot see."""
        from src.forecasting.service import ForecastService

        # Pin the routing so this exercises the fit cache alone, not model selection.
        service = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
        late, early = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-03-01")
        service.forecast(late, ["Smooth SKU"])
        assert service.last_fit_date == service.refit_anchor(late)
        service.forecast(early, ["Smooth SKU"])
        assert service.last_fit_date == service.refit_anchor(early) <= early

    def test_a_forecast_does_not_depend_on_which_dates_were_viewed_first(
        self, synthetic_panel, config
    ):
        """Any decision date can be revisited in the dashboard; the forecast must not move."""
        from src.forecasting.service import ForecastService

        early, late = pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")

        clean = ForecastService(synthetic_panel, config).forecast_sku(early, "Smooth SKU")

        revisited = ForecastService(synthetic_panel, config)
        revisited.forecast_sku(late, "Smooth SKU")
        warmed = revisited.forecast_sku(early, "Smooth SKU")

        assert list(warmed.predicted) == pytest.approx(list(clean.predicted))
        assert warmed.sigma == pytest.approx(clean.sigma)

    def test_explaining_a_forecast_does_not_reuse_a_later_model(self, synthetic_panel, config):
        """explain_forecast must honour the staleness check, not read the cache directly."""
        from src.forecasting.service import ForecastService

        service = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
        service.forecast_sku(pd.Timestamp("2025-04-01"), "Smooth SKU")
        service.explain_forecast(pd.Timestamp("2025-03-01"), "Smooth SKU")
        assert service.last_fit_date == service.refit_anchor(pd.Timestamp("2025-03-01"))

    def test_explaining_a_memoised_forecast_still_refits_the_model(self, synthetic_panel, config):
        """A memoised forecast skips the refit, so the explanation must force one itself.

        Without this, asking for the earlier date's forecast first (which memoises it)
        and only then explaining it would hand the explainer the later model.
        """
        from src.forecasting.service import ForecastService

        early, late = pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")
        service = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
        service.forecast_sku(early, "Smooth SKU")  # memoises the early forecast
        service.forecast_sku(late, "Smooth SKU")  # refits at the later date
        assert service.last_fit_date == service.refit_anchor(late)
        service.explain_forecast(early, "Smooth SKU")
        assert service.last_fit_date == service.refit_anchor(early)

    def test_refit_anchor_depends_only_on_the_date(self, synthetic_panel, config):
        """Anchors sit on a fixed grid, never after the date, and ignore service history."""
        from src.forecasting.service import ForecastService

        cadence = config.forecast.refit_frequency_days
        origin = pd.Timestamp(config.simulation.start_date)
        fresh = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
        used = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
        used.forecast(pd.Timestamp("2025-04-20"), ["Smooth SKU"])
        for day in pd.date_range("2025-03-10", "2025-04-30"):
            anchor = fresh.refit_anchor(day)
            assert anchor <= day
            assert (day - anchor).days < cadence
            assert (anchor - origin).days % cadence == 0
            assert used.refit_anchor(day) == anchor

    def test_forecast_is_identical_whatever_was_forecast_before(self, synthetic_panel, config):
        """Regression: an earlier forecast used to leave a model that a later date reused.

        Forecasting 8 April and then 10 April served 10 April from the 8 April model, while a
        fresh service fitted at 10 April itself, so the same date gave two forecasts.
        """
        from src.forecasting.service import ForecastService

        target = pd.Timestamp("2025-04-10")
        fresh = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb()).forecast_sku(target, "Smooth SKU")
        for earlier in (pd.Timestamp("2025-04-08"), pd.Timestamp("2025-04-09"), pd.Timestamp("2025-03-20")):
            service = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
            service.forecast_sku(earlier, "Smooth SKU")
            got = service.forecast_sku(target, "Smooth SKU")
            assert list(got.predicted) == pytest.approx(list(fresh.predicted)), earlier
            assert got.sigma == pytest.approx(fresh.sigma), earlier

    def test_forecasting_before_any_history_is_rejected(self, synthetic_panel, config):
        from src.forecasting.service import ForecastService

        service = ForecastService(synthetic_panel, config)
        with pytest.raises(ValueError, match="No sales history"):
            service.forecast(pd.Timestamp("2024-01-01"))


@pytest.mark.slow
class TestForecastDeterminismEval:
    """Property check that generalises the refit regression beyond the dates it names."""

    def test_forecasts_match_fresh_ones_in_any_order(self, synthetic_panel, config):
        """Whatever order dates are forecast in, and whether or not the memo serves them,
        each forecast equals the one a fresh service produces for that date."""
        import random

        from src.forecasting.service import ForecastService

        days = list(pd.date_range("2025-03-05", "2025-04-26", freq="4D"))
        random.Random(42).shuffle(days)
        want = {
            day: ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb()).forecast_sku(day, "Smooth SKU")
            for day in days
        }
        shared = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
        for day in days + days[::-1]:  # the reversed second pass is served from the memo
            got = shared.forecast_sku(day, "Smooth SKU")
            assert list(got.predicted) == pytest.approx(list(want[day].predicted)), day
            assert got.sigma == pytest.approx(want[day].sigma), day


def test_lazy_model_choices_do_not_depend_on_request_order(synthetic_panel, config):
    """Regression: a routing chosen lazily at an earlier date was reused for later dates.

    Forecasting 20 February and then 25 April routed Smooth SKU with the February
    selection, while a fresh service selected for 25 April, so one date gave two forecasts.
    """
    from src.forecasting.service import ForecastService

    target = pd.Timestamp("2025-04-25")
    want = ForecastService(synthetic_panel, config).forecast(target)
    used = ForecastService(synthetic_panel, config)
    used.forecast(pd.Timestamp("2025-02-20"))
    got = used.forecast(target)
    for sku in want:
        assert list(got[sku].predicted) == pytest.approx(list(want[sku].predicted)), sku


def test_a_date_just_after_the_data_start_can_still_be_forecast(synthetic_panel, config):
    """Regression: an anchor a few days after the first fittable date had too little history."""
    from src.forecasting.service import ForecastService

    day = pd.Timestamp("2025-01-30")
    service = ForecastService(synthetic_panel, config, model_choices=_pinned_to_gb())
    assert service.refit_anchor(day) == day
    result = service.forecast_sku(day, "Smooth SKU")
    assert len(result.predicted) == config.forecast.horizon_days
