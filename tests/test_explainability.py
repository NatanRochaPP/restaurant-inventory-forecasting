"""Tests for forecast-driver attribution and recommendation reasoning.

Alongside the mechanics, these tests police the language: the system must describe
associations, never causes, and must never imply that an order has been placed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import build_sku_master
from src.data.preprocessing import history_as_of
from src.explainability.forecast_drivers import (
    EXPLAINABLE_FEATURES,
    aggregate_drivers,
    build_reference_row,
    explain_croston,
    explain_horizon,
    explain_simple_model,
)
from src.explainability.recommendation_reason import (
    explain_recommendation,
    explanation_table,
)
from src.forecasting.seasonal_naive import SeasonalNaiveForecast
from src.forecasting.service import ForecastService
from src.inventory.ordering import recommend_order
from src.inventory.policies import PolicyInputs, SsPolicy
from src.segmentation import segment_skus, to_segments

AS_OF = pd.Timestamp("2025-04-01")

#: Words that would overstate what a feature-importance measure can support.
CAUSAL_WORDS = ("causes", "caused", "because the model", "proves", "guarantees")


@pytest.fixture
def service(synthetic_panel, config):
    return ForecastService(synthetic_panel, config)


@pytest.fixture
def gb_setup(service, synthetic_panel, config):
    result, explanations = service.explain_forecast(AS_OF, "Smooth SKU")
    return result, explanations


class TestForecastDrivers:
    def test_every_explainable_feature_receives_a_contribution(self, gb_setup):
        _, explanations = gb_setup
        features = {d.feature for d in explanations[0].drivers}
        # dow and weekend overlap, so one of the pair is dropped by design.
        assert features <= set(EXPLAINABLE_FEATURES)
        assert len(features) >= len(EXPLAINABLE_FEATURES) - 1

    def test_one_explanation_per_horizon_day(self, gb_setup, config):
        _, explanations = gb_setup
        assert len(explanations) == config.forecast.horizon_days

    def test_weekend_contributes_positively_for_a_weekend_uplift_sku(self, gb_setup):
        """The synthetic panel has a 50% Friday-Sunday uplift; drivers should find it."""
        _, explanations = gb_setup
        weekend_days = [e for e in explanations if pd.Timestamp(e.date).dayofweek in (4, 5, 6)]
        assert weekend_days, "expected a weekend day in the horizon"
        for explanation in weekend_days:
            calendar = [d for d in explanation.drivers if d.feature in ("weekend", "dow")]
            assert calendar and max(d.contribution_units for d in calendar) > 0

    def test_contributions_are_measured_against_a_neutral_day(self, gb_setup):
        _, explanations = gb_setup
        for explanation in explanations:
            assert explanation.reference_prediction > 0
            assert np.isfinite(explanation.unexplained_units)

    def test_unexplained_residual_is_reported_not_hidden(self, gb_setup):
        """Ablation is not additive; the gap must be visible rather than absorbed."""
        _, explanations = gb_setup
        explanation = explanations[0]
        attributed = sum(d.contribution_units for d in explanation.drivers)
        expected_gap = explanation.predicted - explanation.reference_prediction - attributed
        assert explanation.unexplained_units == pytest.approx(expected_gap)

    def test_reference_row_neutralises_calendar_and_promotions(self, service, synthetic_panel, config):
        future = service.build_future_frame(AS_OF, 7, ["Smooth SKU"])
        history = history_as_of(synthetic_panel, AS_OF, sku="Smooth SKU")
        reference = build_reference_row(future.iloc[0], history, config)
        assert reference["promo"] == 0.0
        assert reference["bank_holiday"] == 0.0
        assert reference["weekend"] == 0.0
        assert reference["temp_c"] == pytest.approx(history["temp_c"].mean())

    def test_narrative_is_readable_and_non_causal(self, gb_setup):
        _, explanations = gb_setup
        text = explanations[0].narrative()
        assert "contributed most" in text
        assert len(text.splitlines()) >= 2
        lowered = text.lower()
        assert not any(word in lowered for word in CAUSAL_WORDS)

    def test_explanation_note_states_the_limitation(self, gb_setup):
        _, explanations = gb_setup
        assert "not proof of cause" in explanations[0].note

    def test_drivers_convert_to_a_chartable_frame(self, gb_setup):
        _, explanations = gb_setup
        frame = explanations[0].to_frame()
        assert list(frame.columns) == ["feature", "label", "value", "contribution_units", "description"]
        assert frame["contribution_units"].abs().is_monotonic_decreasing

    def test_aggregate_drivers_totals_across_the_horizon(self, gb_setup):
        _, explanations = gb_setup
        aggregated = aggregate_drivers(explanations)
        assert not aggregated.empty
        assert aggregated["contribution_units"].abs().is_monotonic_decreasing
        one_feature = explanations[0].drivers[0].feature
        expected = sum(
            d.contribution_units for e in explanations for d in e.drivers if d.feature == one_feature
        )
        assert aggregated.set_index("feature").loc[one_feature, "contribution_units"] == pytest.approx(expected)

    def test_aggregate_of_featureless_models_is_empty_not_an_error(self, synthetic_panel, config, service):
        result = service.forecast_sku(AS_OF, "Spiky SKU")
        explanations = explain_horizon(result, pd.DataFrame(), pd.DataFrame(), config, None)
        assert aggregate_drivers(explanations).empty


class TestIntermittentAndSimpleModelExplanations:
    def test_croston_is_explained_through_its_parameters(self, synthetic_panel, config, service):
        result = service.forecast_sku(AS_OF, "Spiky SKU")
        explanation = explain_croston(result)
        assert "intermittently" in explanation.note
        assert "units per day" in explanation.note
        assert explanation.drivers == []

    def test_seasonal_naive_explanation_states_the_rule(self, synthetic_panel, config):
        history = history_as_of(synthetic_panel, AS_OF, sku="Smooth SKU")
        future = pd.DataFrame({"date": pd.date_range(AS_OF, periods=7), "sku": "Smooth SKU"})
        result = SeasonalNaiveForecast(7).fit(history).predict(future)
        explanation = explain_simple_model(result)
        assert "previous week" in explanation.note

    def test_horizon_explainer_routes_by_model_type(self, service, synthetic_panel, config):
        result, explanations = service.explain_forecast(AS_OF, "Spiky SKU")
        assert result.model_name.startswith("Croston")
        assert len(explanations) == 1 and explanations[0].drivers == []


class TestRecommendationReason:
    @pytest.fixture
    def recommendation(self, service, config):
        result = service.forecast_sku(AS_OF, "Smooth SKU")
        inputs = PolicyInputs(
            sku="Smooth SKU", daily_forecast=result.predicted, sigma_daily=result.sigma,
            on_hand=60.0, on_order=20.0, lead_time_days=2, review_period_days=2, service_level=0.95,
        )
        return recommend_order(
            inputs, config.sku_economics("Smooth SKU", 5), config,
            as_of=AS_OF, model_name=result.model_name,
        ), result

    def test_headline_states_the_quantity_and_value(self, recommendation):
        rec, _ = recommendation
        explanation = explain_recommendation(rec)
        assert f"{rec.recommended_order_quantity:.0f}" in explanation.headline
        assert rec.sku in explanation.headline

    def test_explanation_covers_demand_cover_and_risk(self, recommendation, service):
        rec, result = recommendation
        explanation = explain_recommendation(
            rec,
            recent_average_daily=service.recent_average_demand(AS_OF, "Smooth SKU"),
            forecast_dates=result.dates,
        )
        text = explanation.full_text()
        assert "covers approximately" in text          # current cover
        assert "protection period" in text             # why the target is what it is
        assert "safety stock" in text
        assert "service level" in text
        assert explanation.risk_reason

    def test_explanation_never_claims_an_order_was_placed(self, recommendation):
        rec, _ = recommendation
        text = explain_recommendation(rec).full_text().lower()
        assert "recommendation only" in text
        assert "no order is placed automatically" in text
        assert "order has been placed" not in text

    def test_explanation_language_stays_associative(self, recommendation, service):
        rec, result = recommendation
        explanation = explain_recommendation(
            rec, recent_average_daily=service.recent_average_demand(AS_OF, "Smooth SKU"),
            forecast_dates=result.dates,
        )
        lowered = explanation.full_text().lower()
        assert not any(word in lowered for word in CAUSAL_WORDS)
        assert "not proof of cause" in lowered

    def test_demand_reason_compares_against_the_recent_average(self, recommendation):
        rec, result = recommendation
        high = explain_recommendation(rec, recent_average_daily=1.0)
        assert "higher than the recent average" in high.demand_reason[0]
        low = explain_recommendation(rec, recent_average_daily=10_000.0)
        assert "lower than the recent average" in low.demand_reason[0]

    def test_constraints_are_listed_when_they_bind(self, service, config):
        result = service.forecast_sku(AS_OF, "Smooth SKU")
        inputs = PolicyInputs(
            sku="Smooth SKU", daily_forecast=result.predicted, sigma_daily=result.sigma,
            on_hand=0.0, lead_time_days=2, review_period_days=2, service_level=0.95,
        )
        economics = config.sku_economics("Smooth SKU", 2)  # very short shelf life
        rec = recommend_order(inputs, economics, config, as_of=AS_OF)
        explanation = explain_recommendation(rec)
        assert explanation.constraint_reason
        assert "unconstrained calculation" in explanation.constraint_reason[0]

    def test_no_order_case_is_explained(self, service, config):
        result = service.forecast_sku(AS_OF, "Smooth SKU")
        inputs = PolicyInputs(
            sku="Smooth SKU", daily_forecast=result.predicted, sigma_daily=result.sigma,
            on_hand=5000.0, lead_time_days=2, review_period_days=2, service_level=0.95,
        )
        rec = recommend_order(inputs, config.sku_economics("Smooth SKU", 5), config, as_of=AS_OF)
        explanation = explain_recommendation(rec)
        assert "No order needed" in explanation.headline

    def test_ss_policy_no_order_mentions_the_reorder_point(self, service, config):
        result = service.forecast_sku(AS_OF, "Smooth SKU")
        inputs = PolicyInputs(
            sku="Smooth SKU", daily_forecast=result.predicted, sigma_daily=result.sigma,
            on_hand=5000.0, lead_time_days=2, review_period_days=2, service_level=0.95,
        )
        rec = recommend_order(
            inputs, config.sku_economics("Smooth SKU", 5), config, as_of=AS_OF, policy=SsPolicy()
        )
        assert "reorder point" in explain_recommendation(rec).headline

    def test_segment_is_mentioned_when_supplied(self, recommendation, synthetic_panel, config):
        rec, _ = recommendation
        master = build_sku_master(synthetic_panel, config)
        segments = to_segments(segment_skus(synthetic_panel, master, config, as_of=AS_OF))
        explanation = explain_recommendation(rec, segment=segments["Smooth SKU"])
        assert any("availability target" in line for line in explanation.cover_reason)


class TestExplanationTable:
    def test_table_has_the_columns_the_dashboard_shows(self, service, config):
        results = service.forecast(AS_OF)
        recommendations = []
        for sku, result in results.items():
            inputs = PolicyInputs(
                sku=sku, daily_forecast=result.predicted, sigma_daily=result.sigma,
                on_hand=40.0, lead_time_days=2, review_period_days=2, service_level=0.95,
            )
            recommendations.append(
                recommend_order(inputs, config.sku_economics(sku, 5), config, as_of=AS_OF,
                                model_name=result.model_name)
            )
        table = explanation_table(recommendations)
        assert list(table.columns) == [
            "SKU", "Current Stock", "On Order", "Forecast Demand", "Recommended Order",
            "Order Value", "Risk", "Model", "Reason",
        ]
        assert len(table) == len(recommendations)
        assert table["Reason"].str.len().min() > 20
