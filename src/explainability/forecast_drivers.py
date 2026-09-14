"""Explaining why demand was forecast at a particular level.

Feature attribution here is computed by *counterfactual ablation*: the model is asked to
predict the day twice, once with the real feature value and once with that feature reset
to a neutral reference (a typical mid-week, non-holiday, non-promotion day at seasonal
average temperature). The difference is reported as that feature's contribution.

Two honesty constraints are built in:

* Ablation contributions do **not** sum exactly to the prediction, because the model
  contains interactions. They are presented as directional influences, never as an exact
  decomposition, and the residual is reported explicitly.
* The wording throughout says a factor "contributed to" the prediction. A gradient
  boosting model identifies association in historical data; it does not establish that
  the weekend *causes* the demand.

Models without features - the Croston family used for intermittent SKUs - are explained
through their own parameters (average demand size and interval) instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.features.calendar import DAY_NAMES, describe_calendar_day
from src.features.demand_features import FEATURE_LABELS, feature_columns
from src.forecasting.base import ForecastResult
from src.forecasting.gradient_boosting import GradientBoostingForecast

logger = logging.getLogger(__name__)

#: Features a restaurant manager can act on or reason about. Identity and calendar
#: bookkeeping columns (sku_code, week_of_year, month) are excluded from the narrative.
EXPLAINABLE_FEATURES: tuple[str, ...] = (
    "weekend", "bank_holiday", "school_holiday", "is_closed", "promo", "temp_c", "dow",
    "lag7", "lag14", "roll28",
)

#: The neutral day contributions are measured against: an open, mid-week, non-holiday,
#: non-promotion day.
NEUTRAL_CALENDAR: dict[str, float] = {
    "weekend": 0.0,
    "bank_holiday": 0.0,
    "school_holiday": 0.0,
    "is_closed": 0.0,
    "promo": 0.0,
    "dow": 2.0,  # Wednesday
}


@dataclass(frozen=True)
class DriverContribution:
    """One feature's estimated influence on a single day's forecast."""

    feature: str
    label: str
    value: float
    reference_value: float
    contribution_units: float

    @property
    def direction(self) -> str:
        return "increased" if self.contribution_units > 0 else "reduced"

    def describe(self) -> str:
        """Manager-facing phrasing. Deliberately associative, never causal."""
        magnitude = abs(self.contribution_units)
        match self.feature:
            case "weekend":
                context = "It is a Friday-to-Sunday trading day" if self.value else "It is a midweek day"
            case "bank_holiday":
                context = "It is a bank holiday" if self.value else "It is not a bank holiday"
            case "school_holiday":
                context = "It falls in the school holidays" if self.value else "It is term time"
            case "is_closed":
                context = "The site is closed that day" if self.value else "The site is trading"
            case "promo":
                context = "A promotion is running" if self.value else "No promotion is running"
            case "temp_c":
                context = f"Forecast temperature of {self.value:.0f}°C (typical is {self.reference_value:.0f}°C)"
            case "dow":
                context = f"It is a {DAY_NAMES[int(self.value)]}"
            case "lag7":
                context = f"Demand 7 days ago was {self.value:.0f} units"
            case "lag14":
                context = f"Demand 14 days ago was {self.value:.0f} units"
            case "roll28":
                context = f"The 28-day average is {self.value:.0f} units/day"
            case _:
                context = f"{self.label} = {self.value:.1f}"
        return f"{context}, which {self.direction} the forecast by about {magnitude:.0f} units."


@dataclass(frozen=True)
class ForecastExplanation:
    """Driver breakdown for one forecast day."""

    sku: str
    date: pd.Timestamp
    model_name: str
    predicted: float
    reference_prediction: float
    drivers: list[DriverContribution] = field(default_factory=list)
    note: str = ""

    @property
    def unexplained_units(self) -> float:
        """Difference between the prediction and the sum of the reported influences.

        Non-zero because the model contains interactions between features; reported so
        the breakdown is never mistaken for an exact decomposition.
        """
        attributed = sum(d.contribution_units for d in self.drivers)
        return float(self.predicted - self.reference_prediction - attributed)

    def top_drivers(self, n: int = 4) -> list[DriverContribution]:
        """The ``n`` most influential features by absolute contribution."""
        return sorted(self.drivers, key=lambda d: abs(d.contribution_units), reverse=True)[:n]

    def narrative(self, n: int = 3) -> str:
        """A short paragraph a restaurant manager can read."""
        if not self.drivers:
            return self.note or f"Forecast of {self.predicted:.0f} units for {describe_calendar_day(self.date)}."
        lines = [
            f"{describe_calendar_day(self.date)}: forecast {self.predicted:.0f} units "
            f"against a typical midweek level of {self.reference_prediction:.0f} units. "
            "The factors that contributed most:"
        ]
        lines.extend(f"  - {d.describe()}" for d in self.top_drivers(n))
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Driver contributions as a frame for charting."""
        return pd.DataFrame(
            [
                {
                    "feature": d.feature,
                    "label": d.label,
                    "value": d.value,
                    "contribution_units": d.contribution_units,
                    "description": d.describe(),
                }
                for d in sorted(self.drivers, key=lambda d: abs(d.contribution_units), reverse=True)
            ]
        )


def build_reference_row(
    row: pd.Series,
    history: pd.DataFrame,
    config: AppConfig,
) -> dict[str, float]:
    """Build the neutral comparison day for one SKU.

    Calendar flags are set to a typical midweek day, promotions off, and temperature to
    the SKU's historical average. Lag and rolling features are set to the SKU's long-run
    average demand, so that "recent demand is running high" shows up as a contribution
    rather than being baked into the reference.
    """
    reference = dict(NEUTRAL_CALENDAR)
    reference["temp_c"] = float(history["temp_c"].mean()) if "temp_c" in history else float(row["temp_c"])
    long_run_mean = float(history["units_sold"].mean()) if not history.empty else float(row.get("roll28", 0.0))
    for column in (*[f"lag{k}" for k in config.features.lags], *[f"roll{w}" for w in config.features.rolling_windows]):
        reference[column] = long_run_mean
    return reference


def explain_gradient_boosting(
    model: GradientBoostingForecast,
    future_row: pd.Series,
    history: pd.DataFrame,
    config: AppConfig,
    features: tuple[str, ...] = EXPLAINABLE_FEATURES,
) -> ForecastExplanation:
    """Attribute one day's gradient-boosting forecast to its inputs.

    Args:
        model: A fitted global gradient-boosting model.
        future_row: One row of the future feature frame (a single SKU-day).
        history: That SKU's observed history, used to build the neutral reference.
        config: Application configuration.
        features: Features to attribute; defaults to the manager-relevant set.

    Returns:
        A :class:`ForecastExplanation` with one contribution per requested feature.
    """
    if not model.is_fitted:
        raise RuntimeError("Cannot explain a forecast from an unfitted model.")
    columns = feature_columns(config)
    base = future_row[columns].astype("float64")
    reference = build_reference_row(future_row, history, config)

    # A single batch: the actual row, the fully neutral row, then one row per feature
    # with that feature alone reset to its reference value.
    rows = [base.to_dict()]
    neutral = base.to_dict()
    for feature, value in reference.items():
        if feature in neutral:
            neutral[feature] = value
    rows.append(neutral)
    attributable = [f for f in features if f in columns and f in reference]
    for feature in attributable:
        variant = base.to_dict()
        variant[feature] = reference[feature]
        rows.append(variant)

    matrix = pd.DataFrame(rows, columns=columns).astype("float64")
    predictions = np.clip(model.model_.predict(matrix), 0.0, None)
    actual_prediction = float(predictions[0])
    reference_prediction = float(predictions[1])

    drivers = [
        DriverContribution(
            feature=feature,
            label=FEATURE_LABELS.get(feature, feature),
            value=float(base[feature]),
            reference_value=float(reference[feature]),
            contribution_units=actual_prediction - float(predictions[2 + i]),
        )
        for i, feature in enumerate(attributable)
    ]
    # Day-of-week and the weekend flag encode the same calendar fact; keep the stronger
    # of the two so the narrative does not double-count it.
    drivers = _deduplicate_calendar_drivers(drivers)

    return ForecastExplanation(
        sku=str(future_row["sku"]),
        date=pd.Timestamp(future_row["date"]),
        model_name=model.name,
        predicted=actual_prediction,
        reference_prediction=reference_prediction,
        drivers=drivers,
        note=(
            "Contributions are estimated by re-running the model with one input replaced "
            "by a typical midweek value. They show association learned from historical "
            "data, not proof of cause."
        ),
    )


def _deduplicate_calendar_drivers(drivers: list[DriverContribution]) -> list[DriverContribution]:
    """Drop the weaker of the overlapping day-of-week and weekend contributions."""
    by_feature = {d.feature: d for d in drivers}
    if "dow" in by_feature and "weekend" in by_feature:
        weaker = min(("dow", "weekend"), key=lambda f: abs(by_feature[f].contribution_units))
        drivers = [d for d in drivers if d.feature != weaker]
    return drivers


def explain_croston(result: ForecastResult) -> ForecastExplanation:
    """Explain an intermittent-demand forecast through its Croston parameters."""
    meta = result.metadata
    size = meta.get("smoothed_demand_size", float("nan"))
    interval = meta.get("smoothed_interval_days", float("nan"))
    rate = meta.get("demand_rate_per_day", float(result.predicted[0]) if result.horizon else 0.0)
    variant = "bias-corrected (SBA)" if meta.get("variant") == "sba" else "classic"
    note = (
        f"This product sells intermittently, so demand is modelled as a rate rather than a "
        f"daily pattern. Recent orders averaged {size:.1f} units roughly every "
        f"{interval:.1f} days, giving about {rate:.2f} units per day using the {variant} "
        "Croston method. Calendar and weather effects are not used for this product "
        "because there is too little non-zero demand to estimate them reliably."
    )
    return ForecastExplanation(
        sku=result.sku,
        date=pd.Timestamp(result.dates[0]) if result.horizon else pd.Timestamp("today"),
        model_name=result.model_name,
        predicted=float(result.predicted[0]) if result.horizon else 0.0,
        reference_prediction=float(rate),
        drivers=[],
        note=note,
    )


def explain_simple_model(result: ForecastResult) -> ForecastExplanation:
    """Explain a naive, seasonal naive or ETS forecast in plain terms."""
    descriptions = {
        "Naive": "The forecast repeats the most recent day's demand.",
        "Seasonal Naive": "The forecast repeats the same weekday from the previous week.",
        "ETS (Holt-Winters)": (
            "The forecast comes from exponential smoothing, which extends the recent level, "
            "trend and weekly pattern of demand. It does not use holidays, weather or promotions."
        ),
    }
    return ForecastExplanation(
        sku=result.sku,
        date=pd.Timestamp(result.dates[0]) if result.horizon else pd.Timestamp("today"),
        model_name=result.model_name,
        predicted=float(result.predicted[0]) if result.horizon else 0.0,
        reference_prediction=float(np.mean(result.predicted)) if result.horizon else 0.0,
        drivers=[],
        note=descriptions.get(result.model_name, f"Forecast produced by {result.model_name}."),
    )


def explain_horizon(
    result: ForecastResult,
    future: pd.DataFrame,
    history: pd.DataFrame,
    config: AppConfig,
    model: GradientBoostingForecast | None = None,
) -> list[ForecastExplanation]:
    """Explain every day of a forecast horizon.

    Falls back to a model-appropriate narrative when the model has no features to
    attribute (Croston, naive, seasonal naive, ETS).
    """
    if model is None or not isinstance(model, GradientBoostingForecast) or not model.is_fitted:
        if result.model_name.startswith("Croston"):
            return [explain_croston(result)]
        return [explain_simple_model(result)]

    rows = future[future["sku"] == result.sku].sort_values("date")
    return [
        explain_gradient_boosting(model, row, history, config)
        for _, row in rows.iterrows()
    ]


def aggregate_drivers(explanations: list[ForecastExplanation]) -> pd.DataFrame:
    """Total each driver's contribution across a forecast horizon.

    Returns:
        A frame of features ordered by absolute total contribution, suitable for the
        dashboard's driver chart. Empty when the model exposes no feature contributions.
    """
    rows = [
        {"feature": d.feature, "label": d.label, "contribution_units": d.contribution_units}
        for explanation in explanations
        for d in explanation.drivers
    ]
    if not rows:
        return pd.DataFrame(columns=["feature", "label", "contribution_units"])
    frame = pd.DataFrame(rows).groupby(["feature", "label"], as_index=False)["contribution_units"].sum()
    return frame.reindex(frame["contribution_units"].abs().sort_values(ascending=False).index).reset_index(drop=True)
