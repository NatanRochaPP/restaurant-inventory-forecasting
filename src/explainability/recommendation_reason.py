"""Explaining why a particular order quantity was recommended.

The forecast explanation answers "why this much demand"; this module answers "why this
much stock". A manager who cannot see the reasoning will either follow the number blindly
or ignore it, and neither is decision support.

Each explanation is assembled from values the recommendation already carries, so the text
can never drift from the arithmetic that produced the number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.explainability.forecast_drivers import ForecastExplanation
from src.features.calendar import BANK_HOLIDAYS, WEEKEND_DAYS, is_school_holiday
from src.inventory.ordering import OrderRecommendation
from src.segmentation import SkuSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationExplanation:
    """Structured, manager-facing justification for one recommendation."""

    sku: str
    as_of: pd.Timestamp
    headline: str
    demand_reason: list[str] = field(default_factory=list)
    cover_reason: list[str] = field(default_factory=list)
    constraint_reason: list[str] = field(default_factory=list)
    risk_reason: list[str] = field(default_factory=list)
    caveat: str = ""

    def bullets(self) -> list[str]:
        """Every reason line in display order."""
        return [*self.demand_reason, *self.cover_reason, *self.constraint_reason, *self.risk_reason]

    def full_text(self) -> str:
        """The complete explanation as a readable block."""
        parts = [self.headline, ""]
        parts.extend(f"- {line}" for line in self.bullets())
        if self.caveat:
            parts.extend(["", self.caveat])
        return "\n".join(parts)

    def short_reason(self) -> str:
        """One-line summary for the recommendations table."""
        return self.demand_reason[0] if self.demand_reason else self.headline


def _calendar_highlights(dates: pd.DatetimeIndex) -> list[str]:
    """Notable calendar events inside the forecast horizon."""
    dates = pd.DatetimeIndex(dates)
    highlights: list[str] = []
    if any(d.dayofweek in WEEKEND_DAYS for d in dates):
        highlights.append("a weekend")
    holidays = [d for d in dates if d in BANK_HOLIDAYS]
    if holidays:
        highlights.append(f"a bank holiday ({holidays[0]:%d %b})")
    if any(is_school_holiday(d) for d in dates):
        highlights.append("school holidays")
    return highlights


def explain_recommendation(
    recommendation: OrderRecommendation,
    *,
    recent_average_daily: float | None = None,
    segment: SkuSegment | None = None,
    forecast_explanation: ForecastExplanation | None = None,
    forecast_dates: pd.DatetimeIndex | None = None,
) -> RecommendationExplanation:
    """Build the explanation for one order recommendation.

    Args:
        recommendation: The recommendation to explain.
        recent_average_daily: The SKU's trailing average daily demand, used to say
            whether the forecast is above or below normal.
        segment: ABC/XYZ segment, mentioned when it changed the service level.
        forecast_explanation: Optional driver breakdown to quote the top factors from.
        forecast_dates: The horizon dates, used to spot weekends and holidays.

    Returns:
        A :class:`RecommendationExplanation` whose statements are all derived from the
        recommendation's own figures.
    """
    rec = recommendation
    mean_daily = float(np.mean(rec.daily_forecast)) if len(rec.daily_forecast) else 0.0

    # -- headline ----------------------------------------------------------------------
    if rec.recommended_order_quantity > 0:
        headline = (
            f"Order {rec.recommended_order_quantity:.0f} units of {rec.sku} "
            f"(about £{rec.order_value:,.2f})."
        )
    elif rec.reorder_point is not None and rec.inventory_position > rec.reorder_point:
        headline = (
            f"No order needed for {rec.sku}: stock of {rec.inventory_position:.0f} units is above "
            f"the reorder point of {rec.reorder_point:.0f}."
        )
    else:
        headline = (
            f"No order needed for {rec.sku}: stock of {rec.inventory_position:.0f} units already "
            f"covers the target of {rec.target_stock:.0f}."
        )

    # -- why this much demand ----------------------------------------------------------
    demand_reason: list[str] = []
    horizon_days = len(rec.daily_forecast)
    if recent_average_daily and recent_average_daily > 0:
        change = (mean_daily - recent_average_daily) / recent_average_daily * 100
        if abs(change) >= 5:
            wording = "higher" if change > 0 else "lower"
            demand_reason.append(
                f"Expected demand of {mean_daily:.0f} units/day over the next {horizon_days} days is "
                f"{abs(change):.0f}% {wording} than the recent average of {recent_average_daily:.0f} units/day."
            )
        else:
            demand_reason.append(
                f"Expected demand of {mean_daily:.0f} units/day is in line with the recent average "
                f"of {recent_average_daily:.0f} units/day."
            )
    else:
        demand_reason.append(f"Expected demand is {mean_daily:.0f} units/day over the next {horizon_days} days.")

    if forecast_dates is not None:
        highlights = _calendar_highlights(forecast_dates)
        if highlights:
            demand_reason.append(
                f"The period ahead includes {', '.join(highlights)}, which the forecast has taken into account."
            )

    if forecast_explanation is not None and forecast_explanation.drivers:
        top = forecast_explanation.top_drivers(2)
        demand_reason.extend(driver.describe() for driver in top)
    elif forecast_explanation is not None and forecast_explanation.note:
        demand_reason.append(forecast_explanation.note)

    # -- why this much cover -----------------------------------------------------------
    cover_reason: list[str] = []
    cover_text = (
        f"{rec.days_of_cover:.1f} days" if np.isfinite(rec.days_of_cover) else "an indefinite period"
    )
    cover_reason.append(
        f"Current stock of {rec.on_hand:.0f} units covers approximately {cover_text} of forecast demand."
    )
    if rec.on_order > 0:
        cover_reason.append(
            f"A further {rec.on_order:.0f} units are already on order, giving an inventory position "
            f"of {rec.inventory_position:.0f} units."
        )
    review_days = f"{rec.review_period_days} day" + ("" if rec.review_period_days == 1 else "s")
    cover_reason.append(
        f"The target stock of {rec.target_stock:.0f} units covers the {rec.protection_period_days}-day "
        f"protection period ({rec.lead_time_days}-day supplier lead time plus "
        f"{review_days} until the next delivery), which needs about "
        f"{rec.expected_demand_protection:.0f} units, plus {rec.safety_stock:.0f} units of safety stock "
        f"for a {rec.service_level:.0%} service level."
    )
    if segment is not None:
        cover_reason.append(
            f"This is a {segment.describe()}, so it is held to a {rec.service_level:.0%} availability target."
        )

    # -- constraints applied -----------------------------------------------------------
    constraint_reason = [adjustment.note for adjustment in rec.adjustments]
    if rec.adjustments and rec.raw_order_quantity > 0:
        constraint_reason.insert(
            0,
            f"The unconstrained calculation was {rec.raw_order_quantity:.0f} units "
            f"(target {rec.target_stock:.0f} less inventory position {rec.inventory_position:.0f}); "
            f"the following adjustments were applied:",
        )

    # -- risk --------------------------------------------------------------------------
    risk_reason: list[str] = []
    if rec.stockout_risk != "low":
        risk_reason.append(
            f"Stockout risk is {rec.stockout_risk}: there is roughly a {rec.stockout_probability:.0%} chance "
            f"demand over the protection period exceeds available stock."
        )
    if rec.waste_risk != "low":
        risk_reason.append(
            f"Waste risk is {rec.waste_risk}: about {rec.projected_waste_units:.0f} units may not be used "
            f"within the {rec.shelf_life_days}-day shelf life."
        )
    if not risk_reason:
        risk_reason.append(
            f"Stockout and waste risks are both low at this quantity "
            f"(stockout probability {rec.stockout_probability:.0%})."
        )

    caveat = (
        "This is a recommendation only - no order is placed automatically. Forecast drivers show "
        "factors associated with demand in the historical data; they are not proof of cause."
    )

    return RecommendationExplanation(
        sku=rec.sku,
        as_of=pd.Timestamp(rec.as_of),
        headline=headline,
        demand_reason=demand_reason,
        cover_reason=cover_reason,
        constraint_reason=constraint_reason,
        risk_reason=risk_reason,
        caveat=caveat,
    )


def explanation_table(
    recommendations: list[OrderRecommendation],
    recent_averages: dict[str, float] | None = None,
    segments: dict[str, SkuSegment] | None = None,
) -> pd.DataFrame:
    """Build the recommendations table with a one-line reason per SKU.

    Returns:
        The columns the Recommendations page displays: SKU, current stock, forecast
        demand, recommended order, risk and reason.
    """
    rows = []
    for rec in recommendations:
        explanation = explain_recommendation(
            rec,
            recent_average_daily=(recent_averages or {}).get(rec.sku),
            segment=(segments or {}).get(rec.sku),
        )
        rows.append(
            {
                "SKU": rec.sku,
                "Current Stock": round(rec.on_hand, 1),
                "On Order": round(rec.on_order, 1),
                "Forecast Demand": round(rec.expected_demand_horizon, 1),
                "Recommended Order": round(rec.recommended_order_quantity, 1),
                "Order Value": round(rec.order_value, 2),
                "Risk": rec.risk_level,
                "Model": rec.model_name,
                "Reason": explanation.short_reason(),
            }
        )
    return pd.DataFrame(rows).sort_values("Recommended Order", ascending=False).reset_index(drop=True)
