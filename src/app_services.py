"""Application services shared by the dashboard and the command-line scripts.

This module holds the orchestration that the Streamlit app needs but that must not live
inside UI components: assembling the forecasting service, deriving the current stock
position, turning forecasts into recommendations, and running the baseline comparison.
Keeping it here means the same logic is exercised by the test suite and can be reused
without importing Streamlit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import AppConfig, SkuEconomics, load_config
from src.data.loader import build_sku_master, load_sales
from src.data.preprocessing import actuals_window
from src.evaluation.forecast_metrics import bias, wape
from src.evaluation.inventory_metrics import compare_simulations, compute_kpis
from src.features.calendar import days_until_next_order, next_order_date
from src.forecasting.model_selection import ModelChoice, ModelSelector
from src.forecasting.service import ForecastService
from src.inventory.ordering import OrderRecommendation, recommend_order
from src.inventory.policies import PolicyInputs, build_policy
from src.inventory.simulator import InventorySimulator, SimulationResult
from src.inventory.strategies import ForecastDrivenStrategy, ParLevelStrategy
from src.scenarios import BASE_CASE, Scenario
from src.segmentation import SkuSegment, segment_skus, service_level_for_segment, to_segments

logger = logging.getLogger(__name__)


@dataclass
class DashboardContext:
    """Everything the dashboard needs, assembled once per session."""

    config: AppConfig
    sales: pd.DataFrame
    sku_master: pd.DataFrame
    segmentation: pd.DataFrame
    segments: dict[str, SkuSegment]
    model_choices: dict[str, ModelChoice]
    service: ForecastService
    economics: dict[str, SkuEconomics] = field(default_factory=dict)

    @property
    def skus(self) -> list[str]:
        return sorted(str(s) for s in self.sku_master["sku"])

    @property
    def data_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.sales["date"].min())

    @property
    def data_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.sales["date"].max())

    def history(self, sku: str, days: int | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Recent observed demand for one SKU, for the history chart."""
        frame = self.sales[self.sales["sku"] == sku].sort_values("date")
        if as_of is not None:
            frame = frame[frame["date"] < pd.Timestamp(as_of)]
        if days is not None:
            frame = frame.tail(days)
        return frame.reset_index(drop=True)


def build_context(config_path: str | None = None, as_of: pd.Timestamp | None = None) -> DashboardContext:
    """Load data and assemble the services the dashboard depends on.

    Args:
        config_path: Optional path to an alternative configuration file.
        as_of: Decision date used for segmentation and model selection. Defaults to the
            configured simulation start, so the dashboard reflects a selection that was
            made without seeing the holdout period.

    Returns:
        A fully populated :class:`DashboardContext`.
    """
    config = load_config(config_path)
    sales = load_sales(config=config)
    master = build_sku_master(sales, config)
    selection_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(config.simulation.start_date)

    segmentation = segment_skus(sales, master, config, as_of=selection_date)
    segments = to_segments(segmentation)
    choices = ModelSelector(config, strategy="rule").select(sales, master, as_of=selection_date)
    service = ForecastService(sales, config, model_choices=choices)
    economics = {
        str(row.sku): config.sku_economics(str(row.sku), int(row.shelf_life_days))
        for row in master.itertuples(index=False)
    }
    logger.info("Dashboard context built for %d SKUs as of %s", len(economics), selection_date.date())
    return DashboardContext(
        config=config,
        sales=sales,
        sku_master=master,
        segmentation=segmentation,
        segments=segments,
        model_choices=choices,
        service=service,
        economics=economics,
    )


def default_as_of(context: DashboardContext) -> pd.Timestamp:
    """A sensible default decision date: the last day the data can support.

    The forecast horizon needs actual demand to score against, so the default sits one
    horizon before the end of the data.
    """
    return context.data_end - pd.Timedelta(days=context.config.forecast.horizon_days - 1)


def stock_positions(
    context: DashboardContext,
    as_of: pd.Timestamp,
    simulation_daily: pd.DataFrame | None = None,
    warmup_days: int = 28,
) -> pd.DataFrame:
    """Determine on-hand and on-order stock for every SKU at ``as_of``.

    The dataset records sales, not stock, so the position must come from somewhere
    explicit. Three sources are used, in order of preference:

    1. a simulation already run for this window - the closing position of the previous
       day, internally consistent with the orders that were placed;
    2. a short warm-up replay ending the day before ``as_of``, which produces a position
       consistent with the same ordering policy and supplier calendar;
    3. a flat estimate of ``initial_stock_days_of_supply`` days of recent average demand,
       used only when there is too little history to replay.

    The distinction matters for interpretation: a flat estimate is not a position any
    real operation would be in, and it makes every SKU look at risk, so the source is
    reported alongside the numbers.

    Args:
        context: The dashboard context.
        as_of: The decision date.
        simulation_daily: Optional per-SKU-day frame from a simulation run.
        warmup_days: Length of the warm-up replay.

    Returns:
        A frame with ``sku``, ``on_hand``, ``on_order`` and ``source``.
    """
    as_of = pd.Timestamp(as_of)
    previous_day = as_of - pd.Timedelta(days=1)

    if simulation_daily is not None and not simulation_daily.empty:
        snapshot = simulation_daily[simulation_daily["date"] == previous_day]
        if not snapshot.empty:
            frame = snapshot[["sku", "closing_stock", "on_order"]].rename(
                columns={"closing_stock": "on_hand"}
            )
            frame["source"] = "closing position from the simulation run"
            return frame.reset_index(drop=True)

    warmup_start = as_of - pd.Timedelta(days=warmup_days)
    if warmup_start > context.data_start + pd.Timedelta(days=context.config.forecast.min_history_days):
        try:
            replay = run_strategy_simulation(context, "ai", warmup_start, previous_day)
            snapshot = replay.daily[replay.daily["date"] == previous_day]
            if not snapshot.empty:
                frame = snapshot[["sku", "closing_stock", "on_order"]].rename(
                    columns={"closing_stock": "on_hand"}
                )
                frame["source"] = f"{warmup_days}-day warm-up replay to {previous_day:%d %b %Y}"
                return frame.reset_index(drop=True)
        except Exception as exc:
            logger.warning("Warm-up replay failed for %s (%s); falling back to an estimate.", as_of.date(), exc)

    lookback = as_of - pd.Timedelta(days=context.config.baseline.lookback_days)
    window = context.sales[(context.sales["date"] < as_of) & (context.sales["date"] >= lookback)]
    means = window.groupby("sku", observed=True)["units_sold"].mean()
    days = context.config.simulation.initial_stock_days_of_supply
    frame = pd.DataFrame(
        {
            "sku": context.skus,
            "on_hand": [float(means.get(sku, 0.0)) * days for sku in context.skus],
            "on_order": 0.0,
            "source": f"estimate of {days:g} days of recent average demand",
        }
    )
    logger.debug("Stock positions for %s estimated from recent demand", as_of.date())
    return frame


def build_recommendations(
    context: DashboardContext,
    as_of: pd.Timestamp,
    stock: pd.DataFrame,
    scenario: Scenario = BASE_CASE,
    skus: list[str] | None = None,
) -> list[OrderRecommendation]:
    """Produce order recommendations for every SKU at a decision date.

    Args:
        context: The dashboard context.
        as_of: The decision date.
        stock: Frame with ``sku``, ``on_hand`` and ``on_order``.
        scenario: What-if overrides. Scenario stock and service-level overrides take
            precedence over the supplied values.
        skus: Restrict to these SKUs.

    Returns:
        One recommendation per SKU, sorted by descending order value.
    """
    as_of = pd.Timestamp(as_of)
    targets = skus or context.skus
    lead_time = (
        scenario.lead_time_days if scenario.lead_time_days is not None
        else context.config.supplier.lead_time_days
    )
    review_period = days_until_next_order(as_of, context.config.supplier.order_days)
    horizon = max(context.config.forecast.horizon_days, lead_time + review_period)
    forecasts = context.service.forecast(as_of, skus=targets, horizon=horizon, scenario=scenario)
    policy = build_policy(
        context.config.inventory.policy, context.config.inventory.s_S_reorder_multiplier
    )
    stock_by_sku = stock.set_index("sku")

    recommendations: list[OrderRecommendation] = []
    for sku in targets:
        forecast = forecasts[sku]
        on_hand = (
            scenario.current_stock if scenario.current_stock is not None
            else float(stock_by_sku.at[sku, "on_hand"]) if sku in stock_by_sku.index else 0.0
        )
        on_order = float(stock_by_sku.at[sku, "on_order"]) if sku in stock_by_sku.index else 0.0

        service_level = context.config.inventory.service_level
        if scenario.service_level is not None:
            service_level = scenario.service_level
        elif sku in context.segments:
            service_level = service_level_for_segment(context.segments[sku], service_level)

        inputs = PolicyInputs(
            sku=sku,
            daily_forecast=forecast.predicted,
            sigma_daily=forecast.sigma,
            on_hand=max(on_hand, 0.0),
            on_order=max(on_order, 0.0),
            lead_time_days=lead_time,
            review_period_days=review_period,
            service_level=service_level,
        )
        recommendations.append(
            recommend_order(
                inputs,
                context.economics[sku],
                context.config,
                as_of=as_of,
                policy=policy,
                model_name=forecast.model_name,
                metadata={
                    "segment": context.segments[sku].segment if sku in context.segments else None,
                    "scenario": scenario.label,
                },
            )
        )
    return sorted(recommendations, key=lambda r: r.order_value, reverse=True)


def overview_metrics(recommendations: list[OrderRecommendation], as_of: pd.Timestamp, config: AppConfig) -> dict:
    """Headline figures for the Overview page.

    Projected costs are expectations over the forecast horizon, not observed outcomes:
    projected stockout cost weights the stockout unit cost by the estimated probability
    of running short, and projected waste cost values the units expected to expire.
    """
    if not recommendations:
        return {}
    orders_needed = [r for r in recommendations if r.order_required]
    return {
        "skus_requiring_orders": len(orders_needed),
        "total_skus": len(recommendations),
        "total_order_value": float(sum(r.order_value for r in recommendations)),
        "total_order_units": float(sum(r.recommended_order_quantity for r in recommendations)),
        "high_stockout_risk": sum(1 for r in recommendations if r.stockout_risk == "high"),
        "medium_stockout_risk": sum(1 for r in recommendations if r.stockout_risk == "medium"),
        "high_waste_risk": sum(1 for r in recommendations if r.waste_risk == "high"),
        "projected_waste_units": float(sum(r.projected_waste_units for r in recommendations)),
        "projected_waste_cost": float(sum(r.projected_waste_cost for r in recommendations)),
        "projected_stockout_units": float(sum(r.expected_shortfall_units for r in recommendations)),
        "projected_stockout_cost": float(sum(r.projected_stockout_cost for r in recommendations)),
        "next_order_date": next_order_date(as_of, config.supplier.order_days, include_today=False),
        "review_period_days": days_until_next_order(as_of, config.supplier.order_days),
    }


def recent_forecast_accuracy(
    context: DashboardContext,
    as_of: pd.Timestamp,
    lookback_days: int = 28,
) -> pd.DataFrame:
    """Replay the deployed forecast over the recent past, totalled across all products.

    Every day is predicted from an origin that precedes it, exactly as the system would
    have done at the time: the models are refitted at each origin against
    :func:`~src.data.preprocessing.history_as_of`, so no day contributes to its own
    prediction. What comes back is a record of how the forecast has actually performed,
    not a fit scored on the data it was fitted to.

    Origins are spaced one horizon apart, so the windows tile the period without
    overlapping and every day is predicted between one and ``horizon`` days ahead - the
    same lead times an order is placed at.

    Args:
        context: The dashboard context.
        as_of: The decision date. The window ends the day before it, because demand on
            ``as_of`` itself has not been observed yet.
        lookback_days: Length of the period to replay, rounded up to a whole number of
            horizons and clamped to the history available.

    Returns:
        One row per date with ``actual_units`` and ``predicted_units`` summed across
        every SKU, and the signed, absolute and percentage error. Empty when there is
        too little history to replay even one horizon.
    """
    as_of = pd.Timestamp(as_of)
    horizon = context.config.forecast.horizon_days
    n_origins = max(1, (lookback_days + horizon - 1) // horizon)

    earliest = context.data_start + pd.Timedelta(days=context.config.forecast.min_history_days)
    origins = [as_of - pd.Timedelta(days=horizon * k) for k in range(n_origins, 0, -1)]
    origins = [origin for origin in origins if origin >= earliest]

    columns = [
        "date", "actual_units", "predicted_units", "error_units",
        "abs_error_units", "error_pct", "sku_abs_error_units",
    ]
    if not origins:
        logger.warning(
            "Too little history before %s to replay a %d-day forecast window.",
            as_of.date(), lookback_days,
        )
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    for origin in origins:  # ascending, so the service's refit cadence is respected
        results = context.service.forecast(origin, horizon=horizon)
        predicted = pd.DataFrame(
            [
                {"date": date, "sku": sku, "predicted_units": float(value)}
                for sku, result in results.items()
                for date, value in zip(result.dates, result.predicted)
            ]
        )
        actual = actuals_window(context.sales, origin, horizon)[["date", "sku", "units_sold"]]
        # An inner join drops horizon days that run past the end of the data, which is
        # what keeps a partial final window from showing as a collapse in demand.
        frames.append(predicted.merge(actual, on=["date", "sku"], how="inner"))

    combined = pd.concat(frames, ignore_index=True)
    # Kept before the SKUs are summed: a shortfall of tomatoes is not cancelled by a
    # surplus of buns, so the total of the per-SKU absolute errors is the figure that
    # says how much stock was actually misplaced.
    combined["sku_abs_error"] = (combined["predicted_units"] - combined["units_sold"]).abs()
    daily = (
        combined.groupby("date", as_index=False)
        .agg(
            actual_units=("units_sold", "sum"),
            predicted_units=("predicted_units", "sum"),
            sku_abs_error_units=("sku_abs_error", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["error_units"] = daily["predicted_units"] - daily["actual_units"]
    daily["abs_error_units"] = daily["error_units"].abs()
    daily["error_pct"] = np.where(
        daily["actual_units"] > 0,
        daily["error_units"] / daily["actual_units"].replace(0, np.nan) * 100,
        np.nan,
    )
    logger.info(
        "Replayed %d forecast origins over %d days to %s",
        len(origins), len(daily), as_of.date(),
    )
    return daily[columns]


def forecast_accuracy_summary(daily: pd.DataFrame) -> dict:
    """Headline accuracy figures for a :func:`recent_forecast_accuracy` replay.

    Two error scales are reported, and they answer different questions. The ``net_``
    figures compare the total forecast against total demand, so a SKU forecast too high
    offsets one forecast too low - that is the right reading for total spend. The
    ``sku_`` figures add the per-SKU errors up in absolute value before dividing, so
    nothing cancels; that is the right reading for stock on a shelf, and it is the one
    comparable to the backtest's reported accuracy.
    """
    if daily.empty:
        return {}
    actual = daily["actual_units"].to_numpy(dtype=float)
    predicted = daily["predicted_units"].to_numpy(dtype=float)
    worst = daily.loc[daily["abs_error_units"].idxmax()]
    return {
        "days": len(daily),
        "start": pd.Timestamp(daily["date"].min()),
        "end": pd.Timestamp(daily["date"].max()),
        "net_wape": wape(actual, predicted),
        "net_mae_units": float(daily["abs_error_units"].mean()),
        "bias_units": bias(actual, predicted),
        "sku_wape": float(daily["sku_abs_error_units"].sum() / actual.sum() * 100)
        if actual.sum() > 0
        else float("nan"),
        "sku_mae_units": float(daily["sku_abs_error_units"].mean()),
        "total_actual": float(actual.sum()),
        "total_predicted": float(predicted.sum()),
        "worst_date": pd.Timestamp(worst["date"]),
        "worst_error_units": float(worst["error_units"]),
    }


def run_strategy_simulation(
    context: DashboardContext,
    strategy_name: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> SimulationResult:
    """Run one strategy over the holdout window.

    Args:
        context: The dashboard context.
        strategy_name: ``"ai"`` or ``"baseline"``.
        start: Simulation start; defaults to the configured window.
        end: Simulation end; defaults to the configured window.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    if strategy_name == "ai":
        strategy = ForecastDrivenStrategy(context.service, context.config, segments=context.segments)
    elif strategy_name == "baseline":
        strategy = ParLevelStrategy(context.sales, context.config)
    else:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Use 'ai' or 'baseline'.")

    simulator = InventorySimulator(
        context.sales, context.config, strategy, sku_master=context.sku_master
    )
    return simulator.run(start, end)


def compare_strategies(
    context: DashboardContext,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict:
    """Run both strategies and build the comparison table.

    Returns:
        A dictionary with the two :class:`~src.inventory.simulator.SimulationResult`
        objects, their KPI dictionaries and the comparison frame.
    """
    baseline = run_strategy_simulation(context, "baseline", start, end)
    ai = run_strategy_simulation(context, "ai", start, end)
    baseline_kpis = compute_kpis(baseline.daily, baseline.orders)
    ai_kpis = compute_kpis(ai.daily, ai.orders)
    return {
        "baseline": baseline,
        "ai": ai,
        "baseline_kpis": baseline_kpis,
        "ai_kpis": ai_kpis,
        "comparison": compare_simulations(
            baseline_kpis, ai_kpis, baseline.strategy_name, ai.strategy_name
        ),
    }


def order_sheet(
    order_view: pd.DataFrame,
    recommendations: list[OrderRecommendation],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """The day's order as a sheet a manager can print, export or send to a supplier.

    Only products with something to order are listed, largest first, with the pack size and
    minimum order quantity beside each line so the quantity can be checked against what the
    supplier will actually deliver. The recommended quantity stays next to the final one, so
    an override remains visible on the sheet rather than being silently absorbed.

    Args:
        order_view: The frame from :func:`apply_overrides`.
        recommendations: The recommendations behind that frame, for the supplier terms.
        as_of: The decision date, written into the ``Order date`` column.

    Returns:
        One row per product to order, or an empty frame with the same columns when nothing
        is due.
    """
    columns = ["Order date", "Product", "Order (units)", "Recommended (units)", "Overridden",
               "Pack size", "Minimum order", "Unit cost (GBP)", "Line value (GBP)"]
    terms = {r.sku: r for r in recommendations}
    rows = []
    for row in order_view.to_dict("records"):
        if float(row["final_order"]) <= 0:
            continue
        rec = terms[row["sku"]]
        rows.append(
            {
                "Order date": pd.Timestamp(as_of).date().isoformat(),
                "Product": row["sku"],
                "Order (units)": round(float(row["final_order"])),
                "Recommended (units)": round(float(row["recommended_order"])),
                "Overridden": "yes" if row["overridden"] else "no",
                "Pack size": rec.pack_size,
                "Minimum order": rec.min_order_quantity,
                "Unit cost (GBP)": round(rec.unit_cost, 2),
                "Line value (GBP)": round(float(row["final_value"]), 2),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("Order (units)", ascending=False).reset_index(drop=True)[columns]


def apply_overrides(
    recommendations: list[OrderRecommendation],
    overrides: dict[str, float],
) -> pd.DataFrame:
    """Combine recommendations with manager overrides for display and export.

    The override is recorded alongside the recommendation; neither the recommendation nor
    the trained model is modified.
    """
    rows = []
    for rec in recommendations:
        override = overrides.get(rec.sku)
        final = float(override) if override is not None else rec.recommended_order_quantity
        rows.append(
            {
                "sku": rec.sku,
                "recommended_order": rec.recommended_order_quantity,
                "override_order": float(override) if override is not None else np.nan,
                "final_order": final,
                "difference": final - rec.recommended_order_quantity,
                "final_value": final * rec.unit_cost,
                "overridden": override is not None,
            }
        )
    return pd.DataFrame(rows)
