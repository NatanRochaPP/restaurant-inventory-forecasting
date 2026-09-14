"""Manager-facing Streamlit dashboard.

Run with::

    streamlit run app.py

The app is deliberately thin: it renders state and collects input, while every
calculation lives in ``src`` (``app_services`` for orchestration, ``inventory`` for
policy, ``explainability`` for the narratives). That separation is what allows the same
numbers to be produced by the command-line scripts and asserted in the test suite.

The system is decision support. It recommends; it never places an order.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.app_services import (
    DashboardContext,
    apply_overrides,
    build_context,
    build_recommendations,
    compare_strategies,
    default_as_of,
    forecast_accuracy_summary,
    overview_metrics,
    recent_forecast_accuracy,
    stock_positions,
)
from src.data.preprocessing import actuals_window
from src.evaluation.frontier import matched_comparison, run_frontier
from src.evaluation.inventory_metrics import daily_totals, kpis_by_sku
from src.explainability.forecast_drivers import aggregate_drivers
from src.explainability.recommendation_reason import explain_recommendation, explanation_table
from src.forecasting.model_selection import ModelSelector
from src.inventory.ordering import recommendations_to_frame
from src.persistence.database import Database
from src.scenarios import BASE_CASE, Scenario
from src.segmentation import service_level_for_segment
from src.viz import (
    demand_forecast_chart,
    driver_chart,
    forecast_accuracy_chart,
    frontier_chart,
    kpi_comparison_chart,
    scenario_comparison_chart,
    simulation_timeseries_chart,
    stock_cover_chart,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="Restaurant Inventory Decision Support",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Overview", "Forecast", "Recommendations", "What-if", "Simulation & Evaluation"]


# -- cached resources ------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading data and fitting models…")
def get_context() -> DashboardContext:
    """Build the dashboard context once per session."""
    return build_context()


@st.cache_resource(show_spinner="Replaying the holdout period for both policies…")
def get_comparison(_context: DashboardContext) -> dict:
    """Run both simulations. Cached because a full replay takes tens of seconds."""
    return compare_strategies(_context)


@st.cache_resource(show_spinner="Sweeping service levels for both policies…")
def get_frontier(_context: DashboardContext) -> pd.DataFrame:
    """Sweep both policies across their tuning parameter."""
    return run_frontier(
        _context.sales,
        _context.config,
        forecast_service=_context.service,
        sku_master=_context.sku_master,
        service_levels=[0.50, 0.70, 0.80, 0.90, 0.95, 0.99],
        buffer_pcts=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    )


@st.cache_data(show_spinner=False)
def get_recommendations(_context: DashboardContext, as_of: pd.Timestamp, _stock: pd.DataFrame):
    """Recommendations for a decision date, cached per date."""
    return build_recommendations(_context, as_of, _stock)


@st.cache_data(show_spinner="Replaying recent forecasts…")
def get_recent_accuracy(_context: DashboardContext, as_of: pd.Timestamp, lookback_days: int):
    """Recent forecast-versus-actual replay, cached per date and window length."""
    return recent_forecast_accuracy(_context, as_of, lookback_days)


def get_state(key: str, default):
    """Read a value from Streamlit session state, initialising it on first use."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# -- shared UI helpers -----------------------------------------------------------------


def risk_badge(level: str) -> str:
    """Risk shown as an icon plus a word, never colour alone."""
    return {"low": "Low", "medium": "Medium", "high": "High"}.get(level, level)


def sidebar(context: DashboardContext) -> tuple[str, pd.Timestamp]:
    """Render the sidebar and return the selected page and decision date."""
    with st.sidebar:
        st.title("Inventory Decision Support")
        st.caption("Forecasting-Driven Inventory Resupply for Restaurants")

        # The page and decision date live in the URL so a view can be linked or bookmarked.
        requested = st.query_params.get("page", PAGES[0])
        if "page" not in st.session_state:
            st.session_state["page"] = requested if requested in PAGES else PAGES[0]
        page = st.radio("Page", PAGES, key="page", label_visibility="collapsed")
        if page != requested:
            st.query_params["page"] = page
        st.divider()

        earliest = (context.data_start + pd.Timedelta(days=context.config.forecast.min_history_days)).date()
        latest = context.data_end.date()
        requested_date = st.query_params.get("as_of")
        try:
            default_date = pd.Timestamp(requested_date).date() if requested_date else default_as_of(context).date()
        except ValueError:
            default_date = default_as_of(context).date()
        chosen = st.date_input(
            "Decision date",
            value=min(max(default_date, earliest), latest),
            min_value=earliest,
            max_value=latest,
            help=(
                "The day the order decision is made. Only demand recorded before this "
                "date is visible to the models."
            ),
        )
        as_of = pd.Timestamp(chosen)
        if str(chosen) != requested_date:
            st.query_params["as_of"] = str(chosen)

        st.divider()
        st.caption(
            f"**Policy** {context.config.inventory.policy}  \n"
            f"**Service level** {context.config.inventory.service_level:.0%}  \n"
            f"**Lead time** {context.config.supplier.lead_time_days} days  \n"
            f"**Horizon** {context.config.forecast.horizon_days} days"
        )
        st.caption(
            "Decision support only — this tool never places an order. "
            "All data is synthetic; no personal or guest data is used."
        )
    return page, as_of


# -- pages -----------------------------------------------------------------------------


def page_overview(context: DashboardContext, as_of: pd.Timestamp) -> None:
    st.header("Overview")
    st.caption(f"Position as at **{as_of:%A %d %B %Y}**")

    stock = stock_positions(context, as_of)
    recommendations = get_recommendations(context, as_of, stock)
    metrics = overview_metrics(recommendations, as_of, context.config)

    row = st.columns(3)
    row[0].metric(
        "SKUs requiring an order",
        f"{metrics['skus_requiring_orders']} of {metrics['total_skus']}",
        help="Products whose inventory position is below the target stock level.",
    )
    row[1].metric(
        "Recommended order value",
        f"£{metrics['total_order_value']:,.2f}",
        help=f"{metrics['total_order_units']:,.0f} units across all products.",
    )
    row[2].metric(
        "Next delivery order",
        f"{metrics['next_order_date']:%a %d %b}",
        help=f"Stock must cover {metrics['review_period_days']} days until then, plus the lead time.",
    )

    row = st.columns(4)
    row[0].metric(
        "High stockout risk",
        metrics["high_stockout_risk"],
        help="Products with more than a 25% estimated chance of running short before the next delivery.",
    )
    row[1].metric("High waste risk", metrics["high_waste_risk"])
    row[2].metric(
        "Projected stockout cost",
        f"£{metrics['projected_stockout_cost']:,.2f}",
        help="Expected shortfall units weighted by their estimated probability and unit stockout cost.",
    )
    row[3].metric(
        "Projected waste cost",
        f"£{metrics['projected_waste_cost']:,.2f}",
        help="Value of stock not expected to be used within its shelf life.",
    )

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        frame = recommendations_to_frame(recommendations)
        st.plotly_chart(
            stock_cover_chart(frame[["sku", "days_of_cover", "risk"]]),
            use_container_width=True,
        )
    with right:
        st.subheader("Attention needed")
        flagged = [r for r in recommendations if r.risk_level != "low"]
        if not flagged:
            st.success("No products are flagged for stockout or waste risk today.")
        for rec in sorted(flagged, key=lambda r: r.stockout_probability, reverse=True)[:6]:
            explanation = explain_recommendation(
                rec,
                recent_average_daily=context.service.recent_average_demand(as_of, rec.sku),
                segment=context.segments.get(rec.sku),
            )
            with st.container(border=True):
                st.markdown(f"**{rec.sku}** · {risk_badge(rec.risk_level)}")
                st.caption(explanation.headline)

    st.divider()
    st.subheader("How accurate has the forecast been?")
    lookback = st.selectbox(
        "Period replayed",
        options=[35, 63, 91],
        format_func=lambda days: f"Last {days // 7} weeks ({days} days)",
        key="overview_accuracy_lookback",
    )
    accuracy = get_recent_accuracy(context, as_of, lookback)
    summary = forecast_accuracy_summary(accuracy)

    if not summary:
        st.info(
            "There is not enough history before this decision date to replay a forecast "
            "window. Choose a later date to see how the forecast has performed."
        )
    else:
        row = st.columns(4)
        row[0].metric(
            "Error per product (WAPE)",
            f"{summary['sku_wape']:.1f}%",
            help=(
                "Every product's daily miss added up in absolute value, as a percentage of "
                "units sold. Nothing cancels out, so this is the figure that reflects stock "
                "on a shelf."
            ),
        )
        row[1].metric(
            "Error on the total (WAPE)",
            f"{summary['net_wape']:.1f}%",
            help=(
                "The same comparison made on the daily total across all products, where a "
                "product forecast too high offsets one forecast too low. Read this for total "
                "spend, not for availability."
            ),
        )
        row[2].metric(
            "Average daily miss",
            f"{summary['sku_mae_units']:,.0f} units",
            help=(
                f"Summed across all products, against {summary['total_actual'] / summary['days']:,.0f} "
                "units sold on an average day in this period."
            ),
        )
        row[3].metric(
            "Bias",
            f"{summary['bias_units']:+,.0f} units/day",
            help="Positive means the forecast ran high on average; negative means it ran low.",
        )

        st.plotly_chart(
            forecast_accuracy_chart(
                accuracy,
                title=(
                    f"Forecast versus actual demand, all products · "
                    f"{summary['start']:%d %b} to {summary['end']:%d %b %Y}"
                ),
            ),
            use_container_width=True,
        )
        st.caption(
            "Each day was predicted from a decision date one to seven days earlier, with the "
            "models refitted at every origin on history strictly before it - so no day "
            "contributed to its own prediction. The lower panel carries the error on its own "
            "units scale; bars above the line are days the forecast ran high, below it days it "
            "ran short. These percentages are weighted by volume and so are not comparable to "
            "the headline backtest accuracy, which averages each product's error equally and "
            "is therefore dominated by the small, erratic lines."
        )
        with st.expander("The numbers behind this chart"):
            st.dataframe(
                accuracy.round(1), use_container_width=True, hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Date", format="ddd DD MMM YYYY"),
                    "actual_units": st.column_config.NumberColumn("Actual units"),
                    "predicted_units": st.column_config.NumberColumn("Forecast units"),
                    "error_units": st.column_config.NumberColumn("Forecast − actual"),
                    "abs_error_units": st.column_config.NumberColumn("Absolute error"),
                    "error_pct": st.column_config.NumberColumn("Error %", format="%.1f%%"),
                    "sku_abs_error_units": st.column_config.NumberColumn("Per-product error"),
                },
            )

    st.divider()
    st.subheader("All products")
    st.caption(f"Stock position source: {stock['source'].iloc[0]}.")
    st.dataframe(
        recommendations_to_frame(recommendations),
        use_container_width=True,
        hide_index=True,
    )


def page_forecast(context: DashboardContext, as_of: pd.Timestamp) -> None:
    st.header("Forecast")

    left, right = st.columns([2, 3])
    sku = left.selectbox("Product", context.skus)
    history_days = right.slider("Days of history to show", 28, 365, 90, step=7)

    choice = context.model_choices[sku]
    segment = context.segments.get(sku)

    result, explanations = context.service.explain_forecast(as_of, sku)
    history = context.history(sku, days=history_days, as_of=as_of)
    actuals = actuals_window(context.sales, as_of, result.horizon, sku)

    row = st.columns(4)
    row[0].metric("Forecast over the horizon", f"{result.total():,.0f} units")
    row[1].metric("Average per day", f"{result.total() / result.horizon:,.1f} units")
    row[2].metric(
        "Recent average per day",
        f"{context.service.recent_average_demand(as_of, sku):,.1f} units",
        help="Trailing 28-day mean, for comparison.",
    )
    row[3].metric("Model", choice.model_name, help=choice.reason)

    st.plotly_chart(
        demand_forecast_chart(
            history, result.to_frame(), actuals if not actuals.empty else None,
            title=f"{sku} — observed demand, forecast and {context.config.forecast.interval_service_level:.0%} interval",
        ),
        use_container_width=True,
    )
    if not actuals.empty:
        st.caption(
            "The actual line is shown for evaluation only. It was not available to the model, "
            "which saw no data on or after the decision date."
        )

    st.subheader("Why the forecast is at this level")
    with st.container(border=True):
        st.markdown(f"**Model chosen:** {choice.model_name}")
        st.caption(choice.reason)
        if segment is not None:
            st.caption(f"Segment {segment.describe()}.")

    drivers = aggregate_drivers(explanations)
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            driver_chart(drivers, title=f"Contributions across the {result.horizon}-day horizon"),
            use_container_width=True,
        )
    with right:
        if drivers.empty:
            st.info(explanations[0].note)
        else:
            day_labels = [f"{pd.Timestamp(e.date):%a %d %b}" for e in explanations]
            selected = st.selectbox("Explain a single day", day_labels)
            explanation = explanations[day_labels.index(selected)]
            st.markdown(explanation.narrative(4).replace("\n", "  \n"))
            st.caption(
                f"Interactions between factors account for a further "
                f"{explanation.unexplained_units:+.0f} units that this breakdown does not attribute."
            )
            st.caption(explanation.note)

    if not drivers.empty:
        with st.expander("Driver contributions as a table"):
            st.dataframe(drivers.round(2), use_container_width=True, hide_index=True)

    with st.expander("Forecast values"):
        st.dataframe(result.to_frame().round(2), use_container_width=True, hide_index=True)


def page_recommendations(context: DashboardContext, as_of: pd.Timestamp) -> None:
    st.header("Order recommendations")
    st.caption(f"For the delivery ordered on **{as_of:%A %d %B %Y}**")

    stock = stock_positions(context, as_of)
    recommendations = get_recommendations(context, as_of, stock)
    by_sku = {r.sku: r for r in recommendations}
    overrides: dict[str, float] = get_state("overrides", {})

    recent = {sku: context.service.recent_average_demand(as_of, sku) for sku in context.skus}
    table = explanation_table(recommendations, recent_averages=recent, segments=context.segments)
    table["Risk"] = table["Risk"].map(risk_badge)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Full order overview")
    order_view = apply_overrides(recommendations, overrides)
    overview = (
        order_view.merge(
            table[["SKU", "Current Stock", "On Order", "Model", "Risk"]],
            left_on="sku",
            right_on="SKU",
            how="left",
        )
        .rename(columns={"override_order": "Order to place"})
        .drop(columns=["SKU"])
    )
    overview["Order to place"] = overview["Order to place"].where(
        overview["Order to place"].notna(), overview["recommended_order"]
    )
    recommended_value = float(sum(rec.recommended_order_quantity * rec.unit_cost for rec in recommendations))
    final_value = float(order_view["final_value"].sum())
    summary = st.columns(4)
    summary[0].metric("Recommended order value", f"£{recommended_value:,.2f}")
    summary[1].metric("Recommended units", f"{order_view['recommended_order'].sum():,.0f}")
    summary[2].metric("Edited order value", f"£{final_value:,.2f}")
    summary[3].metric("SKUs edited", f"{int(order_view['overridden'].sum())}")

    with st.form("order_overview_editor"):
        edited = st.data_editor(
            overview[[
                "sku",
                "Current Stock",
                "On Order",
                "Model",
                "Risk",
                "recommended_order",
                "Order to place",
                "final_order",
                "difference",
                "final_value",
            ]],
            use_container_width=True,
            hide_index=True,
            disabled=["sku", "Current Stock", "On Order", "Model", "Risk", "recommended_order", "final_order", "difference", "final_value"],
            column_config={
                "sku": st.column_config.TextColumn("SKU"),
                "Current Stock": st.column_config.NumberColumn("Current stock", format="%.1f"),
                "On Order": st.column_config.NumberColumn("On order", format="%.1f"),
                "Model": st.column_config.TextColumn("Model"),
                "Risk": st.column_config.TextColumn("Risk"),
                "recommended_order": st.column_config.NumberColumn("Recommended order", format="%.0f"),
                "Order to place": st.column_config.NumberColumn("Order to place", format="%.0f", min_value=0.0),
                "final_order": st.column_config.NumberColumn("Final order", format="%.0f"),
                "difference": st.column_config.NumberColumn("Difference", format="%.0f"),
                "final_value": st.column_config.NumberColumn("Final value (£)", format="%.2f"),
            },
        )
        submitted_overview = st.form_submit_button("Apply changes to the full order", use_container_width=True)

    if submitted_overview:
        updated_overrides: dict[str, float] = {}
        for row in edited.to_dict(orient="records"):
            quantity = float(row["Order to place"])
            recommended = float(row["recommended_order"])
            if abs(quantity - recommended) > 1e-9:
                updated_overrides[str(row["sku"])] = quantity

        st.session_state["overrides"] = updated_overrides
        try:
            database = Database.from_config(context.config)
            database.initialise()
            for row in edited.to_dict(orient="records"):
                quantity = float(row["Order to place"])
                recommended = float(row["recommended_order"])
                if abs(quantity - recommended) <= 1e-9:
                    continue
                database.record_override(
                    run_id=f"dashboard-{as_of:%Y%m%d}",
                    as_of=as_of,
                    sku=str(row["sku"]),
                    recommended_order=recommended,
                    override_order=quantity,
                    reason="Updated in the full order overview",
                )
            st.success("Full order changes applied for this session.")
        except Exception as exc:  # storage must never block the decision
            st.warning(f"Full order changes were kept in this session only — they could not be saved: {exc}")

    if not order_view["overridden"].any():
        st.caption("No manual changes yet. Edit any order quantity above to review the final order before saving.")

    st.divider()
    st.subheader("Review a product")
    sku = st.selectbox("Product", [r.sku for r in recommendations])
    rec = by_sku[sku]
    explanation = explain_recommendation(
        rec,
        recent_average_daily=recent[sku],
        segment=context.segments.get(sku),
        forecast_dates=pd.DatetimeIndex(
            pd.date_range(as_of, periods=len(rec.daily_forecast), freq="D")
        ),
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"#### {explanation.headline}")
        for line in explanation.bullets():
            st.markdown(f"- {line}")
        st.caption(explanation.caveat)

    with right:
        st.markdown("#### The calculation")
        st.dataframe(
            pd.DataFrame(
                {
                    "Step": [
                        "On hand", "On order", "Inventory position",
                        f"Expected demand over {rec.protection_period_days} days",
                        "Safety stock", "Target stock",
                        "Raw order (target − position)", "Recommended order",
                    ],
                    "Units": [
                        round(rec.on_hand, 1), round(rec.on_order, 1), round(rec.inventory_position, 1),
                        round(rec.expected_demand_protection, 1), round(rec.safety_stock, 1),
                        round(rec.target_stock, 1), round(rec.raw_order_quantity, 1),
                        round(rec.recommended_order_quantity, 1),
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Override")
        with st.form(f"override_{sku}"):
            quantity = st.number_input(
                "Order quantity to place",
                min_value=0.0,
                value=float(overrides.get(sku, rec.recommended_order_quantity)),
                step=float(rec.pack_size),
                help=f"Supplier case size is {rec.pack_size} units.",
            )
            reason = st.text_input("Reason for the override", placeholder="e.g. private booking on Saturday")
            submitted = st.form_submit_button("Record override", use_container_width=True)
        if submitted:
            overrides[sku] = float(quantity)
            st.session_state["overrides"] = overrides
            try:
                database = Database.from_config(context.config)
                database.initialise()
                database.record_override(
                    run_id=f"dashboard-{as_of:%Y%m%d}",
                    as_of=as_of,
                    sku=sku,
                    recommended_order=rec.recommended_order_quantity,
                    override_order=float(quantity),
                    reason=reason,
                )
                st.success(
                    f"Override recorded: {quantity:,.0f} units instead of "
                    f"{rec.recommended_order_quantity:,.0f}. The trained model is unchanged."
                )
            except Exception as exc:  # storage must never block the decision
                st.warning(f"Override held in this session only — it could not be saved: {exc}")

    if overrides:
        st.divider()
        st.subheader("Overrides this session")
        st.dataframe(
            apply_overrides(recommendations, overrides).query("overridden"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Overrides are recorded alongside the recommendation. They do not retrain or alter the model.")


def page_what_if(context: DashboardContext, as_of: pd.Timestamp) -> None:
    st.header("What-if analysis")
    st.caption(
        "Explore how the recommendation responds to different conditions. "
        "Nothing here changes the trained model or any saved result."
    )

    sku = st.selectbox("Product", context.skus)
    stock = stock_positions(context, as_of)
    base_stock = round(float(stock.set_index("sku").at[sku, "on_hand"]), 1)
    # The default service level is the SKU's segment-adjusted target, so leaving the
    # slider alone genuinely means "no change" rather than silently overriding it.
    segment = context.segments.get(sku)
    base_service_level = (
        service_level_for_segment(segment, context.config.inventory.service_level)
        if segment is not None
        else context.config.inventory.service_level
    )
    base_lead_time = context.config.supplier.lead_time_days

    controls = st.columns(3)
    uplift = controls[0].slider("Demand uplift (%)", -50, 100, 0, step=5)
    use_temperature = controls[0].checkbox("Override temperature", value=False)
    temperature = controls[0].slider("Temperature (°C)", -5, 35, 15, step=1, disabled=not use_temperature)
    promo = controls[1].selectbox("Promotion", ["As planned", "Force on", "Force off"])
    service_level = controls[1].slider(
        "Service level (%)", 50, 99, int(round(base_service_level * 100)),
        help=f"Default for this product is {base_service_level:.0%} (segment {segment.segment if segment else 'n/a'}).",
    )
    lead_time = controls[2].slider("Supplier lead time (days)", 0, 7, base_lead_time)
    current_stock = controls[2].number_input("Current stock (units)", min_value=0.0, value=base_stock, step=1.0)

    # Only fields the manager actually moved become overrides.
    changed_service = abs(service_level / 100 - base_service_level) > 0.005
    scenario = Scenario(
        label="Scenario",
        demand_uplift_pct=float(uplift),
        temperature_c=float(temperature) if use_temperature else None,
        promo={"As planned": None, "Force on": True, "Force off": False}[promo],
        service_level=service_level / 100 if changed_service else None,
        lead_time_days=int(lead_time) if lead_time != base_lead_time else None,
        current_stock=float(current_stock) if abs(current_stock - base_stock) > 1e-9 else None,
    )
    if scenario.is_base_case:
        st.info("No changes applied yet — adjust a control above to compare a scenario with the base case.")

    base_recommendation = build_recommendations(context, as_of, stock, BASE_CASE, [sku])[0]
    scenario_recommendation = build_recommendations(context, as_of, stock, scenario, [sku])[0]

    st.caption("Changes applied: " + "; ".join(scenario.describe()) + ".")

    row = st.columns(4)
    row[0].metric(
        "Forecast demand over horizon",
        f"{scenario_recommendation.expected_demand_horizon:,.0f}",
        delta=f"{scenario_recommendation.expected_demand_horizon - base_recommendation.expected_demand_horizon:+,.0f}",
    )
    row[1].metric(
        "Target stock",
        f"{scenario_recommendation.target_stock:,.0f}",
        delta=f"{scenario_recommendation.target_stock - base_recommendation.target_stock:+,.0f}",
    )
    row[2].metric(
        "Recommended order",
        f"{scenario_recommendation.recommended_order_quantity:,.0f}",
        delta=f"{scenario_recommendation.recommended_order_quantity - base_recommendation.recommended_order_quantity:+,.0f}",
    )
    row[3].metric(
        "Stockout risk",
        risk_badge(scenario_recommendation.stockout_risk),
        help=f"Estimated probability {scenario_recommendation.stockout_probability:.0%}.",
    )

    base_forecast = context.service.forecast_sku(as_of, sku).to_frame()
    scenario_forecast = context.service.forecast_sku(as_of, sku, scenario=scenario).to_frame()
    st.plotly_chart(
        scenario_comparison_chart(base_forecast, scenario_forecast, "Scenario", title=f"{sku} — forecast under the scenario"),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Base case")
        st.markdown(explain_recommendation(base_recommendation).full_text())
    with right:
        st.subheader("Scenario")
        st.markdown(explain_recommendation(scenario_recommendation).full_text())

    with st.expander("Side-by-side figures"):
        comparison = pd.DataFrame(
            {
                "Measure": [
                    "Inventory position", "Expected demand over protection period",
                    "Safety stock", "Target stock", "Raw order", "Recommended order", "Order value (£)",
                ],
                "Base case": [
                    base_recommendation.inventory_position, base_recommendation.expected_demand_protection,
                    base_recommendation.safety_stock, base_recommendation.target_stock,
                    base_recommendation.raw_order_quantity, base_recommendation.recommended_order_quantity,
                    base_recommendation.order_value,
                ],
                "Scenario": [
                    scenario_recommendation.inventory_position, scenario_recommendation.expected_demand_protection,
                    scenario_recommendation.safety_stock, scenario_recommendation.target_stock,
                    scenario_recommendation.raw_order_quantity, scenario_recommendation.recommended_order_quantity,
                    scenario_recommendation.order_value,
                ],
            }
        ).round(1)
        st.dataframe(comparison, use_container_width=True, hide_index=True)


def page_simulation(context: DashboardContext, as_of: pd.Timestamp) -> None:
    st.header("Simulation and evaluation")
    st.caption(
        f"Historical replay from **{pd.Timestamp(context.config.simulation.start_date):%d %b %Y}** to "
        f"**{pd.Timestamp(context.config.simulation.end_date):%d %b %Y}**. On every simulated day the "
        "ordering decision is taken before that day's demand is revealed."
    )

    if not st.session_state.get("run_simulation") and not st.session_state.get("simulation_done"):
        st.info(
            "Replaying six months of trading for both policies takes about a minute on a laptop. "
            "The result is cached for the rest of the session."
        )
        if st.button("Run the comparison", type="primary"):
            st.session_state["run_simulation"] = True
            st.rerun()
        return

    results = get_comparison(context)
    st.session_state["simulation_done"] = True
    comparison = results["comparison"]
    baseline_label = results["baseline"].strategy_name
    ai_label = results["ai"].strategy_name

    st.subheader("Headline comparison")
    st.caption(
        "Both policies use the same supplier calendar, lead time, opening stock and cost model. "
        "They differ in the demand signal that drives the order."
    )
    row = st.columns(4)
    indexed = comparison.set_index("key")
    for column, key, label in zip(
        row,
        ["unit_service_level", "waste_units", "stockout_units", "total_cost"],
        ["Service level", "Waste units", "Stockout units", "Total cost"],
    ):
        column.metric(
            label,
            f"{indexed.loc[key, ai_label]:,.1f}",
            delta=f"{indexed.loc[key, 'improvement_pct']:+.1f}% vs baseline",
            delta_color="normal",
        )

    st.dataframe(
        comparison[["metric", "unit", baseline_label, ai_label, "improvement_pct"]].rename(
            columns={"improvement_pct": "Improvement (%)"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "A positive improvement always means the AI policy performed better on that measure. "
        "These two runs sit at whatever operating point each policy's settings produce — see the "
        "frontier below before drawing conclusions from a single pair of runs."
    )

    st.plotly_chart(
        kpi_comparison_chart(
            comparison,
            ["waste_units", "stockout_units", "total_cost", "average_inventory_units"],
            baseline_label,
            ai_label,
            title="Key measures, indexed to the baseline",
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Comparing the policies fairly")
    st.markdown(
        "Any ordering policy can trade waste for availability by holding more stock, so two runs "
        "at different service levels cannot be ranked directly. Sweeping each policy across its "
        "tuning parameter shows the whole trade-off: the lower curve achieves the same "
        "availability for less waste."
    )
    if st.button("Run the service-level sweep"):
        st.session_state["run_frontier"] = True
    if st.session_state.get("run_frontier"):
        frontier = get_frontier(context)
        st.plotly_chart(frontier_chart(frontier), use_container_width=True)
        matched = matched_comparison(frontier)
        if matched.empty:
            st.warning("The two policies' service levels do not overlap; widen the sweep to compare them.")
        else:
            st.dataframe(matched, use_container_width=True, hide_index=True)
            best = matched.loc[matched["waste_reduction_pct"].idxmax()]
            st.success(
                f"At a matched service level of {best['matched_service_level_pct']:.2f}%, the "
                f"forecast-driven policy produces {best['waste_reduction_pct']:.1f}% less waste, "
                f"{best['cost_reduction_pct']:.1f}% lower total cost and "
                f"{best['inventory_reduction_pct']:.1f}% less average inventory."
            )
        with st.expander("Sweep results"):
            st.dataframe(frontier, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Over time")
    series = {
        baseline_label: daily_totals(results["baseline"].daily),
        ai_label: daily_totals(results["ai"].daily),
    }
    metric_choice = st.selectbox(
        "Measure",
        ["closing_stock", "waste_units", "stockout_units", "total_cost"],
        format_func=lambda k: {
            "closing_stock": "Stock on hand (units)",
            "waste_units": "Waste (units)",
            "stockout_units": "Unmet demand (units)",
            "total_cost": "Daily cost (£)",
        }[k],
    )
    st.plotly_chart(
        simulation_timeseries_chart(
            series, metric_choice,
            {
                "closing_stock": "Stock on hand (units)",
                "waste_units": "Waste (units)",
                "stockout_units": "Unmet demand (units)",
                "total_cost": "Daily cost (£)",
            }[metric_choice],
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("By product")
    per_sku = kpis_by_sku(results["ai"].daily, results["ai"].orders).merge(
        kpis_by_sku(results["baseline"].daily, results["baseline"].orders),
        on="sku", suffixes=("_ai", "_baseline"),
    )
    st.dataframe(
        per_sku[[
            "sku", "unit_service_level_baseline", "unit_service_level_ai",
            "waste_units_baseline", "waste_units_ai",
            "total_cost_baseline", "total_cost_ai",
        ]].round(1),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Which model is used for each product, and why"):
        st.dataframe(
            ModelSelector.to_frame(context.model_choices)[["sku", "segment", "model_name", "reason"]],
            use_container_width=True,
            hide_index=True,
        )


# -- entry point -----------------------------------------------------------------------


def main() -> None:
    context = get_context()
    page, as_of = sidebar(context)

    if page == "Overview":
        page_overview(context, as_of)
    elif page == "Forecast":
        page_forecast(context, as_of)
    elif page == "Recommendations":
        page_recommendations(context, as_of)
    elif page == "What-if":
        page_what_if(context, as_of)
    else:
        page_simulation(context, as_of)


if __name__ == "__main__":
    main()
