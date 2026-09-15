"""Plotly figures for the dashboard.

Kept free of Streamlit so the figures can be produced and inspected from a script or a
notebook. Every figure follows one set of conventions:

* a fixed categorical order - blue, orange, aqua - assigned to entities, never cycled by
  rank, so a series keeps its colour when a filter changes the selection;
* a diverging blue/red pair reserved for signed quantities (a driver that raised or
  lowered the forecast), which is the only place colour carries polarity;
* thin marks, a hairline recessive grid, one y-axis per figure, and a legend whenever
  more than one series is plotted;
* every chart is accompanied in the app by the table it was built from, which is also
  the accessibility fallback for the lighter hues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -- design tokens ---------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e9e8e4"

#: Fixed categorical order. Validated for colour-vision deficiency on the light surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
#: Diverging pair for signed values, with a neutral midpoint.
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
#: Status palette, always paired with a text label - never colour alone.
STATUS = {"low": "#0ca30c", "medium": "#fab219", "high": "#d03b3b"}

FONT = dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color=TEXT_PRIMARY)


def _base_layout(title: str = "", height: int = 380, showlegend: bool = True) -> dict:
    """Shared layout: recessive axes, generous margins, legend above the plot."""
    return dict(
        title=dict(text=title, font=dict(size=15, color=TEXT_PRIMARY), x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 30, b=10),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=FONT,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=12)),
        hovermode="x unified",
        xaxis=dict(showgrid=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY)),
        yaxis=dict(
            gridcolor=GRID, griddash="solid", gridwidth=1, zeroline=False,
            linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY),
        ),
    )


def demand_forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    actuals: pd.DataFrame | None = None,
    title: str = "Demand history and forecast",
) -> go.Figure:
    """Historical demand with the forecast and its uncertainty interval.

    Args:
        history: Observed demand with ``date`` and ``units_sold``.
        forecast: Forecast with ``forecast_date``, ``predicted_demand`` and optional
            ``lower_bound`` / ``upper_bound``.
        actuals: Optional realised demand over the forecast window, shown for backtesting.
        title: Figure title.
    """
    figure = go.Figure()

    if not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history["date"], y=history["units_sold"], name="Observed demand",
                mode="lines", line=dict(color=SERIES[0], width=2, shape="linear"),
                hovertemplate="%{y:.0f} units<extra>Observed</extra>",
            )
        )

    has_interval = {"lower_bound", "upper_bound"} <= set(forecast.columns)
    if has_interval and forecast["upper_bound"].notna().all():
        band_x = list(forecast["forecast_date"]) + list(forecast["forecast_date"][::-1])
        band_y = list(forecast["upper_bound"]) + list(forecast["lower_bound"][::-1])
        figure.add_trace(
            go.Scatter(
                x=band_x, y=band_y, fill="toself", fillcolor="rgba(235, 104, 52, 0.10)",
                # mode must be explicit: Plotly defaults to lines+markers below 20 points,
                # which would scatter dots along the band outline.
                mode="lines", line=dict(width=0), name="Forecast range", hoverinfo="skip",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=forecast["forecast_date"], y=forecast["predicted_demand"], name="Forecast",
            mode="lines+markers", line=dict(color=SERIES[1], width=2, dash="dot"),
            marker=dict(size=8, color=SERIES[1], line=dict(width=2, color=SURFACE)),
            hovertemplate="%{y:.0f} units<extra>Forecast</extra>",
        )
    )

    if actuals is not None and not actuals.empty:
        figure.add_trace(
            go.Scatter(
                x=actuals["date"], y=actuals["units_sold"], name="Actual (holdout)",
                mode="lines+markers", line=dict(color=SERIES[2], width=2),
                marker=dict(size=8, color=SERIES[2], line=dict(width=2, color=SURFACE)),
                hovertemplate="%{y:.0f} units<extra>Actual</extra>",
            )
        )

    if not forecast.empty:
        origin = pd.Timestamp(forecast["forecast_date"].min())
        figure.add_vline(x=origin, line=dict(color=TEXT_SECONDARY, width=1, dash="dot"))
        figure.add_annotation(
            x=origin, y=1.0, yref="paper", text="decision date", showarrow=False,
            xanchor="left", yanchor="bottom", font=dict(size=11, color=TEXT_SECONDARY),
        )

    layout = _base_layout(title)
    layout["yaxis"]["title"] = dict(text="Units per day", font=dict(size=12, color=TEXT_SECONDARY))
    figure.update_layout(**layout)
    return figure


def forecast_accuracy_chart(
    daily: pd.DataFrame,
    title: str = "Forecast versus actual demand, all products",
) -> go.Figure:
    """Total demand against the forecast, with the daily error given its own scale.

    Two stacked panels share one date axis. The upper compares total actual and
    predicted units; the lower plots the signed error on an axis of its own, which is
    what makes the size of a miss readable - a 40-unit error is invisible against a
    3,000-unit total but plain when it has its own scale. This is the one figure in the
    app with two vertical scales; they are stacked panels rather than a dual axis on
    shared marks, so no mark is read against the wrong one.

    Args:
        daily: Frame from :func:`~src.app_services.recent_forecast_accuracy`, with
            ``date``, ``actual_units``, ``predicted_units`` and ``error_units``.
        title: Figure title.
    """
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.66, 0.34], vertical_spacing=0.07,
    )
    if daily.empty:
        figure.update_layout(**_base_layout(title, height=460, showlegend=False))
        figure.add_annotation(
            text="Not enough history to replay a forecast window.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=TEXT_SECONDARY),
        )
        return figure

    figure.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["actual_units"], name="Actual demand",
            mode="lines", line=dict(color=SERIES[0], width=2),
            hovertemplate="%{y:,.0f} units<extra>Actual</extra>",
        ),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["predicted_units"], name="Forecast",
            mode="lines", line=dict(color=SERIES[1], width=2, dash="dot"),
            hovertemplate="%{y:,.0f} units<extra>Forecast</extra>",
        ),
        row=1, col=1,
    )

    error = daily["error_units"].to_numpy(dtype=float)
    figure.add_trace(
        go.Bar(
            x=daily["date"], y=error, name="Forecast − actual",
            marker=dict(color=[POSITIVE if value >= 0 else NEGATIVE for value in error]),
            customdata=daily["error_pct"].to_numpy(dtype=float),
            hovertemplate="%{y:+,.0f} units (%{customdata:+.1f}%)<extra>Error</extra>",
        ),
        row=2, col=1,
    )

    layout = _base_layout(title, height=470)
    axis_x = layout.pop("xaxis")
    axis_y = layout.pop("yaxis")
    figure.update_layout(**layout, bargap=0.35)
    for row in (1, 2):
        figure.update_xaxes(**axis_x, row=row, col=1)
        figure.update_yaxes(**axis_y, row=row, col=1)

    axis_font = dict(size=12, color=TEXT_SECONDARY)
    figure.update_yaxes(title=dict(text="Units per day", font=axis_font), row=1, col=1)
    figure.update_yaxes(
        title=dict(text="Forecast − actual (units)", font=axis_font), row=2, col=1
    )
    figure.add_hline(y=0, line=dict(color=TEXT_SECONDARY, width=1), row=2, col=1)
    return figure


def driver_chart(drivers: pd.DataFrame, title: str = "What contributed to this forecast") -> go.Figure:
    """Signed driver contributions as a diverging horizontal bar chart.

    Args:
        drivers: Frame with ``label`` and ``contribution_units``, ordered by importance.
        title: Figure title.
    """
    figure = go.Figure()
    if drivers.empty:
        figure.update_layout(**_base_layout(title, height=200, showlegend=False))
        figure.add_annotation(
            text="This model does not expose feature-level contributions.",
            showarrow=False, font=dict(color=TEXT_SECONDARY),
        )
        return figure

    frame = drivers.sort_values("contribution_units")
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in frame["contribution_units"]]
    figure.add_trace(
        go.Bar(
            x=frame["contribution_units"], y=frame["label"], orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{v:+.0f}" for v in frame["contribution_units"]],
            textposition="outside", textfont=dict(color=TEXT_SECONDARY, size=12),
            hovertemplate="%{y}: %{x:+.1f} units<extra></extra>",
            showlegend=False,
        )
    )
    layout = _base_layout(title, height=max(240, 42 * len(frame)), showlegend=False)
    layout["hovermode"] = "closest"
    layout["bargap"] = 0.45
    # Outside labels need room, or the longest bar's value is clipped by the plot edge.
    extent = float(frame["contribution_units"].abs().max()) or 1.0
    layout["xaxis"] = dict(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=True, zerolinecolor=TEXT_SECONDARY,
        zerolinewidth=1, tickfont=dict(color=TEXT_SECONDARY),
        range=[
            min(float(frame["contribution_units"].min()), 0.0) - 0.18 * extent,
            max(float(frame["contribution_units"].max()), 0.0) + 0.18 * extent,
        ],
        title=dict(text="Units added to / removed from a typical midweek day", font=dict(size=12, color=TEXT_SECONDARY)),
    )
    layout["yaxis"] = dict(showgrid=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY))
    figure.update_layout(**layout)
    return figure


def kpi_comparison_chart(
    comparison: pd.DataFrame,
    metrics: list[str],
    baseline_label: str,
    ai_label: str,
    title: str = "Baseline versus AI",
) -> go.Figure:
    """Grouped bars comparing the two policies on selected KPIs.

    Values are indexed to the baseline (baseline = 100) so metrics on different scales
    share one axis - a second y-axis would make the comparison unreadable.
    """
    frame = comparison[comparison["key"].isin(metrics)].copy()
    if frame.empty:
        return go.Figure(layout=_base_layout(title, showlegend=False))

    frame["baseline_indexed"] = 100.0
    frame["ai_indexed"] = np.where(
        frame[baseline_label] != 0, frame[ai_label] / frame[baseline_label] * 100, np.nan
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["metric"], y=frame["baseline_indexed"], name=baseline_label,
            marker=dict(color=SERIES[0], cornerradius=4),
            customdata=frame[baseline_label],
            hovertemplate="%{x}<br>%{customdata:,.1f}<extra>" + baseline_label + "</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=frame["metric"], y=frame["ai_indexed"], name=ai_label,
            marker=dict(color=SERIES[1], cornerradius=4),
            customdata=frame[ai_label],
            text=[f"{v:,.0f}" for v in frame["ai_indexed"]],
            textposition="outside", textfont=dict(color=TEXT_SECONDARY, size=12),
            hovertemplate="%{x}<br>%{customdata:,.1f}<extra>" + ai_label + "</extra>",
        )
    )
    layout = _base_layout(title, height=400)
    layout["barmode"] = "group"
    layout["bargap"] = 0.35
    layout["bargroupgap"] = 0.08
    layout["hovermode"] = "closest"
    layout["yaxis"]["title"] = dict(
        text="Indexed to baseline (100 = baseline)", font=dict(size=12, color=TEXT_SECONDARY)
    )
    figure.update_layout(**layout)
    return figure


def frontier_chart(
    frontier: pd.DataFrame,
    x: str = "unit_service_level",
    y: str = "waste_units",
    title: str = "Waste against service level",
) -> go.Figure:
    """The service-level frontier for both policies.

    Each policy traces a curve as its tuning parameter is swept. A curve lying below and
    to the right of the other achieves more availability for less waste.
    """
    figure = go.Figure()
    for index, (strategy, group) in enumerate(frontier.groupby("strategy", observed=True)):
        group = group.sort_values(x)
        color = SERIES[index % len(SERIES)]
        figure.add_trace(
            go.Scatter(
                x=group[x], y=group[y], name=str(strategy), mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=9, color=color, line=dict(width=2, color=SURFACE)),
                customdata=group["setting_value"],
                hovertemplate=(
                    "Service level %{x:.2f}%<br>%{y:,.0f} units of waste"
                    "<br>setting %{customdata:.3f}<extra>" + str(strategy) + "</extra>"
                ),
            )
        )
    layout = _base_layout(title, height=420)
    layout["hovermode"] = "closest"
    layout["xaxis"] = dict(
        showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID,
        tickfont=dict(color=TEXT_SECONDARY),
        title=dict(text="Service level (unit fill rate, %)", font=dict(size=12, color=TEXT_SECONDARY)),
    )
    layout["yaxis"]["title"] = dict(text="Waste (units)", font=dict(size=12, color=TEXT_SECONDARY))
    figure.update_layout(**layout)
    return figure


def simulation_timeseries_chart(
    series: dict[str, pd.DataFrame],
    column: str,
    label: str,
    title: str = "",
) -> go.Figure:
    """Daily site-level series for one metric, one line per policy."""
    figure = go.Figure()
    for index, (name, frame) in enumerate(series.items()):
        color = SERIES[index % len(SERIES)]
        rolling = frame[column].rolling(7, min_periods=1).mean()
        figure.add_trace(
            go.Scatter(
                x=frame["date"], y=rolling, name=name, mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="%{y:,.1f}<extra>" + name + "</extra>",
            )
        )
    layout = _base_layout(title or f"{label} (7-day rolling mean)")
    layout["yaxis"]["title"] = dict(text=label, font=dict(size=12, color=TEXT_SECONDARY))
    figure.update_layout(**layout)
    return figure


def stock_cover_chart(recommendations: pd.DataFrame, title: str = "Days of cover by SKU") -> go.Figure:
    """Horizontal bars of current cover, coloured by risk and labelled with the level."""
    frame = recommendations.sort_values("days_of_cover")
    colors = [STATUS.get(str(r), SERIES[0]) for r in frame["risk"]]
    figure = go.Figure(
        go.Bar(
            x=frame["days_of_cover"], y=frame["sku"], orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{v:.1f} d · {r}" for v, r in zip(frame["days_of_cover"], frame["risk"])],
            textposition="outside", textfont=dict(color=TEXT_SECONDARY, size=11),
            hovertemplate="%{y}: %{x:.1f} days of cover<extra></extra>",
            showlegend=False,
        )
    )
    layout = _base_layout(title, height=max(280, 26 * len(frame)), showlegend=False)
    layout["hovermode"] = "closest"
    layout["bargap"] = 0.4
    layout["xaxis"] = dict(
        showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=TEXT_SECONDARY),
        title=dict(text="Days of forecast demand covered by stock on hand", font=dict(size=12, color=TEXT_SECONDARY)),
    )
    layout["yaxis"] = dict(showgrid=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY))
    figure.update_layout(**layout)
    return figure


def scenario_comparison_chart(
    base: pd.DataFrame,
    scenario: pd.DataFrame,
    scenario_label: str,
    title: str = "Base case versus scenario",
) -> go.Figure:
    """Forecast under the base case and under a what-if scenario."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=base["forecast_date"], y=base["predicted_demand"], name="Base case",
            mode="lines+markers", line=dict(color=SERIES[0], width=2),
            marker=dict(size=8, color=SERIES[0], line=dict(width=2, color=SURFACE)),
            hovertemplate="%{y:.0f} units<extra>Base case</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=scenario["forecast_date"], y=scenario["predicted_demand"], name=scenario_label,
            mode="lines+markers", line=dict(color=SERIES[1], width=2, dash="dot"),
            marker=dict(size=8, color=SERIES[1], line=dict(width=2, color=SURFACE)),
            hovertemplate="%{y:.0f} units<extra>" + scenario_label + "</extra>",
        )
    )
    layout = _base_layout(title)
    layout["yaxis"]["title"] = dict(text="Units per day", font=dict(size=12, color=TEXT_SECONDARY))
    figure.update_layout(**layout)
    return figure
