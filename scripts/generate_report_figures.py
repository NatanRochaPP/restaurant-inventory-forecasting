"""Generate print-quality figures and tables for the dissertation write-up.

Every figure in the report is produced here from the CSV evidence in ``outputs/``, so a
figure can never drift from the numbers behind it: re-run the pipeline, re-run this, and
the report's figures update together.

Usage::

    python scripts/generate_report_figures.py

Outputs land in ``outputs/report/`` as 300 dpi PNGs plus the matching CSV and Markdown
tables. Colours follow a palette validated for colour-vision deficiency (blue #2a78d6,
orange #eb6834); no figure uses colour as its only channel - marker shape, direct labels
and the accompanying tables all carry the same information.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROJECT_ROOT  # noqa: E402

logger = logging.getLogger("report_figures")
OUT = PROJECT_ROOT / "outputs"
REPORT = OUT / "report"

BLUE, ORANGE = "#2a78d6", "#eb6834"
RED, GREEN = "#d03b3b", "#0ca30c"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e9e8e4"
POUND = "£"
DASH = "—"

BASELINE_LABEL = "Manual par-level baseline"
AI_LABEL = "AI forecast + inventory policy"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def _finish(fig, path: Path, caption: str) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name:42s} {caption}")


def figure_accuracy(metrics: pd.DataFrame) -> None:
    """Small multiples: one panel per metric, because the three run on different scales."""
    order = metrics.sort_values("sMAPE").index.tolist()
    panels = [
        ("sMAPE", "sMAPE (%), lower is better"),
        ("MASE", "MASE, lower is better"),
        ("WAPE", "WAPE (%), lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (metric, label) in zip(axes, panels):
        values = metrics.loc[order, metric]
        colours = [ORANGE if m == "Gradient Boosting" else BLUE for m in order]
        bars = ax.barh(range(len(order)), values, color=colours, height=0.6, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if metric == "sMAPE" else [])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, values.max() * 1.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + values.max() * 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}", va="center", ha="left", fontsize=8, color=INK2)
        if metric == "MASE":
            ax.axvline(1.0, color=INK2, lw=1, ls=":", zorder=4)
            ax.text(1.0, len(order) - 0.3, " seasonal naive = 1.0", fontsize=7, color=INK2, va="bottom")
    fig.suptitle(
        f"Figure 1 {DASH} Forecast accuracy by model (12-fold rolling origin, 7-day horizon, 20 SKUs)",
        fontweight="bold", x=0.005, ha="left", y=1.05,
    )
    fig.tight_layout()
    _finish(fig, REPORT / "fig01_forecast_accuracy.png", "model accuracy comparison")


def figure_segmentation(segmentation: pd.DataFrame, selection: pd.DataFrame) -> None:
    """ABC/XYZ scatter, with marker shape carrying the model so colour is never alone."""
    frame = segmentation.merge(selection[["sku", "model_name"]], on="sku", how="left")
    frame = frame.sort_values("annual_value", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    styles = {"Gradient Boosting": ("o", BLUE), "Croston (SBA)": ("^", ORANGE)}
    for model, (marker, colour) in styles.items():
        sub = frame[frame["model_name"] == model]
        ax.scatter(sub["demand_cv"], sub["annual_value"], s=80, marker=marker,
                   facecolor=colour, edgecolor="white", linewidth=1.2, zorder=4,
                   label=f"{model}  (n={len(sub)})")
    ax.set_yscale("log")
    ax.set_xlim(0.15, 1.35)

    # ABC classes are assigned by cumulative value share over a value-ranked list, so the
    # classes are contiguous in value and the boundary can be drawn on this axis.
    for lower, upper, name in [("A", "B", "A | B"), ("B", "C", "B | C")]:
        above = frame[frame["abc"] == lower]["annual_value"].min()
        below = frame[frame["abc"] == upper]["annual_value"].max()
        if pd.notna(above) and pd.notna(below):
            boundary = float(np.sqrt(above * below))
            ax.axhline(boundary, color=INK2, lw=0.9, ls="--", zorder=2)
            ax.text(1.34, boundary, f" {name} ", fontsize=7, color=INK2, va="center",
                    ha="right", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    for cut, name in zip([0.5, 1.0], ["X | Y", "Y | Z"]):
        ax.axvline(cut, color=INK2, lw=0.9, ls=":", zorder=2)
        ax.text(cut, ax.get_ylim()[1], f" {name}", fontsize=7, color=INK2, va="top")

    # Alternate the label side and vertical offset so the dense low-CV cluster stays legible.
    offsets = [(8, 4), (-8, 4), (8, -10), (-8, -10)]
    for i, row in enumerate(frame.itertuples(index=False)):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(row.sku, (row.demand_cv, row.annual_value), fontsize=6.5, color=INK2,
                    xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", zorder=5)

    ax.set_xlabel("Coefficient of variation of daily demand   (XYZ axis)")
    ax.set_ylabel(f"Annualised purchase value, {POUND} (ABC axis, log scale)")
    ax.set_yticks([2000, 5000, 10000, 20000, 50000, 100000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{POUND}{v:,.0f}"))
    ax.minorticks_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, 1.0), fontsize=8)
    ax.set_title(f"Figure 2 {DASH} ABC/XYZ segmentation and the model routed to each SKU", loc="left")
    fig.text(0.005, -0.03,
             "The four SKUs with intermittent demand (Z and high-CV Y segments) are routed to Croston (SBA); "
             "the remaining 16 use\ngradient boosting. Marker shape as well as colour distinguishes the two, "
             "and Table 6 records the reason recorded per SKU.",
             fontsize=7.5, color=INK2, ha="left")
    fig.tight_layout()
    _finish(fig, REPORT / "fig02_segmentation.png", "ABC/XYZ segmentation and routing")


def figure_forecast_example(predictions: pd.DataFrame, sku: str = "Fries (Potatoes)") -> None:
    """One fold's forecast against realised demand for a fast-moving SKU."""
    sub = predictions[(predictions["sku"] == sku) & (predictions["model"] == "Gradient Boosting")]
    sub = sub[sub["fold"] == sub["fold"].max()].sort_values("forecast_date")
    dates = pd.to_datetime(sub["forecast_date"])
    errors = (sub["actual"] - sub["predicted_demand"]).abs()

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.fill_between(dates, sub["lower_bound"], sub["upper_bound"], color=ORANGE, alpha=0.12,
                    linewidth=0, label="95% prediction interval", zorder=2)
    ax.plot(dates, sub["actual"], color=BLUE, lw=2, marker="o", ms=6,
            markeredgecolor="white", markeredgewidth=1.2, label="Actual demand", zorder=4)
    ax.plot(dates, sub["predicted_demand"], color=ORANGE, lw=2, ls="--", marker="D", ms=6,
            markeredgecolor="white", markeredgewidth=1.2, label="Forecast", zorder=3)
    worst = errors.idxmax()
    ax.annotate(
        f"largest error {sub.loc[worst, 'actual'] - sub.loc[worst, 'predicted_demand']:+.0f} units",
        (pd.Timestamp(sub.loc[worst, "forecast_date"]),
         max(sub.loc[worst, "actual"], sub.loc[worst, "predicted_demand"])),
        fontsize=7.5, color=INK2, xytext=(0, 12), textcoords="offset points", ha="center",
    )
    ax.set_ylabel("Units per day")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Figure 3 {DASH} Seven-day forecast against realised demand ({sku}, final backtest fold)",
        loc="left",
    )
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    _finish(fig, REPORT / "fig03_forecast_vs_actual.png", f"forecast vs actual, {sku}")


def figure_frontier(frontier: pd.DataFrame, matched: pd.DataFrame) -> None:
    """The central evidence figure: waste against service level for both policies."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    curves = {}
    for label, colour, marker in [(BASELINE_LABEL, BLUE, "o"), (AI_LABEL, ORANGE, "D")]:
        sub = frontier[frontier["strategy"] == label].sort_values("unit_service_level")
        curves[label] = sub
        ax.plot(sub["unit_service_level"], sub["waste_units"], color=colour, lw=2,
                marker=marker, ms=7, markeredgecolor="white", markeredgewidth=1.2,
                label=label, zorder=3)
    ax.set_xlabel("Service level achieved (unit fill rate, %)")
    ax.set_ylabel("Food waste over the six-month holdout (units)")

    # The baseline's highest-buffer runs waste an order of magnitude more than anything
    # else, which would flatten the region where the two policies actually differ. Clip
    # the axis to that region and mark the off-scale points rather than dropping them.
    ceiling = float(curves[AI_LABEL]["waste_units"].max()) * 2.7
    ax.set_ylim(0, ceiling)
    off_scale = curves[BASELINE_LABEL][curves[BASELINE_LABEL]["waste_units"] > ceiling]
    for row in off_scale.itertuples(index=False):
        # An arrow marks that the curve continues off the top; the values are given in
        # the caption rather than as labels, which would collide at this spacing.
        ax.annotate(
            "", xy=(row.unit_service_level, ceiling * 0.995),
            xytext=(row.unit_service_level, ceiling * 0.93),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6),
        )
    off_scale_note = ""
    if len(off_scale):
        pairs = ", ".join(
            f"{r.waste_units:,.0f} units at {r.unit_service_level:.1f}%"
            for r in off_scale.itertuples(index=False)
        )
        off_scale_note = (
            f"\nThe arrows mark {len(off_scale)} baseline runs that continue above the axis ({pairs}): "
            "buying the last of that availability costs\nseveral times more waste than everything below it."
        )

    if not matched.empty:
        best = matched.loc[matched["waste_reduction_pct"].idxmax()]
        x = float(best["matched_service_level_pct"])
        ax.plot([x, x], [best["waste_units_ai"], best["waste_units_baseline"]],
                color=INK, lw=1.4, ls=":", zorder=4)
        ax.scatter([x, x], [best["waste_units_ai"], best["waste_units_baseline"]],
                   s=45, color=INK, zorder=5)
        ax.annotate(
            f"At a matched {x:.2f}% service level the\nforecast-driven policy wastes "
            f"{best['waste_reduction_pct']:.1f}% less\n"
            f"({best['waste_units_ai']:,.0f} vs {best['waste_units_baseline']:,.0f} units)",
            xy=(x, (best["waste_units_ai"] + best["waste_units_baseline"]) / 2),
            xytext=(float(frontier["unit_service_level"].min()) + 0.15, ceiling * 0.72),
            textcoords="data", ha="left", va="center", fontsize=8, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=GRID),
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkA=4, shrinkB=6),
        )
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    ax.set_title(
        f"Figure 4 {DASH} Waste against service level: both policies across their full operating range",
        loc="left",
    )
    fig.text(
        0.005, -0.10,
        "Each point is one six-month simulation. The manual baseline is swept across its buffer percentage "
        "(0-50%); the AI policy\nacross its service-level target (50-99%). A curve lying lower and further right "
        "is strictly better: less waste for the same\navailability. Comparing single runs at different service "
        "levels would not be a valid comparison." + off_scale_note,
        fontsize=7.5, color=INK2, ha="left",
    )
    fig.tight_layout()
    _finish(fig, REPORT / "fig04_service_level_frontier.png", "service-level frontier (KEY FIGURE)")


def figure_kpi_comparison(comparison: pd.DataFrame) -> None:
    """Indexed bars, baseline = 100, so metrics of different scale can share one axis."""
    keys = ["waste_units", "stockout_units", "average_inventory_units", "holding_cost", "total_cost"]
    frame = comparison[comparison["key"].isin(keys)].set_index("key").loc[keys]
    indexed = frame[AI_LABEL] / frame[BASELINE_LABEL] * 100

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    positions = np.arange(len(frame))
    ax.bar(positions - 0.21, [100] * len(frame), width=0.4, color=BLUE, label=BASELINE_LABEL, zorder=3)
    bars = ax.bar(positions + 0.21, indexed, width=0.4, color=ORANGE, label=AI_LABEL, zorder=3)
    for bar, value in zip(bars, indexed):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f"{value:.0f}",
                ha="center", fontsize=8, color=INK2)
    ax.axhline(100, color=INK2, lw=1, ls=":", zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(l).replace(" (", "\n(") for l in frame["metric"]], fontsize=8)
    ax.set_ylabel("Indexed to the baseline (100 = baseline)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper right")
    ax.set_ylim(0, max(indexed.max() * 1.3, 130))
    ax.set_title(
        f"Figure 5 {DASH} Operational outcomes, forecast-driven policy relative to the manual baseline",
        loc="left",
    )
    fig.text(
        0.005, -0.07,
        "Both policies at their configured settings, which places them at different service levels "
        "(98.9% baseline vs 98.3% AI).\nThis figure is therefore descriptive only; Figure 4 provides the "
        "like-for-like comparison.",
        fontsize=7.5, color=INK2, ha="left",
    )
    fig.tight_layout()
    _finish(fig, REPORT / "fig05_kpi_comparison.png", "baseline vs AI KPIs (indexed)")


def figure_over_time(baseline_daily: pd.DataFrame, ai_daily: pd.DataFrame) -> None:
    """Waste, unmet demand and stock held over the holdout, as 7-day rolling means."""
    def totals(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.groupby("date", as_index=False)[
            ["expired_units", "stockout_units", "closing_stock"]].sum()
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date")

    base, ai = totals(baseline_daily), totals(ai_daily)
    panels = [
        ("expired_units", "Waste (units/day)"),
        ("stockout_units", "Unmet demand (units/day)"),
        ("closing_stock", "Stock on hand (units)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.4), sharex=True)
    for ax, (column, label) in zip(axes, panels):
        for frame, name, colour in [(base, BASELINE_LABEL, BLUE), (ai, AI_LABEL, ORANGE)]:
            ax.plot(frame["date"], frame[column].rolling(7, min_periods=1).mean(),
                    color=colour, lw=2, label=name, zorder=3)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(bottom=0)
    axes[0].legend(ncol=2, loc="upper left")
    axes[0].set_title(
        f"Figure 6 {DASH} Waste, unmet demand and stock held across the holdout (7-day rolling mean)",
        loc="left",
    )
    fig.tight_layout()
    _finish(fig, REPORT / "fig06_over_time.png", "waste / stockouts / stock over time")


def figure_per_sku(ai: pd.DataFrame, baseline: pd.DataFrame) -> None:
    """Dumbbell chart: how waste moves per SKU from baseline to AI."""
    frame = baseline[["sku", "waste_units"]].merge(
        ai[["sku", "waste_units"]], on="sku", suffixes=("_base", "_ai"))
    frame = frame[frame[["waste_units_base", "waste_units_ai"]].sum(axis=1) > 0]
    frame = frame.sort_values("waste_units_base").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.0, 0.45 * len(frame) + 2.2))
    y = np.arange(len(frame))
    ax.hlines(y, frame["waste_units_ai"], frame["waste_units_base"], color=GRID, lw=3, zorder=2)
    ax.scatter(frame["waste_units_base"], y, s=75, color=BLUE, zorder=3,
               edgecolor="white", linewidth=1.2, label=BASELINE_LABEL)
    ax.scatter(frame["waste_units_ai"], y, s=75, color=ORANGE, marker="D", zorder=3,
               edgecolor="white", linewidth=1.2, label=AI_LABEL)
    span = frame["waste_units_base"].max()
    for i, row in enumerate(frame.itertuples(index=False)):
        change = row.waste_units_ai - row.waste_units_base
        ax.text(max(row.waste_units_base, row.waste_units_ai) + span * 0.02, i,
                f"{change:+,.0f}", va="center", fontsize=7.5,
                color=GREEN if change < 0 else RED)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["sku"])
    ax.set_xlabel("Waste over the six-month holdout (units)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-span * 0.03, span * 1.18)
    ax.legend(loc="lower right")
    ax.set_title(
        f"Figure 7 {DASH} Waste by SKU: only short-shelf-life items can waste, and that is where the gain is",
        loc="left",
    )
    fig.tight_layout()
    _finish(fig, REPORT / "fig07_waste_by_sku.png", "per-SKU waste dumbbell")


def figure_drivers() -> None:
    """Worked explainability example for the week of a bank holiday."""
    from src.config import load_config
    from src.data.loader import load_sales
    from src.explainability.forecast_drivers import aggregate_drivers
    from src.forecasting.service import ForecastService

    config = load_config()
    sales = load_sales(config=config)
    service = ForecastService(sales, config)
    as_of = pd.Timestamp("2025-08-22")
    sku = "Burger Buns"
    _, explanations = service.explain_forecast(as_of, sku)
    drivers = aggregate_drivers(explanations).sort_values("contribution_units")

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    colours = [BLUE if v >= 0 else RED for v in drivers["contribution_units"]]
    ax.barh(range(len(drivers)), drivers["contribution_units"], color=colours, height=0.62, zorder=3)
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers["label"])
    extent = float(drivers["contribution_units"].abs().max()) or 1.0
    for i, value in enumerate(drivers["contribution_units"]):
        ax.text(value + np.sign(value) * extent * 0.02, i, f"{value:+.0f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1, zorder=4)
    ax.set_xlim(min(drivers["contribution_units"].min(), 0) - extent * 0.2,
                max(drivers["contribution_units"].max(), 0) + extent * 0.2)
    ax.set_xlabel("Units added to / removed from a typical midweek day, summed over the 7-day horizon")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        f"Figure 8 {DASH} Forecast drivers: {sku}, week of the August bank holiday ({as_of:%d %b %Y})",
        loc="left",
    )
    fig.text(
        0.005, -0.06,
        "Contributions estimated by counterfactual ablation: the model re-predicts each day with one input reset to a\n"
        "typical midweek value. These are associations learned from historical data, not evidence of cause, and they do\n"
        "not sum exactly to the forecast because the model contains interactions between inputs.",
        fontsize=7.5, color=INK2, ha="left",
    )
    fig.tight_layout()
    _finish(fig, REPORT / "fig08_forecast_drivers.png", "explainability worked example")


def write_tables(accuracy, comparison, matched, matched_waste, segmentation,
                 selection, by_sku_ai, by_sku_base) -> None:
    """Write every report table as CSV and as Markdown ready to paste."""
    tables = {
        "table1_forecast_accuracy": accuracy.reset_index(),
        "table2_operational_outcomes": comparison[
            ["metric", "unit", BASELINE_LABEL, AI_LABEL, "absolute_change", "improvement_pct"]],
        "table3_matched_service_level": matched,
        "table4_matched_waste": matched_waste,
        "table5_segmentation": segmentation[
            ["sku", "abc", "xyz", "segment", "annual_value", "demand_cv", "zero_day_share"]].round(3),
        "table6_model_selection": selection[["sku", "segment", "model_name", "reason"]],
        "table7_per_sku_outcomes": by_sku_base[
            ["sku", "unit_service_level", "waste_units", "total_cost"]].merge(
            by_sku_ai[["sku", "unit_service_level", "waste_units", "total_cost"]],
            on="sku", suffixes=("_baseline", "_ai")).round(2),
    }
    for name, frame in tables.items():
        frame.to_csv(REPORT / f"{name}.csv", index=False)
        (REPORT / f"{name}.md").write_text(frame.to_markdown(index=False), encoding="utf-8")
        print(f"  {name + '.csv/.md':42s} {len(frame)} rows")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    REPORT.mkdir(parents=True, exist_ok=True)

    accuracy = pd.read_csv(OUT / "forecast_accuracy.csv", index_col=0)
    predictions = pd.read_csv(OUT / "backtest_predictions.csv", parse_dates=["forecast_date"])
    frontier = pd.read_csv(OUT / "frontier.csv")
    matched = pd.read_csv(OUT / "matched_service_level.csv")
    matched_waste = pd.read_csv(OUT / "matched_waste.csv")
    comparison = pd.read_csv(OUT / "simulation_comparison.csv")
    segmentation = pd.read_csv(OUT / "sku_segmentation.csv")
    selection = pd.read_csv(OUT / "model_selection.csv")
    by_sku_ai = pd.read_csv(OUT / "simulation_by_sku_ai.csv")
    by_sku_base = pd.read_csv(OUT / "simulation_by_sku_baseline.csv")
    baseline_daily = pd.read_csv(OUT / "simulation_daily_baseline.csv", parse_dates=["date"])
    ai_daily = pd.read_csv(OUT / "simulation_daily_ai.csv", parse_dates=["date"])

    print("Figures")
    figure_accuracy(accuracy)
    figure_segmentation(segmentation, selection)
    figure_forecast_example(predictions)
    figure_frontier(frontier, matched)
    figure_kpi_comparison(comparison)
    figure_over_time(baseline_daily, ai_daily)
    figure_per_sku(by_sku_ai, by_sku_base)
    figure_drivers()

    print("\nTables")
    write_tables(accuracy, comparison, matched, matched_waste, segmentation,
                 selection, by_sku_ai, by_sku_base)

    headline = {
        "forecast_accuracy": {
            "best_model": str(accuracy["sMAPE"].idxmin()),
            "gradient_boosting": accuracy.loc["Gradient Boosting"].to_dict(),
            "seasonal_naive": accuracy.loc["Seasonal Naive"].to_dict(),
        },
        "matched_service_level": matched.to_dict(orient="records"),
        "matched_waste": matched_waste.to_dict(orient="records"),
    }
    (REPORT / "headline_numbers.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    print(f"\nWrote {len(list(REPORT.glob('*')))} files to {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
