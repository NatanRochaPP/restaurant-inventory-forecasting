"""Sales data validation.

The system refuses to produce order recommendations from data it cannot vouch for.
Every check here raises :class:`DataValidationError` with an actionable message rather
than allowing a silently wrong forecast to reach a manager.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import AppConfig

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the sales data violates an assumption the pipeline relies on."""


def validate_sales_frame(df: pd.DataFrame, config: AppConfig, *, strict_grid: bool = True) -> pd.DataFrame:
    """Validate a tidy sales frame.

    Checks performed:
      * all configured required columns are present;
      * ``date`` is datetime-like and ``units_sold`` is non-negative and non-null;
      * there is at most one row per (date, SKU);
      * each SKU has a gap-free daily date grid (``strict_grid``);
      * binary flag columns only contain 0/1;
      * ``shelf_life_days`` is constant within a SKU.

    Args:
        df: Tidy sales frame, one row per SKU-day.
        config: Application configuration supplying the required column list.
        strict_grid: When True, missing calendar days inside a SKU's date range are an
            error. Lag and rolling features assume a contiguous daily grid.

    Returns:
        The same frame, unmodified, once every check has passed.

    Raises:
        DataValidationError: On the first failed check, with the offending detail.
    """
    missing = [c for c in config.data.required_columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Sales data is missing required column(s): {missing}. "
            f"Present columns: {sorted(df.columns)}"
        )

    if df.empty:
        raise DataValidationError("Sales data is empty - cannot forecast or recommend orders.")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise DataValidationError(
            "Column 'date' must be datetime64. Load the CSV with parse_dates=['date']."
        )

    if df["units_sold"].isna().any():
        n = int(df["units_sold"].isna().sum())
        raise DataValidationError(f"Column 'units_sold' contains {n} null value(s).")

    if (df["units_sold"] < 0).any():
        bad = df.loc[df["units_sold"] < 0, ["date", "sku", "units_sold"]].head(5)
        raise DataValidationError(f"Negative demand is not valid. First offending rows:\n{bad}")

    dupes = df.duplicated(subset=["date", "sku"])
    if dupes.any():
        sample = df.loc[dupes, ["date", "sku"]].head(5).to_dict("records")
        raise DataValidationError(
            f"Found {int(dupes.sum())} duplicate (date, sku) row(s); demand would be "
            f"double-counted. Examples: {sample}"
        )

    for flag in ("bank_holiday", "school_holiday", "weekend", "promo"):
        if flag in df.columns:
            values = set(pd.unique(df[flag].dropna()))
            if not values <= {0, 1}:
                raise DataValidationError(
                    f"Flag column '{flag}' must be binary 0/1 but contains {sorted(values)[:6]}."
                )

    shelf = df.groupby("sku", observed=True)["shelf_life_days"].nunique()
    inconsistent = shelf[shelf > 1]
    if not inconsistent.empty:
        raise DataValidationError(
            f"shelf_life_days must be constant per SKU; inconsistent for: {list(inconsistent.index)}"
        )

    if strict_grid:
        _validate_daily_grid(df)

    logger.debug(
        "Validated sales frame: %d rows, %d SKUs, %s to %s",
        len(df), df["sku"].nunique(), df["date"].min().date(), df["date"].max().date(),
    )
    return df


def _validate_daily_grid(df: pd.DataFrame) -> None:
    """Ensure every SKU has a contiguous daily date range."""
    gaps: dict[str, int] = {}
    for sku, grp in df.groupby("sku", observed=True):
        dates = pd.DatetimeIndex(grp["date"].unique()).sort_values()
        expected = pd.date_range(dates.min(), dates.max(), freq="D")
        if len(dates) != len(expected):
            gaps[str(sku)] = len(expected) - len(dates)
    if gaps:
        raise DataValidationError(
            "Missing calendar days would corrupt lag/rolling features. "
            f"Missing day counts by SKU: {gaps}. "
            "Reindex to a complete daily grid before continuing."
        )


def validate_forecast_inputs(history: pd.DataFrame, min_history_days: int, sku: str) -> None:
    """Guard a model fit against insufficient history.

    Raises:
        DataValidationError: If fewer than ``min_history_days`` observations exist.
    """
    if len(history) < min_history_days:
        raise DataValidationError(
            f"SKU '{sku}' has only {len(history)} day(s) of history but "
            f"{min_history_days} are required (forecast.min_history_days). "
            "Extend the training window or lower the threshold in config.yaml."
        )
