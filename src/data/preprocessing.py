"""Time-aware slicing helpers.

This module owns the single convention the whole project relies on to stay free of
data leakage:

    A decision taken on day ``as_of`` may use observed demand up to and including
    ``as_of - 1`` only. The forecast covers ``as_of`` .. ``as_of + horizon - 1``.

Every backtest, simulation step and dashboard forecast obtains its training data
through :func:`history_as_of`, so the rule is enforced in one place instead of being
re-implemented (and eventually broken) at each call site.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns that are only observable after the fact and must never reach a forecast
#: input frame for a future date.
OUTCOME_COLUMNS = ("units_sold",)


def history_as_of(
    sales: pd.DataFrame,
    as_of: pd.Timestamp,
    sku: str | None = None,
) -> pd.DataFrame:
    """Return all observations strictly before ``as_of``.

    Args:
        sales: Tidy sales frame containing a ``date`` column.
        as_of: The decision date. Demand recorded on this date is *not* yet known.
        sku: Optional SKU filter.

    Returns:
        A copy of the qualifying rows, sorted by (sku, date).
    """
    as_of = pd.Timestamp(as_of)
    mask = sales["date"] < as_of
    if sku is not None:
        mask &= sales["sku"] == sku
    out = sales.loc[mask].sort_values(["sku", "date"], kind="stable")
    return out.reset_index(drop=True)


def actuals_window(
    sales: pd.DataFrame,
    start: pd.Timestamp,
    horizon: int,
    sku: str | None = None,
) -> pd.DataFrame:
    """Return realised demand for the evaluation window ``[start, start + horizon)``.

    This is for scoring a forecast *after* the fact. It must never be fed into a model
    fit or an order recommendation for the same window.
    """
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=horizon)
    mask = (sales["date"] >= start) & (sales["date"] < end)
    if sku is not None:
        mask &= sales["sku"] == sku
    return sales.loc[mask].sort_values(["sku", "date"], kind="stable").reset_index(drop=True)


def horizon_dates(as_of: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Generate the forecast dates for a decision taken on ``as_of``.

    The window starts on the decision date itself and spans ``horizon`` calendar days.

    Raises:
        ValueError: If ``horizon`` is not a positive integer.
    """
    if horizon < 1:
        raise ValueError(f"Forecast horizon must be >= 1 day, got {horizon}.")
    return pd.date_range(pd.Timestamp(as_of), periods=int(horizon), freq="D")


def to_wide(sales: pd.DataFrame, value: str = "units_sold") -> pd.DataFrame:
    """Pivot a tidy frame to a date x SKU matrix."""
    return sales.pivot(index="date", columns="sku", values=value).sort_index()


def rolling_origins(
    dates: pd.DatetimeIndex,
    n_folds: int,
    horizon: int,
    min_train_days: int = 0,
) -> list[pd.Timestamp]:
    """Compute expanding-window rolling-origin cut points.

    Folds are laid out so that the final fold's evaluation window ends on the last
    available date, matching the AE1 baseline experiment.

    Args:
        dates: The full sorted date index of the dataset.
        n_folds: Number of evaluation folds.
        horizon: Forecast horizon in days.
        min_train_days: Reject origins that would leave less training history than this.

    Returns:
        Forecast origin timestamps, earliest first. Each origin is the first date of
        its evaluation window.

    Raises:
        ValueError: If the date index is too short for the requested layout.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    needed = horizon * n_folds + min_train_days
    if len(dates) < needed:
        raise ValueError(
            f"Need at least {needed} days for {n_folds} folds at horizon {horizon} "
            f"(min_train_days={min_train_days}) but only {len(dates)} are available."
        )
    origins = [dates[len(dates) - horizon * (n_folds - k)] for k in range(n_folds)]
    return [pd.Timestamp(o) for o in origins]


def assert_no_outcome_columns(frame: pd.DataFrame, context: str = "future frame") -> None:
    """Fail loudly if a frame intended for prediction still carries realised demand.

    Raises:
        ValueError: If any column in :data:`OUTCOME_COLUMNS` is present.
    """
    present = [c for c in OUTCOME_COLUMNS if c in frame.columns]
    if present:
        raise ValueError(
            f"{context} contains outcome column(s) {present}. Realised demand must not "
            "be available when forecasting or recommending an order."
        )
