"""Calendar features: day of week, weekends, UK bank holidays and school holidays.

Calendar effects are the part of demand that is genuinely known in advance, which is
what makes them legitimate forecast inputs for a future date (see the research question
in README.md). The bank-holiday list mirrors the one used by ``baseline_forecasting.py``
so that features remain comparable with the AE1 baseline results; that script is left
untouched as frozen evidence.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: England & Wales bank holidays covering the dataset plus a forward margin so the
#: dashboard can build feature rows for dates just beyond the historical data.
BANK_HOLIDAYS: pd.DatetimeIndex = pd.to_datetime(
    [
        "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27",
        "2024-08-26", "2024-12-25", "2024-12-26",
        "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26",
        "2025-08-25", "2025-12-25", "2025-12-26",
        "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25",
        "2026-08-31", "2026-12-25", "2026-12-28",
    ]
)

#: Days the restaurant does not trade.
CLOSED_DAYS: pd.DatetimeIndex = pd.to_datetime(["2024-12-25", "2025-12-25", "2026-12-25"])

#: Friday, Saturday and Sunday are the busy days for this site.
WEEKEND_DAYS: tuple[int, ...] = (4, 5, 6)

DAY_NAMES: tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def is_school_holiday(d: pd.Timestamp) -> int:
    """Approximate England state-school holiday flag for a single date.

    Mirrors the rule used to generate the dataset: summer, Christmas, Easter and the
    February, May and October half-terms.
    """
    d = pd.Timestamp(d)
    if d.month in (7, 8):
        return 1
    if d.month == 12 and d.day >= 20:
        return 1
    if d.month == 1 and d.day <= 2:
        return 1
    if d.month == 4 and 1 <= d.day <= 14:
        return 1
    if d.month == 10 and 26 <= d.day <= 31:
        return 1
    if d.month == 2 and 12 <= d.day <= 18:
        return 1
    if d.month == 5 and 26 <= d.day <= 31:
        return 1
    return 0


def calendar_features(dates: pd.DatetimeIndex | pd.Series) -> pd.DataFrame:
    """Build the calendar feature block for a set of dates.

    Args:
        dates: Dates to describe. Order is preserved.

    Returns:
        A frame with columns ``date``, ``dow``, ``weekend``, ``bank_holiday``,
        ``school_holiday``, ``week_of_year``, ``month`` and ``is_closed``.
    """
    idx = pd.DatetimeIndex(pd.Series(dates).values)
    out = pd.DataFrame({"date": idx})
    out["dow"] = idx.dayofweek.astype("int8")
    out["weekend"] = np.isin(idx.dayofweek, WEEKEND_DAYS).astype("int8")
    out["bank_holiday"] = idx.isin(BANK_HOLIDAYS).astype("int8")
    out["school_holiday"] = np.array([is_school_holiday(d) for d in idx], dtype="int8")
    out["week_of_year"] = idx.isocalendar().week.to_numpy(dtype="int16")
    out["month"] = idx.month.astype("int8")
    out["is_closed"] = idx.isin(CLOSED_DAYS).astype("int8")
    return out


def describe_calendar_day(d: pd.Timestamp) -> str:
    """Human-readable label for a date, for use in manager-facing explanations."""
    d = pd.Timestamp(d)
    parts = [DAY_NAMES[d.dayofweek]]
    if d in BANK_HOLIDAYS:
        parts.append("bank holiday")
    if is_school_holiday(d):
        parts.append("school holiday")
    if d in CLOSED_DAYS:
        parts.append("site closed")
    return f"{d:%d %b} ({', '.join(parts)})"


def next_order_date(as_of: pd.Timestamp, order_days: list[int], *, include_today: bool = True) -> pd.Timestamp:
    """Find the next date on which an order may be placed.

    Args:
        as_of: The current date.
        order_days: Weekday numbers on which the supplier accepts orders (0 = Monday).
        include_today: Whether ``as_of`` itself counts as a candidate.

    Returns:
        The next permitted order date.
    """
    if not order_days:
        raise ValueError("order_days must not be empty.")
    d = pd.Timestamp(as_of)
    start = 0 if include_today else 1
    for offset in range(start, start + 8):
        cand = d + pd.Timedelta(days=offset)
        if cand.dayofweek in order_days:
            return cand
    raise ValueError(f"No order day found within 8 days of {as_of} for order_days={order_days}.")


def days_until_next_order(as_of: pd.Timestamp, order_days: list[int]) -> int:
    """Number of days from ``as_of`` to the following order opportunity.

    This is the *effective* review period: the demand a delivery must cover before the
    next chance to reorder arrives.
    """
    nxt = next_order_date(as_of, order_days, include_today=False)
    return int((nxt - pd.Timestamp(as_of)).days)
