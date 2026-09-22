"""Repair of imperfect sales feeds, for running on data the project does not control.

Validation (:mod:`src.data.validation`) refuses a frame that breaks an assumption the
pipeline relies on, which is the right behaviour for a dataset generated once and checked
in. A live feed is different: tills go down, a day arrives twice, a row lands with a
negative quantity, and a run that stops on the first fault leaves the kitchen with no
order at all. This module repairs what can be repaired, records what it did, and leaves
the frame in the state validation expects.

Two repairs are deliberate choices rather than obvious ones.

*Missing days are filled, not dropped.* Lag and rolling features are built with positional
``groupby(...).shift(k)`` (see :mod:`src.features.demand_features`), so a hole in a
product's dates silently moves ``lag7`` onto the wrong day. Filling the hole keeps the
grid honest; the filled value is the median of the same weekday over the preceding four
weeks, which is the best guess available from the product's own history, and every filled
row is flagged so it can be excluded or inspected.

*Calendar columns for a filled day are copied from the other products on that date.* Every
product shares one calendar, so if any product recorded that date its holiday, weekend and
weather values are the real ones. Only when no product recorded the date are the calendar
flags derived from the date itself, and a bank holiday cannot be recovered that way, so the
day is flagged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.config import AppConfig, get_config

logger = logging.getLogger(__name__)

IMPUTED_COLUMN = "imputed"
LOOKBACK_WEEKS = 4


@dataclass(frozen=True)
class DataQualityReport:
    """What a repair pass changed, for logging and for the run log.

    Attributes:
        invalid_rows_dropped: Rows with a missing or negative quantity.
        duplicate_rows_dropped: Repeated ``(date, sku)`` rows beyond the first.
        missing_days_filled: Calendar days inserted inside a product's own date range.
        days_without_calendar: Filled days no other product recorded, so the holiday and
            weather values could not be recovered.
        skus_affected: Products touched by any of the above.
    """

    invalid_rows_dropped: int = 0
    duplicate_rows_dropped: int = 0
    missing_days_filled: int = 0
    days_without_calendar: int = 0
    skus_affected: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        """True when the feed needed no repair."""
        return not (self.invalid_rows_dropped or self.duplicate_rows_dropped or self.missing_days_filled)

    def summary(self) -> str:
        if self.clean:
            return "Sales feed needed no repair."
        parts = []
        if self.invalid_rows_dropped:
            parts.append(f"{self.invalid_rows_dropped} invalid row(s) dropped")
        if self.duplicate_rows_dropped:
            parts.append(f"{self.duplicate_rows_dropped} duplicate row(s) dropped")
        if self.missing_days_filled:
            note = f"{self.missing_days_filled} missing day(s) filled"
            if self.days_without_calendar:
                note += f" ({self.days_without_calendar} without a calendar to copy)"
            parts.append(note)
        return "Sales feed repaired: " + "; ".join(parts) + f". Products affected: {', '.join(self.skus_affected)}."


def _calendar_by_date(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row per date holding the calendar and weather values products share."""
    if not columns:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    return df.groupby("date")[columns].first()


def _fill_value(history: pd.Series, day: pd.Timestamp) -> float:
    """Best guess for a missing day: same weekday recently, else the product's median."""
    recent = history[(history.index < day) & (history.index >= day - pd.Timedelta(weeks=LOOKBACK_WEEKS))]
    same_weekday = recent[recent.index.dayofweek == day.dayofweek]
    for candidate in (same_weekday, recent, history):
        if not candidate.empty:
            return float(candidate.median())
    return 0.0


def repair_sales_frame(
    df: pd.DataFrame, config: AppConfig | None = None
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Make an imperfect sales frame usable, and report every change.

    Rows with a missing or negative quantity are dropped, repeated ``(date, sku)`` rows are
    reduced to the first, and calendar days missing inside a product's own date range are
    inserted with an estimated quantity. The result is sorted by ``(sku, date)`` and carries
    an ``imputed`` flag; it is intended to satisfy
    :func:`~src.data.validation.validate_sales_frame`.

    Args:
        df: A tidy sales frame, as read from the feed.
        config: Application configuration; loaded from disk when omitted.

    Returns:
        The repaired frame and a :class:`DataQualityReport`.

    Raises:
        ValueError: If the frame lacks ``date``, ``sku`` or ``units_sold``, which cannot be
            repaired because there is nothing to repair them from.
    """
    config = config or get_config()
    essential = ["date", "sku", "units_sold"]
    missing = [c for c in essential if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot repair a sales frame without column(s): {missing}")

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    affected: set[str] = set()

    quantities = pd.to_numeric(frame["units_sold"], errors="coerce")
    invalid = quantities.isna() | (quantities < 0)
    if invalid.any():
        affected.update(frame.loc[invalid, "sku"].astype(str))
        frame = frame.loc[~invalid]
    frame["units_sold"] = pd.to_numeric(frame["units_sold"])

    frame = frame.sort_values(["sku", "date"], kind="stable")
    duplicated = frame.duplicated(subset=["date", "sku"], keep="first")
    if duplicated.any():
        affected.update(frame.loc[duplicated, "sku"].astype(str))
        frame = frame.loc[~duplicated]

    per_sku_columns = [c for c in ("category", "shelf_life_days") if c in frame.columns]
    shared_columns = [c for c in frame.columns if c not in {"date", "sku", "units_sold", IMPUTED_COLUMN, *per_sku_columns}]
    calendar = _calendar_by_date(frame, shared_columns)

    if IMPUTED_COLUMN not in frame.columns:
        frame[IMPUTED_COLUMN] = False
    frame[IMPUTED_COLUMN] = frame[IMPUTED_COLUMN].astype(bool)

    filled_rows: list[dict] = []
    without_calendar = 0
    for sku, group in frame.groupby("sku", observed=True):
        observed = group.set_index("date")["units_sold"].astype(float).sort_index()
        full = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
        gaps = full.difference(observed.index)
        if gaps.empty:
            continue
        affected.add(str(sku))
        constants = {c: group.iloc[0][c] for c in per_sku_columns}
        for day in gaps:
            row = {"date": day, "sku": sku, "units_sold": _fill_value(observed, day), IMPUTED_COLUMN: True}
            row.update(constants)
            if day in calendar.index:
                row.update(calendar.loc[day].to_dict())
            else:
                without_calendar += 1
                if "dow" in shared_columns:
                    row["dow"] = day.dayofweek
                if "weekend" in shared_columns:
                    row["weekend"] = int(day.dayofweek >= 5)
            filled_rows.append(row)

    if filled_rows:
        frame = pd.concat([frame, pd.DataFrame(filled_rows)], ignore_index=True)

    for column, dtype in df.dtypes.items():
        if column in frame.columns and column != IMPUTED_COLUMN:
            frame[column] = frame[column].astype(dtype)

    frame = frame.sort_values(["sku", "date"], kind="stable").reset_index(drop=True)
    report = DataQualityReport(
        invalid_rows_dropped=int(invalid.sum()),
        duplicate_rows_dropped=int(duplicated.sum()),
        missing_days_filled=len(filled_rows),
        days_without_calendar=without_calendar,
        skus_affected=tuple(sorted(affected)),
    )
    if not report.clean:
        logger.warning("%s", report.summary())
    return frame, report
