"""Demand-history features (lags and rolling statistics) built without leakage.

Two entry points exist and they must stay consistent:

``engineer_features``
    Adds lag and rolling columns to a historical panel for model *training*. Every
    statistic is computed within a SKU and shifted so that row ``d`` only ever sees
    demand from days strictly before ``d``.

``build_future_features``
    Builds the input rows for *prediction* over a future window. Rolling statistics are
    frozen at the forecast origin and lags fall back to the most recent comparable
    observation, so no value from inside the forecast window can influence its own
    prediction.

The AE1 baseline script computed ``roll28`` with ``groupby(...).shift(1).rolling(28)``,
which rolled across SKU boundaries and then re-aligned onto the wrong rows via
``reset_index(drop=True)``. ``engineer_features`` uses ``groupby.transform`` instead;
the two differ on 87% of rows (correlation 0.19), so results produced here are not
directly comparable with the AE1 table without re-running the backtest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.features.calendar import calendar_features
from src.features.weather import WeatherProvider

logger = logging.getLogger(__name__)

BASE_FEATURES: tuple[str, ...] = (
    "dow",
    "weekend",
    "bank_holiday",
    "school_holiday",
    "is_closed",
    "promo",
    "temp_c",
    "week_of_year",
    "month",
)

#: Human-readable labels used by the explainability layer.
FEATURE_LABELS: dict[str, str] = {
    "dow": "Day of week",
    "weekend": "Weekend (Fri-Sun)",
    "bank_holiday": "Bank holiday",
    "school_holiday": "School holiday",
    "is_closed": "Site closed",
    "promo": "Promotion running",
    "temp_c": "Temperature",
    "week_of_year": "Week of year",
    "month": "Month",
    "sku_code": "Product identity",
    "lag7": "Demand 7 days ago",
    "lag14": "Demand 14 days ago",
    "roll28": "28-day average demand",
}


@dataclass(frozen=True)
class SkuEncoder:
    """Stable integer encoding of SKU names for the global gradient-boosting model."""

    mapping: dict[str, int]

    @classmethod
    def fit(cls, skus: list[str]) -> "SkuEncoder":
        """Build an encoder from a SKU list; ordering is sorted for reproducibility."""
        return cls(mapping={s: i for i, s in enumerate(sorted(set(skus)))})

    def transform(self, skus: pd.Series) -> pd.Series:
        """Map SKU names to codes, raising on anything unseen."""
        unknown = set(skus.unique()) - set(self.mapping)
        if unknown:
            raise ValueError(
                f"SKU(s) {sorted(unknown)} were not present when the encoder was fitted. "
                "Refit the model after adding a new product."
            )
        return skus.map(self.mapping).astype("int16")

    @property
    def skus(self) -> list[str]:
        return sorted(self.mapping, key=lambda s: self.mapping[s])


def feature_columns(config: AppConfig) -> list[str]:
    """Return the ordered feature-matrix column names implied by the configuration."""
    lags = [f"lag{k}" for k in config.features.lags]
    rolls = [f"roll{w}" for w in config.features.rolling_windows]
    return [*BASE_FEATURES, *lags, *rolls, "sku_code"]


def lag_columns(config: AppConfig) -> list[str]:
    return [f"lag{k}" for k in config.features.lags]


def rolling_columns(config: AppConfig) -> list[str]:
    return [f"roll{w}" for w in config.features.rolling_windows]


def engineer_features(
    sales: pd.DataFrame,
    config: AppConfig,
    encoder: SkuEncoder | None = None,
) -> pd.DataFrame:
    """Add calendar, lag and rolling features to a historical sales panel.

    Args:
        sales: Tidy frame with at least ``date``, ``sku`` and ``units_sold``.
        config: Application configuration supplying the lag and window definitions.
        encoder: Optional SKU encoder; fitted from the data when omitted.

    Returns:
        A copy of ``sales`` with calendar columns refreshed, one column per configured
        lag and rolling window, and a ``sku_code`` column. Rows whose lag or rolling
        features are not yet defined keep NaN values so callers can drop them
        explicitly rather than training on silently imputed history.
    """
    required = {"date", "sku", "units_sold"}
    missing = required - set(sales.columns)
    if missing:
        raise ValueError(f"engineer_features requires columns {sorted(missing)}.")

    df = sales.sort_values(["sku", "date"], kind="stable").reset_index(drop=True)

    cal = calendar_features(df["date"])
    for col in ("dow", "weekend", "bank_holiday", "school_holiday", "is_closed", "week_of_year", "month"):
        df[col] = cal[col].to_numpy()

    if "promo" not in df.columns:
        df["promo"] = 0
    if "temp_c" not in df.columns:
        raise ValueError("engineer_features requires a 'temp_c' column.")

    grouped = df.groupby("sku", observed=True)["units_sold"]
    for k in config.features.lags:
        df[f"lag{k}"] = grouped.shift(k).astype("float32")
    for w in config.features.rolling_windows:
        # shift(1) first so the window ends on the previous day, then roll *within* the
        # SKU via transform - the point the AE1 script got wrong.
        df[f"roll{w}"] = grouped.transform(
            lambda s, _w=w: s.shift(1).rolling(_w, min_periods=_w).mean()
        ).astype("float32")

    encoder = encoder or SkuEncoder.fit([str(s) for s in df["sku"].unique()])
    df["sku_code"] = encoder.transform(df["sku"].astype("string"))
    return df


def prepare_feature_matrix(
    frame: pd.DataFrame,
    config: AppConfig,
    *,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split an engineered frame into (X, y).

    Args:
        frame: Output of :func:`engineer_features` or :func:`build_future_features`.
        config: Application configuration defining the feature column order.
        dropna: Drop rows with undefined lag/rolling features (training behaviour).

    Returns:
        ``(X, y)`` where ``y`` is ``None`` when the frame carries no ``units_sold``.
    """
    cols = feature_columns(config)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing column(s) {missing}. Run engineer_features first.")

    work = frame
    if dropna:
        work = work.dropna(subset=cols)
    X = work[cols].astype("float64")
    y = work["units_sold"].astype("float64") if "units_sold" in work.columns else None
    return X, y


def _lag_value(series: pd.Series, target: pd.Timestamp, lag: int) -> float:
    """Look up demand ``lag`` days before ``target`` from observed history.

    Falls back to the most recent same-weekday observation, then to the last observed
    value, so a prediction row is never built from a value inside the forecast window.
    """
    wanted = target - pd.Timedelta(days=lag)
    if wanted in series.index:
        return float(series.loc[wanted])
    if series.empty:
        return float("nan")
    same_dow = series[series.index.dayofweek == wanted.dayofweek]
    if not same_dow.empty:
        return float(same_dow.iloc[-1])
    return float(series.iloc[-1])


def build_future_features(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    *,
    as_of: pd.Timestamp,
    config: AppConfig,
    weather: WeatherProvider,
    encoder: SkuEncoder,
    skus: list[str] | None = None,
    promo_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build prediction rows for a future window without using future outcomes.

    Args:
        history: Observed sales strictly before ``as_of`` (use
            :func:`~src.data.preprocessing.history_as_of`).
        future_dates: The dates to predict.
        as_of: The decision date.
        config: Application configuration.
        weather: Provider supplying the temperature a manager would have on ``as_of``.
        encoder: SKU encoder used when the model was trained.
        skus: SKUs to build rows for; defaults to every SKU in ``history``.
        promo_calendar: Optional tidy frame of planned promotions with ``date``, ``sku``
            and ``promo``. Ignored when ``features.assume_promotions_known`` is False.

    Returns:
        One row per (SKU, date) carrying every model feature and no ``units_sold``.

    Raises:
        ValueError: If ``history`` contains observations on or after ``as_of``.
    """
    as_of = pd.Timestamp(as_of)
    future_dates = pd.DatetimeIndex(future_dates)
    if not history.empty and history["date"].max() >= as_of:
        raise ValueError(
            f"History passed to build_future_features contains dates >= as_of ({as_of.date()}). "
            "This would leak information that is not available at decision time."
        )

    skus = skus or sorted(str(s) for s in history["sku"].unique())
    cal = calendar_features(future_dates).set_index("date")
    temps = weather.temperature_forecast(as_of, future_dates)

    promo_map: dict[tuple[str, pd.Timestamp], int] = {}
    if config.features.assume_promotions_known and promo_calendar is not None:
        sub = promo_calendar[promo_calendar["date"].isin(future_dates)]
        promo_map = {
            (str(r.sku), pd.Timestamp(r.date)): int(r.promo) for r in sub.itertuples(index=False)
        }

    rolling_windows = config.features.rolling_windows
    rows: list[dict] = []
    hist_by_sku = {
        str(sku): grp.set_index("date")["units_sold"].astype(float).sort_index()
        for sku, grp in history.groupby("sku", observed=True)
    }

    for sku in skus:
        series = hist_by_sku.get(sku, pd.Series(dtype=float))
        # Rolling statistics are frozen at the origin: identical for every horizon day.
        frozen_rolls = {
            f"roll{w}": (float(series.iloc[-w:].mean()) if len(series) >= w else float("nan"))
            for w in rolling_windows
        }
        for d in future_dates:
            row: dict = {
                "date": d,
                "sku": sku,
                "temp_c": float(temps.loc[d]),
                "promo": promo_map.get((sku, d), 0),
            }
            row.update(
                {
                    "dow": int(cal.at[d, "dow"]),
                    "weekend": int(cal.at[d, "weekend"]),
                    "bank_holiday": int(cal.at[d, "bank_holiday"]),
                    "school_holiday": int(cal.at[d, "school_holiday"]),
                    "week_of_year": int(cal.at[d, "week_of_year"]),
                    "month": int(cal.at[d, "month"]),
                    "is_closed": int(cal.at[d, "is_closed"]),
                }
            )
            for k in config.features.lags:
                row[f"lag{k}"] = _lag_value(series, d, k)
            row.update(frozen_rolls)
            rows.append(row)

    out = pd.DataFrame.from_records(rows)
    out["sku_code"] = encoder.transform(out["sku"].astype("string"))
    return out.sort_values(["sku", "date"], kind="stable").reset_index(drop=True)


def demand_volatility(history: pd.Series, seasonal_period: int = 7) -> float:
    """Estimate daily demand standard deviation from history.

    Uses the seasonal difference so that a strong weekly pattern is not mistaken for
    random variability, which would inflate safety stock.
    """
    values = np.asarray(history, dtype=float)
    if len(values) <= seasonal_period:
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    diffs = values[seasonal_period:] - values[:-seasonal_period]
    # Var(X_t - X_{t-m}) = 2 * Var(X) for a stationary series.
    return float(np.std(diffs, ddof=1) / np.sqrt(2.0))
