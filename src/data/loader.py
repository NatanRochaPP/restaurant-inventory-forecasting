"""Loading of the sales dataset and the derived SKU master."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import AppConfig, get_config
from src.data.repair import repair_sales_frame
from src.data.validation import DataValidationError, validate_sales_frame

logger = logging.getLogger(__name__)

SALES_DTYPES = {
    "sku": "string",
    "category": "string",
    "shelf_life_days": "int16",
    "units_sold": "int32",
    "temp_c": "float32",
    "bank_holiday": "int8",
    "school_holiday": "int8",
    "weekend": "int8",
    "dow": "int8",
    "promo": "int8",
}


def load_sales(
    path: str | Path | None = None,
    config: AppConfig | None = None,
    *,
    validate: bool = True,
    repair: bool = False,
) -> pd.DataFrame:
    """Load the tidy daily sales dataset.

    Args:
        path: Optional CSV path. Defaults to ``data.sales_csv`` from the configuration.
        config: Application configuration; loaded from disk when omitted.
        validate: Run :func:`~src.data.validation.validate_sales_frame` after loading.
        repair: Repair an imperfect feed before validating (see
            :func:`~src.data.repair.repair_sales_frame`) and log what was changed. Off by
            default, so the project's own dataset is still rejected if it breaks an
            assumption rather than being quietly patched.

    Returns:
        A frame sorted by (sku, date) with a datetime ``date`` column.

    Raises:
        FileNotFoundError: If the CSV cannot be found.
        DataValidationError: If the contents fail validation.
    """
    config = config or get_config()
    csv_path = Path(path) if path is not None else config.path(config.data.sales_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sales data not found at {csv_path}. Run 'python baseline_forecasting.py' "
            "to regenerate the synthetic dataset (seed 42)."
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])
    for col, dtype in SALES_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    df = df.sort_values(["sku", "date"], kind="stable").reset_index(drop=True)

    if repair:
        df, report = repair_sales_frame(df, config)
        logger.info("%s", report.summary())
    if validate:
        validate_sales_frame(df, config)
    logger.info(
        "Loaded %d sales rows for %d SKUs (%s to %s)",
        len(df), df["sku"].nunique(), df["date"].min().date(), df["date"].max().date(),
    )
    return df


@lru_cache(maxsize=2)
def _cached_sales(csv_path: str) -> pd.DataFrame:
    return load_sales(csv_path)


def load_sales_cached(config: AppConfig | None = None) -> pd.DataFrame:
    """Process-cached variant of :func:`load_sales` for interactive use (Streamlit)."""
    config = config or get_config()
    return _cached_sales(str(config.path(config.data.sales_csv))).copy()


def build_sku_master(sales: pd.DataFrame, config: AppConfig | None = None) -> pd.DataFrame:
    """Build the SKU master table by merging dataset facts with configured economics.

    Args:
        sales: Tidy sales frame.
        config: Application configuration supplying per-SKU costs and pack sizes.

    Returns:
        One row per SKU with category, shelf life, unit cost, pack size, minimum order
        quantity and the derived holding, waste and stockout unit costs.
    """
    config = config or get_config()
    base = (
        sales.groupby("sku", observed=True)
        .agg(
            category=("category", "first"),
            shelf_life_days=("shelf_life_days", "first"),
            mean_daily_units=("units_sold", "mean"),
            zero_day_share=("units_sold", lambda s: float((s == 0).mean())),
        )
        .reset_index()
    )

    records = []
    for row in base.itertuples(index=False):
        econ = config.sku_economics(str(row.sku), int(row.shelf_life_days))
        records.append(
            {
                "sku": str(row.sku),
                "category": str(row.category),
                "shelf_life_days": econ.shelf_life_days,
                "mean_daily_units": float(row.mean_daily_units),
                "zero_day_share": float(row.zero_day_share),
                "unit_cost": econ.unit_cost,
                "pack_size": econ.pack_size,
                "min_order_quantity": econ.min_order_quantity,
                "holding_cost_per_unit_per_day": econ.holding_cost_per_unit_per_day,
                "waste_cost_per_unit": econ.waste_cost_per_unit,
                "stockout_cost_per_unit": econ.stockout_cost_per_unit,
            }
        )
    master = pd.DataFrame.from_records(records).sort_values("sku").reset_index(drop=True)

    unconfigured = [s for s in master["sku"] if s not in config.skus]
    if unconfigured:
        logger.warning(
            "No per-SKU economics configured for %s; falling back to sku_defaults "
            "(unit_cost=%.2f). Add them to config.yaml before citing cost results.",
            unconfigured, config.sku_defaults.unit_cost,
        )
    return master


def load_sku_list(sales: pd.DataFrame) -> list[str]:
    """Return the sorted list of SKU names present in the data."""
    if "sku" not in sales.columns:
        raise DataValidationError("Frame has no 'sku' column.")
    return sorted(str(s) for s in sales["sku"].unique())
