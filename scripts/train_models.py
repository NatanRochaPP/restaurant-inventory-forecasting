"""Select and persist the forecasting model for each SKU.

Training is separated from inference: this script decides which model each SKU should
use, records why, fits the global gradient-boosting model on all history up to the
selection date and writes both to ``models/``. The dashboard and the simulator then load
that registry instead of re-deciding at runtime.

Usage::

    python scripts/train_models.py                     # rule-based routing (fast)
    python scripts/train_models.py --strategy validated  # compare candidates per SKU
    python scripts/train_models.py --as-of 2025-07-01    # selection date
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.data.loader import build_sku_master, load_sales  # noqa: E402
from src.data.preprocessing import history_as_of  # noqa: E402
from src.features.demand_features import SkuEncoder  # noqa: E402
from src.forecasting.gradient_boosting import GradientBoostingForecast  # noqa: E402
from src.forecasting.model_selection import ModelSelector  # noqa: E402
from src.persistence.database import Database  # noqa: E402
from src.segmentation import segment_skus  # noqa: E402

logger = logging.getLogger("train_models")
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="Select and fit forecasting models per SKU.")
    parser.add_argument("--strategy", choices=["rule", "validated"], default="rule")
    parser.add_argument("--as-of", type=str, default=None, help="Selection date (YYYY-MM-DD).")
    parser.add_argument("--folds", type=int, default=3, help="Validation folds for --strategy validated.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-database", action="store_true", help="Skip writing to SQLite.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    sales = load_sales(config=config)
    master = build_sku_master(sales, config)
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(config.simulation.start_date)

    print(f"Selection date: {as_of.date()} (models see no data on or after this date)")

    segmentation = segment_skus(sales, master, config, as_of=as_of)
    print("\nABC/XYZ segmentation")
    print(
        segmentation[["sku", "abc", "xyz", "segment", "annual_value", "demand_cv", "zero_day_share"]]
        .round(3)
        .to_string(index=False)
    )

    selector = ModelSelector(config, strategy=args.strategy, validation_folds=args.folds)
    choices = selector.select(sales, master, as_of=as_of)
    table = ModelSelector.to_frame(choices)
    print(f"\nModel selection ({args.strategy})")
    print(table[["sku", "segment", "model_name"]].to_string(index=False))
    print("\nModels in use:", table["model_name"].value_counts().to_dict())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = MODELS_DIR / "model_selection.json"
    ModelSelector.save(choices, registry_path)

    # Fit and persist the global model so inference never retrains implicitly.
    history = history_as_of(sales, as_of)
    encoder = SkuEncoder.fit([str(s) for s in sales["sku"].unique()])
    model = GradientBoostingForecast(config, encoder).fit(history)
    model_path = MODELS_DIR / "gradient_boosting.joblib"
    joblib.dump(
        {
            "model": model,
            "as_of": str(as_of.date()),
            "n_train_rows": model.n_train_rows_,
            "random_seed": config.forecast.random_seed,
        },
        model_path,
    )

    segmentation.to_csv(PROJECT_ROOT / "outputs" / "sku_segmentation.csv", index=False)
    table.to_csv(PROJECT_ROOT / "outputs" / "model_selection.csv", index=False)

    if not args.no_database:
        database = Database.from_config(config)
        database.initialise()
        database.save_sku_master(master, segmentation)
        database.save_sales(sales)
        database.save_model_metadata(choices)
        print(f"\nDatabase updated: {database.path}")

    print(f"Saved registry to {registry_path}")
    print(f"Saved fitted global model to {model_path} ({model.n_train_rows_:,} training rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
