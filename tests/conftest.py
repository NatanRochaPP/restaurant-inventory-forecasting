"""Shared pytest fixtures.

Tests run against a small deterministic panel rather than the full 14,620-row dataset so
that the suite stays fast, plus a session-scoped slice of the real data for the
end-to-end test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig, load_config
from src.data.loader import load_sales

SEED = 42


@pytest.fixture(scope="session")
def config() -> AppConfig:
    """The real project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def sales(config: AppConfig) -> pd.DataFrame:
    """The full synthetic sales dataset."""
    return load_sales(config=config)


def make_synthetic_panel() -> pd.DataFrame:
    """Build the test panel.

    Exposed as a plain function as well as a fixture so that class-scoped fixtures (which
    cannot request function-scoped ones) can build their own copy.
    """
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    rows = []
    for sku, base, shelf, intermittent in [("Smooth SKU", 100.0, 5, False), ("Spiky SKU", 4.0, 3, True)]:
        for d in dates:
            weekend = 1 if d.dayofweek in (4, 5, 6) else 0
            mu = base * (1.5 if weekend else 1.0)
            units = rng.poisson(mu)
            if intermittent and rng.random() < 0.45:
                units = 0
            rows.append(
                {
                    "date": d,
                    "sku": sku,
                    "category": "Test",
                    "shelf_life_days": shelf,
                    "units_sold": int(units),
                    "temp_c": float(12 + 5 * np.sin(d.dayofyear / 58.0)),
                    "bank_holiday": 0,
                    "school_holiday": 0,
                    "weekend": weekend,
                    "dow": d.dayofweek,
                    "promo": 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A tiny two-SKU, 120-day panel with a known weekly pattern.

    'Smooth SKU' has a clean weekend uplift; 'Spiky SKU' is intermittent, so the two
    routing branches of the model selector are both exercised.
    """
    return make_synthetic_panel()


@pytest.fixture
def smooth_history(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """Single-SKU history for univariate model tests."""
    return synthetic_panel[synthetic_panel["sku"] == "Smooth SKU"].reset_index(drop=True)
