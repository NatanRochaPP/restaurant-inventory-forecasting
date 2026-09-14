"""SQLite storage layer.

SQLite is used deliberately: the project is a single-site prototype that must run on a
student laptop with no server to install, and the whole database is a single file that
can be shipped alongside the dissertation as reproducible evidence. Nothing in the schema
depends on SQLite specifics, so a move to PostgreSQL would be mechanical - but there is
no workload here that justifies it.

The schema records what was decided and why, not just the numbers: every forecast carries
the model that produced it, every recommendation carries its inputs, and manager overrides
are stored separately from the recommendation so that the model's output is never
retrospectively rewritten.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config import AppConfig
from src.forecasting.base import ForecastResult
from src.forecasting.model_selection import ModelChoice
from src.inventory.ordering import OrderRecommendation

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS skus (
    sku                           TEXT PRIMARY KEY,
    category                      TEXT NOT NULL,
    shelf_life_days               INTEGER NOT NULL,
    unit_cost                     REAL NOT NULL,
    pack_size                     INTEGER NOT NULL,
    min_order_quantity            INTEGER NOT NULL,
    holding_cost_per_unit_per_day REAL NOT NULL,
    waste_cost_per_unit           REAL NOT NULL,
    stockout_cost_per_unit        REAL NOT NULL,
    abc                           TEXT,
    xyz                           TEXT,
    segment                       TEXT
);

CREATE TABLE IF NOT EXISTS sales (
    date           TEXT NOT NULL,
    sku            TEXT NOT NULL,
    units_sold     INTEGER NOT NULL,
    temp_c         REAL,
    bank_holiday   INTEGER,
    school_holiday INTEGER,
    weekend        INTEGER,
    dow            INTEGER,
    promo          INTEGER,
    PRIMARY KEY (date, sku),
    FOREIGN KEY (sku) REFERENCES skus (sku)
);

CREATE TABLE IF NOT EXISTS inventory_snapshots (
    date            TEXT NOT NULL,
    sku             TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    opening_stock   REAL NOT NULL,
    received        REAL NOT NULL,
    expired_units   REAL NOT NULL,
    demand          REAL NOT NULL,
    served          REAL NOT NULL,
    stockout_units  REAL NOT NULL,
    closing_stock   REAL NOT NULL,
    on_order        REAL NOT NULL,
    inventory_value REAL NOT NULL,
    mean_age_days   REAL,
    PRIMARY KEY (date, sku, strategy)
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id           TEXT NOT NULL,
    as_of            TEXT NOT NULL,
    sku              TEXT NOT NULL,
    forecast_date    TEXT NOT NULL,
    predicted_demand REAL NOT NULL,
    lower_bound      REAL,
    upper_bound      REAL,
    model            TEXT NOT NULL,
    PRIMARY KEY (run_id, sku, forecast_date)
);

CREATE TABLE IF NOT EXISTS recommendations (
    run_id                     TEXT NOT NULL,
    as_of                      TEXT NOT NULL,
    sku                        TEXT NOT NULL,
    model                      TEXT NOT NULL,
    policy                     TEXT NOT NULL,
    on_hand                    REAL NOT NULL,
    on_order                   REAL NOT NULL,
    inventory_position         REAL NOT NULL,
    forecast_demand_horizon    REAL NOT NULL,
    forecast_demand_protection REAL NOT NULL,
    safety_stock               REAL NOT NULL,
    target_stock               REAL NOT NULL,
    recommended_order          REAL NOT NULL,
    order_value                REAL NOT NULL,
    days_of_cover              REAL,
    stockout_probability       REAL,
    projected_waste_units      REAL,
    risk                       TEXT,
    PRIMARY KEY (run_id, sku, as_of)
);

-- Overrides are stored separately so a manager's decision never rewrites the model's
-- recommendation. Both are retained for the audit trail.
CREATE TABLE IF NOT EXISTS recommendation_overrides (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT NOT NULL,
    as_of              TEXT NOT NULL,
    sku                TEXT NOT NULL,
    recommended_order  REAL NOT NULL,
    override_order     REAL NOT NULL,
    reason             TEXT,
    recorded_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulated_orders (
    run_id      TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    sku         TEXT NOT NULL,
    ordered_on  TEXT NOT NULL,
    arrives_on  TEXT NOT NULL,
    quantity    REAL NOT NULL,
    order_value REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id      TEXT PRIMARY KEY,
    strategy    TEXT NOT NULL,
    start_date  TEXT NOT NULL,
    end_date    TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulation_results (
    run_id   TEXT NOT NULL,
    strategy TEXT NOT NULL,
    metric   TEXT NOT NULL,
    value    REAL,
    PRIMARY KEY (run_id, metric)
);

CREATE TABLE IF NOT EXISTS model_metadata (
    sku              TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    reason           TEXT NOT NULL,
    segment          TEXT,
    strategy         TEXT NOT NULL,
    selected_at      TEXT NOT NULL,
    candidate_scores TEXT,
    PRIMARY KEY (sku, selected_at)
);

CREATE INDEX IF NOT EXISTS idx_sales_sku_date ON sales (sku, date);
CREATE INDEX IF NOT EXISTS idx_forecasts_sku ON forecasts (sku, forecast_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_strategy ON inventory_snapshots (strategy, date);
"""


class Database:
    """Thin repository over a SQLite file.

    Args:
        path: Path to the database file. Parent directories are created on demand.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: AppConfig) -> "Database":
        """Build a database handle from ``data.database`` in the configuration."""
        return cls(config.path(config.data.database))

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with foreign keys enabled, committing on success."""
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialise(self) -> None:
        """Create every table and index if they do not already exist."""
        with self.connect() as connection:
            connection.executescript(SCHEMA)
        logger.info("Initialised database schema at %s", self.path)

    def tables(self) -> list[str]:
        """Names of the tables present in the database."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    # -- writes ------------------------------------------------------------------------

    def save_sku_master(self, master: pd.DataFrame, segments: pd.DataFrame | None = None) -> int:
        """Upsert the SKU master, optionally attaching ABC/XYZ segmentation."""
        frame = master.copy()
        if segments is not None and not segments.empty:
            frame = frame.merge(segments[["sku", "abc", "xyz", "segment"]], on="sku", how="left")
        for column in ("abc", "xyz", "segment"):
            if column not in frame.columns:
                frame[column] = None
        columns = [
            "sku", "category", "shelf_life_days", "unit_cost", "pack_size", "min_order_quantity",
            "holding_cost_per_unit_per_day", "waste_cost_per_unit", "stockout_cost_per_unit",
            "abc", "xyz", "segment",
        ]
        return self._replace_rows("skus", frame[columns])

    def save_sales(self, sales: pd.DataFrame) -> int:
        """Store the sales history."""
        frame = sales.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        columns = [
            "date", "sku", "units_sold", "temp_c", "bank_holiday",
            "school_holiday", "weekend", "dow", "promo",
        ]
        return self._replace_rows("sales", frame[[c for c in columns if c in frame.columns]])

    def save_forecasts(self, results: list[ForecastResult], run_id: str, as_of: pd.Timestamp) -> int:
        """Store a set of forecasts under one run identifier."""
        if not results:
            return 0
        frame = pd.concat([r.to_frame() for r in results], ignore_index=True)
        frame["run_id"] = run_id
        frame["as_of"] = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        frame["forecast_date"] = pd.to_datetime(frame["forecast_date"]).dt.strftime("%Y-%m-%d")
        columns = [
            "run_id", "as_of", "sku", "forecast_date",
            "predicted_demand", "lower_bound", "upper_bound", "model",
        ]
        return self._replace_rows("forecasts", frame[columns])

    def save_recommendations(self, recommendations: list[OrderRecommendation], run_id: str) -> int:
        """Store order recommendations with the inputs that produced them."""
        if not recommendations:
            return 0
        frame = pd.DataFrame([r.to_row() for r in recommendations])
        frame["run_id"] = run_id
        frame["as_of"] = pd.to_datetime(frame["as_of"]).dt.strftime("%Y-%m-%d")
        frame = frame.rename(columns={"model": "model", "policy": "policy"})
        columns = [
            "run_id", "as_of", "sku", "model", "policy", "on_hand", "on_order",
            "inventory_position", "forecast_demand_horizon", "forecast_demand_protection",
            "safety_stock", "target_stock", "recommended_order", "order_value",
            "days_of_cover", "stockout_probability", "projected_waste_units", "risk",
        ]
        return self._replace_rows("recommendations", frame[columns])

    def record_override(
        self,
        run_id: str,
        as_of: pd.Timestamp,
        sku: str,
        recommended_order: float,
        override_order: float,
        reason: str = "",
    ) -> None:
        """Record a manager's override.

        The original recommendation is left untouched: an override is an operational
        decision, not a correction to the model, and the trained model is never modified.
        """
        if override_order < 0:
            raise ValueError(f"Override quantity for '{sku}' cannot be negative, got {override_order}.")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO recommendation_overrides
                    (run_id, as_of, sku, recommended_order, override_order, reason, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                    sku,
                    float(recommended_order),
                    float(override_order),
                    reason,
                    pd.Timestamp.now().isoformat(timespec="seconds"),
                ),
            )
        logger.info("Recorded override for %s: %.0f -> %.0f", sku, recommended_order, override_order)

    def save_simulation(
        self,
        run_id: str,
        strategy: str,
        daily: pd.DataFrame,
        orders: pd.DataFrame,
        kpis: dict[str, float],
        start: pd.Timestamp,
        end: pd.Timestamp,
        config: AppConfig | None = None,
    ) -> None:
        """Persist a complete simulation run: snapshots, orders and KPIs."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO simulation_runs
                    (run_id, strategy, start_date, end_date, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    strategy,
                    pd.Timestamp(start).strftime("%Y-%m-%d"),
                    pd.Timestamp(end).strftime("%Y-%m-%d"),
                    json.dumps(config.model_dump(mode="json") if config else {}),
                    pd.Timestamp.now().isoformat(timespec="seconds"),
                ),
            )

        snapshots = daily.copy()
        snapshots["date"] = pd.to_datetime(snapshots["date"]).dt.strftime("%Y-%m-%d")
        snapshot_columns = [
            "date", "sku", "strategy", "opening_stock", "received", "expired_units",
            "demand", "served", "stockout_units", "closing_stock", "on_order",
            "inventory_value", "mean_age_days",
        ]
        self._replace_rows("inventory_snapshots", snapshots[snapshot_columns])

        if not orders.empty:
            order_frame = orders.copy()
            order_frame["run_id"] = run_id
            order_frame["strategy"] = strategy
            for column in ("ordered_on", "arrives_on"):
                order_frame[column] = pd.to_datetime(order_frame[column]).dt.strftime("%Y-%m-%d")
            self._append_rows(
                "simulated_orders",
                order_frame[["run_id", "strategy", "sku", "ordered_on", "arrives_on", "quantity", "order_value"]],
            )

        kpi_frame = pd.DataFrame(
            [{"run_id": run_id, "strategy": strategy, "metric": k, "value": float(v)} for k, v in kpis.items()]
        )
        self._replace_rows("simulation_results", kpi_frame)
        logger.info("Saved simulation run '%s' (%s)", run_id, strategy)

    def save_model_metadata(self, choices: dict[str, ModelChoice]) -> int:
        """Store which model was selected for each SKU and the recorded reason."""
        frame = pd.DataFrame(
            [
                {
                    "sku": choice.sku,
                    "model_name": choice.model_name,
                    "reason": choice.reason,
                    "segment": choice.segment,
                    "strategy": choice.strategy,
                    "selected_at": choice.selected_at,
                    "candidate_scores": json.dumps(choice.candidate_scores),
                }
                for choice in choices.values()
            ]
        )
        return self._replace_rows("model_metadata", frame)

    # -- reads -------------------------------------------------------------------------

    def read(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Run a read-only query and return the result as a frame."""
        with self.connect() as connection:
            return pd.read_sql_query(query, connection, params=params)

    def load_table(self, table: str) -> pd.DataFrame:
        """Read an entire table.

        Raises:
            ValueError: If the table does not exist, listing the tables that do.
        """
        if table not in self.tables():
            raise ValueError(f"Table '{table}' does not exist. Available tables: {self.tables()}")
        return self.read(f"SELECT * FROM {table}")  # noqa: S608 - name validated above

    def load_overrides(self, run_id: str | None = None) -> pd.DataFrame:
        """Read recorded overrides, optionally for one run."""
        if run_id is None:
            return self.read("SELECT * FROM recommendation_overrides ORDER BY recorded_at DESC")
        return self.read(
            "SELECT * FROM recommendation_overrides WHERE run_id = ? ORDER BY recorded_at DESC",
            (run_id,),
        )

    def load_simulation_comparison(self) -> pd.DataFrame:
        """KPIs for every stored simulation run, one row per strategy and metric."""
        return self.read(
            """
            SELECT r.run_id, r.strategy, r.start_date, r.end_date, s.metric, s.value
            FROM simulation_runs r
            JOIN simulation_results s ON s.run_id = r.run_id
            ORDER BY r.created_at DESC, s.metric
            """
        )

    # -- internals ---------------------------------------------------------------------

    def _replace_rows(self, table: str, frame: pd.DataFrame) -> int:
        """Insert rows, replacing any that collide on the primary key."""
        return self._write(table, frame, "replace")

    def _append_rows(self, table: str, frame: pd.DataFrame) -> int:
        return self._write(table, frame, "append")

    def _write(self, table: str, frame: pd.DataFrame, mode: str) -> int:
        if frame.empty:
            return 0
        placeholders = ", ".join("?" for _ in frame.columns)
        columns = ", ".join(frame.columns)
        verb = "INSERT OR REPLACE" if mode == "replace" else "INSERT"
        sql = f"{verb} INTO {table} ({columns}) VALUES ({placeholders})"  # noqa: S608
        records = [tuple(None if pd.isna(v) else v for v in row) for row in frame.itertuples(index=False)]
        with self.connect() as connection:
            connection.executemany(sql, records)
        return len(records)
