"""Day-by-day perishability-aware inventory simulator.

The simulator replays a historical holdout period one day at a time. On each day, for
each SKU, it executes a fixed sequence:

1. receive deliveries whose lead time has elapsed;
2. write off stock that has passed its use-by date;
3. on a supplier order day, forecast demand and compute the inventory position;
4. produce an order recommendation and place it into the delivery pipeline;
5. serve that day's *actual* demand from usable stock, first-expired-first-out;
6. record stockouts, waste, closing stock, inventory value and costs.

The ordering decision at step 4 happens strictly before demand is revealed at step 5, and
every strategy reads history through :func:`~src.data.preprocessing.history_as_of`. That
ordering is the mechanism that prevents future demand influencing a past decision, and it
is asserted directly in ``tests/test_simulator.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.config import AppConfig
from src.data.loader import build_sku_master
from src.inventory.ordering import OrderRecommendation
from src.inventory.shelf_life import InventoryLedger
from src.inventory.strategies import OrderingStrategy

logger = logging.getLogger(__name__)


@dataclass
class PendingDelivery:
    """An order placed but not yet received."""

    sku: str
    ordered_on: pd.Timestamp
    arrives_on: pd.Timestamp
    quantity: float
    unit_cost: float


@dataclass
class SkuState:
    """Simulation state for one SKU."""

    ledger: InventoryLedger
    pipeline: list[PendingDelivery] = field(default_factory=list)

    @property
    def on_order(self) -> float:
        """Units ordered but not yet delivered."""
        return float(sum(d.quantity for d in self.pipeline))


@dataclass
class SimulationResult:
    """Raw output of a simulation run.

    Attributes:
        strategy_name: The ordering strategy that produced it.
        daily: One row per SKU-day with stock movements and costs.
        orders: One row per simulated order.
        recommendations: Full recommendation records, for explainability.
        start: First simulated date.
        end: Last simulated date.
    """

    strategy_name: str
    daily: pd.DataFrame
    orders: pd.DataFrame
    recommendations: list[OrderRecommendation]
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def n_days(self) -> int:
        return int(self.daily["date"].nunique()) if not self.daily.empty else 0


class InventorySimulator:
    """Replays a holdout period under a given ordering strategy.

    Args:
        sales: Full tidy sales history, providing the actual demand to serve.
        config: Application configuration.
        strategy: The ordering strategy under test.
        skus: SKUs to simulate; defaults to all.
        sku_master: Pre-built SKU master; built from ``sales`` when omitted.
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        config: AppConfig,
        strategy: OrderingStrategy,
        skus: list[str] | None = None,
        sku_master: pd.DataFrame | None = None,
    ) -> None:
        self.sales = sales
        self.config = config
        self.strategy = strategy
        self.skus = skus or sorted(str(s) for s in sales["sku"].unique())
        master = sku_master if sku_master is not None else build_sku_master(sales, config)
        self.economics = {
            str(row.sku): config.sku_economics(str(row.sku), int(row.shelf_life_days))
            for row in master.itertuples(index=False)
            if str(row.sku) in self.skus
        }
        missing = set(self.skus) - set(self.economics)
        if missing:
            raise ValueError(f"No economics available for SKU(s) {sorted(missing)}.")
        self._demand = self._demand_lookup(sales)

    @staticmethod
    def _demand_lookup(sales: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
        """Index actual demand by (SKU, date) for fast lookup during the run."""
        return {
            (str(row.sku), pd.Timestamp(row.date)): float(row.units_sold)
            for row in sales[["sku", "date", "units_sold"]].itertuples(index=False)
        }

    # -- setup -------------------------------------------------------------------------

    def _validate_window(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        data_start = pd.Timestamp(self.sales["date"].min())
        data_end = pd.Timestamp(self.sales["date"].max())
        if start > end:
            raise ValueError(f"Simulation start {start.date()} is after end {end.date()}.")
        if start <= data_start:
            raise ValueError(
                f"Simulation must start after the first observation ({data_start.date()}) so that "
                "models have history to train on."
            )
        if end > data_end:
            raise ValueError(
                f"Simulation end {end.date()} is beyond the last observation ({data_end.date()}); "
                "there is no actual demand to serve."
            )
        available = (start - data_start).days
        if available < self.config.simulation.warmup_days:
            logger.warning(
                "Only %d days of history precede the simulation start but warmup_days is %d. "
                "Early forecasts will be weaker than configured.",
                available, self.config.simulation.warmup_days,
            )

    def _initial_state(self, start: pd.Timestamp) -> dict[str, SkuState]:
        """Open every SKU with the configured days of cover, identical for all strategies."""
        states: dict[str, SkuState] = {}
        lookback = start - pd.Timedelta(days=self.config.baseline.lookback_days)
        history = self.sales[(self.sales["date"] < start) & (self.sales["date"] >= lookback)]
        means = history.groupby("sku", observed=True)["units_sold"].mean()
        for sku in self.skus:
            econ = self.economics[sku]
            ledger = InventoryLedger(sku, econ.shelf_life_days, econ.unit_cost)
            opening = float(means.get(sku, 0.0)) * self.config.simulation.initial_stock_days_of_supply
            if opening > 0:
                ledger.receive(opening, start - pd.Timedelta(days=1))
            states[sku] = SkuState(ledger=ledger)
        return states

    def _review_period_days(self, day: pd.Timestamp) -> int:
        """Days until the next ordering opportunity - the true review period."""
        from src.features.calendar import days_until_next_order

        return days_until_next_order(day, self.config.supplier.order_days)

    # -- run ---------------------------------------------------------------------------

    def run(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> SimulationResult:
        """Execute the simulation over the holdout window.

        Args:
            start: First simulated day; defaults to ``simulation.start_date``.
            end: Last simulated day; defaults to ``simulation.end_date``.

        Returns:
            A :class:`SimulationResult` with per-day records, orders and recommendations.
        """
        start = pd.Timestamp(start or self.config.simulation.start_date)
        end = pd.Timestamp(end or self.config.simulation.end_date)
        self._validate_window(start, end)

        states = self._initial_state(start)
        order_days = self.config.supplier.order_days
        lead_time = self.config.supplier.lead_time_days

        daily_rows: list[dict] = []
        order_rows: list[dict] = []
        recommendations: list[OrderRecommendation] = []

        for day in pd.date_range(start, end, freq="D"):
            is_review_day = day.dayofweek in order_days
            if is_review_day:
                # One model pass for all SKUs, using only data before today.
                self.strategy.prepare(day, self.skus)
            review_period = self._review_period_days(day) if is_review_day else 0
            ordered_today: list[tuple[str, float]] = []
            day_rows: dict[str, dict] = {}

            for sku in self.skus:
                state = states[sku]
                econ = self.economics[sku]
                opening_stock = state.ledger.on_hand

                # 1. Receive deliveries due today.
                arrivals = [d for d in state.pipeline if d.arrives_on == day]
                received = float(sum(d.quantity for d in arrivals))
                for delivery in arrivals:
                    state.ledger.receive(delivery.quantity, day, unit_cost=delivery.unit_cost)
                state.pipeline = [d for d in state.pipeline if d.arrives_on > day]

                # 2. Remove expired stock before any demand can be served from it.
                wasted_units, wasted_cost = state.ledger.remove_expired(day)

                # Stock the ordering decision can actually see: delivered and unexpired.
                available_stock = state.ledger.on_hand
                on_order = state.on_order

                # 3-4. Order recommendation on supplier order days only.
                recommended = 0.0
                if is_review_day:
                    recommendation = self.strategy.recommend(
                        sku=sku,
                        as_of=day,
                        on_hand=available_stock,
                        on_order=on_order,
                        economics=econ,
                        review_period_days=review_period,
                        lead_time_days=lead_time,
                    )
                    recommendations.append(recommendation)
                    recommended = recommendation.recommended_order_quantity
                    if recommended > 0:
                        arrives = day + pd.Timedelta(days=lead_time)
                        state.pipeline.append(
                            PendingDelivery(sku, day, arrives, recommended, econ.unit_cost)
                        )
                        ordered_today.append((sku, recommended))
                        order_rows.append(
                            {
                                "sku": sku,
                                "ordered_on": day,
                                "arrives_on": arrives,
                                "quantity": recommended,
                                "order_value": recommended * econ.unit_cost,
                                "strategy": self.strategy.name,
                            }
                        )

                # 5. Serve today's actual demand - revealed only now.
                demand = self._demand.get((sku, day), 0.0)
                served, short = state.ledger.issue(demand)

                closing_stock = state.ledger.on_hand
                day_rows[sku] = {
                    "date": day,
                    "sku": sku,
                    "strategy": self.strategy.name,
                    "opening_stock": opening_stock,
                    "received": received,
                    "expired_units": wasted_units,
                    "available_stock": available_stock,
                    "demand": demand,
                    "served": served,
                    "stockout_units": short,
                    "stockout_day": int(short > 0),
                    "closing_stock": closing_stock,
                    "on_order": state.on_order,
                    "inventory_value": state.ledger.value,
                    "mean_age_days": state.ledger.mean_age_days(day),
                    "recommended_order": recommended,
                    "is_review_day": int(is_review_day),
                    "holding_cost": closing_stock * econ.holding_cost_per_unit_per_day,
                    "waste_cost": wasted_cost + wasted_units * self.config.costs.waste_disposal_cost_per_unit,
                    "stockout_cost": short * econ.stockout_cost_per_unit,
                    "ordering_cost": 0.0,
                }

            # A delivery charge is incurred once per order day, shared across the SKUs on
            # that order so that per-SKU costs still sum to the true total.
            if ordered_today:
                share = self.config.supplier.ordering_cost_per_order / len(ordered_today)
                for sku, _ in ordered_today:
                    day_rows[sku]["ordering_cost"] = share

            daily_rows.extend(day_rows.values())

        daily = pd.DataFrame(daily_rows)
        daily["total_cost"] = (
            daily["holding_cost"] + daily["waste_cost"] + daily["stockout_cost"] + daily["ordering_cost"]
        )
        logger.info(
            "Simulated %s over %s to %s: %d SKU-days, %d orders",
            self.strategy.name, start.date(), end.date(), len(daily), len(order_rows),
        )
        return SimulationResult(
            strategy_name=self.strategy.name,
            daily=daily,
            orders=pd.DataFrame(order_rows),
            recommendations=recommendations,
            start=start,
            end=end,
        )
