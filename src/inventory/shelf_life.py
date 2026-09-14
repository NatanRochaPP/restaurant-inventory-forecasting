"""Perishability: batch tracking, expiry and shelf-life constraints.

Food inventory is not a single number. A kitchen holding 40 lettuces delivered today and
20 delivered four days ago is in a materially different position from one holding 60
delivered today, even though both report "60 on hand". :class:`InventoryLedger` therefore
tracks stock as dated batches, issues them first-expired-first-out, and writes off
whatever passes its use-by date.

Expired units are removed *before* demand is served on the same day, so expired stock can
never satisfy demand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Batch:
    """A dated quantity of stock.

    Attributes:
        received_on: Delivery date.
        expires_on: First date on which the batch is no longer usable.
        quantity: Units remaining in the batch.
        unit_cost: Purchase cost per unit, used for waste and holding valuation.
    """

    received_on: pd.Timestamp
    expires_on: pd.Timestamp
    quantity: float
    unit_cost: float

    def age_days(self, today: pd.Timestamp) -> int:
        """Days since delivery."""
        return int((pd.Timestamp(today) - self.received_on).days)

    def days_to_expiry(self, today: pd.Timestamp) -> int:
        """Days remaining before the batch expires (negative once expired)."""
        return int((self.expires_on - pd.Timestamp(today)).days)


class InventoryLedger:
    """First-expired-first-out inventory for a single perishable SKU.

    Args:
        sku: Product name, used in log messages and errors.
        shelf_life_days: Days a delivered unit remains usable.
        unit_cost: Default purchase cost per unit.
    """

    def __init__(self, sku: str, shelf_life_days: int, unit_cost: float) -> None:
        if shelf_life_days < 1:
            raise ValueError(f"Shelf life for '{sku}' must be at least 1 day, got {shelf_life_days}.")
        self.sku = sku
        self.shelf_life_days = int(shelf_life_days)
        self.unit_cost = float(unit_cost)
        self._batches: list[Batch] = []

    # -- state -------------------------------------------------------------------------

    @property
    def on_hand(self) -> float:
        """Total usable units currently held."""
        return float(sum(b.quantity for b in self._batches))

    @property
    def value(self) -> float:
        """Book value of stock on hand."""
        return float(sum(b.quantity * b.unit_cost for b in self._batches))

    @property
    def batches(self) -> list[Batch]:
        """The current batches, earliest expiry first."""
        return sorted(self._batches, key=lambda b: b.expires_on)

    def units_expiring_within(self, today: pd.Timestamp, days: int) -> float:
        """Units that will expire within ``days`` days - the near-term waste exposure."""
        cutoff = pd.Timestamp(today) + pd.Timedelta(days=days)
        return float(sum(b.quantity for b in self._batches if b.expires_on <= cutoff))

    def mean_age_days(self, today: pd.Timestamp) -> float:
        """Quantity-weighted mean age of stock on hand."""
        total = self.on_hand
        if total <= 0:
            return 0.0
        return float(sum(b.quantity * b.age_days(today) for b in self._batches) / total)

    # -- operations --------------------------------------------------------------------

    def receive(
        self,
        quantity: float,
        received_on: pd.Timestamp,
        shelf_life_days: int | None = None,
        unit_cost: float | None = None,
    ) -> Batch:
        """Add a delivered batch.

        Args:
            quantity: Units received; must be positive.
            received_on: Delivery date.
            shelf_life_days: Override the SKU's default shelf life for this batch.
            unit_cost: Override the default unit cost for this batch.

        Returns:
            The batch that was added.
        """
        if quantity <= 0:
            raise ValueError(f"Received quantity for '{self.sku}' must be positive, got {quantity}.")
        received_on = pd.Timestamp(received_on)
        life = int(shelf_life_days if shelf_life_days is not None else self.shelf_life_days)
        batch = Batch(
            received_on=received_on,
            expires_on=received_on + pd.Timedelta(days=life),
            quantity=float(quantity),
            unit_cost=float(unit_cost if unit_cost is not None else self.unit_cost),
        )
        self._batches.append(batch)
        return batch

    def remove_expired(self, today: pd.Timestamp) -> tuple[float, float]:
        """Write off every batch that has reached its expiry date.

        Must be called before demand is served, so that expired stock cannot be sold.

        Args:
            today: The current date. A batch expires when ``expires_on <= today``.

        Returns:
            ``(units_wasted, cost_of_wasted_stock)``.
        """
        today = pd.Timestamp(today)
        expired = [b for b in self._batches if b.expires_on <= today]
        if not expired:
            return 0.0, 0.0
        units = float(sum(b.quantity for b in expired))
        cost = float(sum(b.quantity * b.unit_cost for b in expired))
        self._batches = [b for b in self._batches if b.expires_on > today]
        logger.debug("%s: wrote off %.1f expired units on %s", self.sku, units, today.date())
        return units, cost

    def issue(self, quantity: float) -> tuple[float, float]:
        """Consume stock first-expired-first-out.

        Args:
            quantity: Units of demand to serve.

        Returns:
            ``(units_served, units_short)``. Unserved demand is lost, not backordered.
        """
        if quantity < 0:
            raise ValueError(f"Demand for '{self.sku}' cannot be negative, got {quantity}.")
        remaining = float(quantity)
        served = 0.0
        for batch in sorted(self._batches, key=lambda b: b.expires_on):
            if remaining <= 0:
                break
            take = min(batch.quantity, remaining)
            batch.quantity -= take
            remaining -= take
            served += take
        self._batches = [b for b in self._batches if b.quantity > 1e-9]
        return served, float(remaining)

    def snapshot(self, today: pd.Timestamp) -> dict[str, float]:
        """Summary of the current position for logging and dashboards."""
        return {
            "on_hand": self.on_hand,
            "value": self.value,
            "n_batches": float(len(self._batches)),
            "mean_age_days": self.mean_age_days(today),
            "expiring_within_2_days": self.units_expiring_within(today, 2),
        }


def usable_shelf_life_days(shelf_life_days: int, usable_fraction: float) -> float:
    """Portion of shelf life a restaurant will realistically sell through.

    Kitchens do not serve stock on its final day, so the configured
    ``inventory.shelf_life_usable_fraction`` discounts the nominal life.
    """
    if not 0 < usable_fraction <= 1:
        raise ValueError(f"Usable fraction must be in (0, 1], got {usable_fraction}.")
    return float(max(shelf_life_days * usable_fraction, 1.0))


def shelf_life_order_cap(
    daily_forecast,
    shelf_life_days: int,
    usable_fraction: float,
    inventory_position: float,
    lead_time_days: int = 0,
) -> float:
    """Largest order that can plausibly be consumed before it expires.

    A delivery arriving after ``lead_time_days`` must be sold within its usable shelf
    life. Anything ordered beyond the demand expected in that window is waste in
    waiting, so the recommendation is capped at

        expected demand over (lead time + usable shelf life) - inventory position

    Args:
        daily_forecast: Point forecast per day from the decision date.
        shelf_life_days: Nominal shelf life of the SKU.
        usable_fraction: Configured usable portion of that shelf life.
        inventory_position: Stock already on hand or on order.
        lead_time_days: Days until the order arrives.

    Returns:
        The cap in units, never negative.
    """
    from src.inventory.safety_stock import expected_demand_over

    usable = usable_shelf_life_days(shelf_life_days, usable_fraction)
    window = lead_time_days + usable
    demand = expected_demand_over(daily_forecast, window)
    return float(max(demand - inventory_position, 0.0))
