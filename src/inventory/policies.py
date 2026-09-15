"""Inventory policies.

The policy layer knows nothing about how the forecast was produced. It receives a daily
demand forecast, an uncertainty estimate and the current stock position, and returns the
levels that define a replenishment decision. That separation is what allows the
simulation to hold the policy fixed while swapping the forecast (AI versus manual), which
is the comparison the research question needs.

Two policies are supported:

Base stock (R, S)
    Review every ``R`` days and order up to ``S``. Simple, and the natural fit for a
    restaurant that orders on fixed delivery days.

(s, S)
    Review every ``R`` days but only order when the inventory position has fallen to the
    reorder point ``s``, then order up to ``S``. Raises fewer, larger orders, which
    matters when there is a fixed cost per delivery.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from src.inventory.safety_stock import expected_demand_over, safety_stock

logger = logging.getLogger(__name__)


def inventory_position(on_hand: float, on_order: float, backorders: float = 0.0) -> float:
    """Inventory position = on hand + on order - backorders.

    This, not on-hand stock, is what a replenishment decision must be based on:
    ignoring stock already in transit causes double ordering on every review day.

    Raises:
        ValueError: If any component is negative.
    """
    for label, value in (("on_hand", on_hand), ("on_order", on_order), ("backorders", backorders)):
        if value < 0:
            raise ValueError(f"{label} must be non-negative, got {value}.")
    return float(on_hand + on_order - backorders)


@dataclass(frozen=True)
class PolicyInputs:
    """Everything a policy needs to size a replenishment.

    Attributes:
        sku: Product name.
        daily_forecast: Expected demand per day from the decision date onwards.
        sigma_daily: One-day forecast error standard deviation.
        on_hand: Usable stock currently held.
        on_order: Stock ordered but not yet delivered.
        backorders: Unfulfilled demand carried forward (zero for lost-sales settings).
        lead_time_days: Days between placing and receiving an order.
        review_period_days: Days until the next ordering opportunity.
        service_level: Target cycle service level in (0, 1).
    """

    sku: str
    daily_forecast: np.ndarray
    sigma_daily: float
    on_hand: float
    on_order: float = 0.0
    backorders: float = 0.0
    lead_time_days: int = 2
    review_period_days: int = 1
    service_level: float = 0.95

    def __post_init__(self) -> None:
        forecast = np.asarray(self.daily_forecast, dtype=float)
        if forecast.size == 0:
            raise ValueError(f"No demand forecast supplied for '{self.sku}'; cannot size an order.")
        if np.any(forecast < 0):
            raise ValueError(f"Demand forecast for '{self.sku}' contains negative values.")
        if self.lead_time_days < 0:
            raise ValueError(f"Lead time must be non-negative, got {self.lead_time_days}.")
        if self.review_period_days < 1:
            raise ValueError(f"Review period must be at least 1 day, got {self.review_period_days}.")
        if not 0.0 < self.service_level < 1.0:
            raise ValueError(f"Service level must be in (0, 1), got {self.service_level}.")
        object.__setattr__(self, "daily_forecast", forecast)

    @property
    def protection_period_days(self) -> int:
        """Lead time plus review period: the exposure a delivery must cover."""
        return int(self.lead_time_days + self.review_period_days)

    @property
    def position(self) -> float:
        """Current inventory position."""
        return inventory_position(self.on_hand, self.on_order, self.backorders)

    @property
    def mean_daily_forecast(self) -> float:
        """Average forecast demand per day over the horizon."""
        return float(np.mean(self.daily_forecast))


@dataclass(frozen=True)
class PolicyTargets:
    """The levels a policy derives from its inputs."""

    policy_name: str
    protection_period_days: int
    expected_demand_protection: float
    safety_stock: float
    target_stock: float
    reorder_point: float | None = None
    should_order: bool = True
    detail: dict[str, float] = field(default_factory=dict)


class InventoryPolicy(ABC):
    """Base class for replenishment policies."""

    name: str = "policy"

    @abstractmethod
    def targets(self, inputs: PolicyInputs) -> PolicyTargets:
        """Compute the target stock level (and reorder point, if any)."""

    def raw_order_quantity(self, inputs: PolicyInputs) -> tuple[float, PolicyTargets]:
        """Unconstrained order quantity: ``max(0, target stock - inventory position)``.

        Constraints such as pack size, minimum order quantity and shelf life are applied
        afterwards by :func:`~src.inventory.ordering.recommend_order`.
        """
        targets = self.targets(inputs)
        if not targets.should_order:
            return 0.0, targets
        return float(max(targets.target_stock - inputs.position, 0.0)), targets


class BaseStockPolicy(InventoryPolicy):
    """Order-up-to policy: review every period and top up to ``S``."""

    name = "Base stock (R, S)"

    def targets(self, inputs: PolicyInputs) -> PolicyTargets:
        protection = inputs.protection_period_days
        demand = expected_demand_over(inputs.daily_forecast, protection)
        ss = safety_stock(inputs.sigma_daily, protection, inputs.service_level)
        return PolicyTargets(
            policy_name=self.name,
            protection_period_days=protection,
            expected_demand_protection=demand,
            safety_stock=ss,
            target_stock=demand + ss,
            reorder_point=None,
            should_order=True,
            detail={"lead_time_days": float(inputs.lead_time_days), "review_period_days": float(inputs.review_period_days)},
        )


class SsPolicy(InventoryPolicy):
    """(s, S) policy: order up to ``S`` only once the position falls to ``s``.

    ``s`` covers demand over the lead time alone (the exposure if an order is placed
    now), while ``S`` covers lead time plus review period. Between them sits the band in
    which no order is raised, which avoids paying a delivery charge for a trivial top-up.

    Args:
        reorder_multiplier: Scales the reorder point. Values above 1 trigger orders
            earlier (more cautious); below 1 defers them.
    """

    name = "(s, S)"

    def __init__(self, reorder_multiplier: float = 1.0) -> None:
        if reorder_multiplier <= 0:
            raise ValueError(f"Reorder multiplier must be positive, got {reorder_multiplier}.")
        self.reorder_multiplier = float(reorder_multiplier)

    def targets(self, inputs: PolicyInputs) -> PolicyTargets:
        protection = inputs.protection_period_days
        demand_full = expected_demand_over(inputs.daily_forecast, protection)
        ss_full = safety_stock(inputs.sigma_daily, protection, inputs.service_level)
        target = demand_full + ss_full

        demand_lead = expected_demand_over(inputs.daily_forecast, inputs.lead_time_days)
        ss_lead = safety_stock(inputs.sigma_daily, inputs.lead_time_days, inputs.service_level)
        reorder_point = (demand_lead + ss_lead) * self.reorder_multiplier

        return PolicyTargets(
            policy_name=self.name,
            protection_period_days=protection,
            expected_demand_protection=demand_full,
            safety_stock=ss_full,
            target_stock=target,
            reorder_point=reorder_point,
            should_order=inputs.position <= reorder_point,
            detail={
                "expected_demand_lead_time": demand_lead,
                "safety_stock_lead_time": ss_lead,
                "reorder_multiplier": self.reorder_multiplier,
            },
        )


class ParLevelPolicy(InventoryPolicy):
    """Par-level policy representing current manual practice.

    This is the baseline the research question compares against. A manager sets a par
    level from recent average usage - not a forecast - and tops up to it on each delivery
    day, adding a flat percentage buffer for comfort:

        par = average daily usage x cover days x (1 + buffer)

    It has no view of the coming weekend, bank holidays, weather or promotions, which is
    precisely the gap the forecasting system is meant to close. Because the buffer is a
    fixed percentage rather than a function of forecast error, it over-covers stable
    items and under-covers volatile ones.

    Args:
        buffer_pct: The manager's comfort margin, e.g. 0.20 for 20%.
        cover_days: Days of cover to hold. Defaults to the protection period, so the
            baseline and the AI policy target the same exposure window.
    """

    name = "Manual par level"

    def __init__(self, buffer_pct: float = 0.20, cover_days: int | None = None) -> None:
        if buffer_pct < 0:
            raise ValueError(f"Par-level buffer must be non-negative, got {buffer_pct}.")
        self.buffer_pct = float(buffer_pct)
        self.cover_days = cover_days

    def targets(self, inputs: PolicyInputs) -> PolicyTargets:
        cover = float(self.cover_days if self.cover_days is not None else inputs.protection_period_days)
        average_daily = inputs.mean_daily_forecast
        expected = average_daily * cover
        par = expected * (1.0 + self.buffer_pct)
        return PolicyTargets(
            policy_name=self.name,
            protection_period_days=int(inputs.protection_period_days),
            expected_demand_protection=expected,
            safety_stock=par - expected,
            target_stock=par,
            reorder_point=None,
            should_order=True,
            detail={"cover_days": cover, "buffer_pct": self.buffer_pct, "average_daily_usage": average_daily},
        )


def build_policy(name: str, reorder_multiplier: float = 1.0) -> InventoryPolicy:
    """Instantiate the policy named in the configuration.

    Args:
        name: ``"base_stock"`` or ``"s_S"``.
        reorder_multiplier: Passed to the (s, S) policy.

    Raises:
        ValueError: If the policy name is not recognised.
    """
    match name:
        case "base_stock":
            return BaseStockPolicy()
        case "s_S":
            return SsPolicy(reorder_multiplier)
    raise ValueError(f"Unknown inventory policy '{name}'. Use 'base_stock' or 's_S'.")
