"""Order recommendation: from policy target to a quantity a manager can actually order.

A raw ``target stock - inventory position`` figure is not an order. Suppliers sell in
cases, impose minimum quantities, and a fresh product ordered beyond what the kitchen can
sell before its use-by date is waste bought in advance. This module applies those
constraints in a fixed, auditable sequence and records every adjustment, so the dashboard
can explain exactly why the recommended number differs from the raw calculation.

The system is decision support: it produces a recommendation and never places an order.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.config import AppConfig, SkuEconomics
from src.inventory.policies import InventoryPolicy, PolicyInputs, build_policy
from src.inventory.safety_stock import expected_demand_over, stockout_probability
from src.inventory.shelf_life import shelf_life_order_cap, usable_shelf_life_days

logger = logging.getLogger(__name__)

RiskLevel = Literal["low", "medium", "high"]

#: Stockout probability above which a SKU is flagged to the manager.
STOCKOUT_RISK_THRESHOLDS: tuple[float, float] = (0.10, 0.25)
#: Share of projected stock expected to expire before use.
WASTE_RISK_THRESHOLDS: tuple[float, float] = (0.05, 0.15)


@dataclass(frozen=True)
class ConstraintAdjustment:
    """A single change applied to the raw order quantity."""

    name: str
    before: float
    after: float
    note: str


@dataclass(frozen=True)
class OrderRecommendation:
    """A complete, explainable replenishment recommendation for one SKU."""

    sku: str
    as_of: pd.Timestamp
    policy_name: str
    model_name: str

    on_hand: float
    on_order: float
    backorders: float
    inventory_position: float

    lead_time_days: int
    review_period_days: int
    protection_period_days: int
    service_level: float
    sigma_daily: float

    daily_forecast: np.ndarray
    expected_demand_protection: float
    expected_demand_horizon: float

    safety_stock: float
    target_stock: float
    reorder_point: float | None

    raw_order_quantity: float
    recommended_order_quantity: float
    adjustments: list[ConstraintAdjustment]

    unit_cost: float
    pack_size: int
    min_order_quantity: int
    shelf_life_days: int
    order_value: float
    waste_cost_per_unit: float
    stockout_cost_per_unit: float

    days_of_cover: float
    stockout_probability: float
    projected_waste_units: float
    stockout_risk: RiskLevel
    waste_risk: RiskLevel

    metadata: dict = field(default_factory=dict)

    @property
    def order_required(self) -> bool:
        """Whether the manager needs to place an order for this SKU."""
        return self.recommended_order_quantity > 0

    @property
    def expected_shortfall_units(self) -> float:
        """Demand over the protection period that current stock cannot cover.

        Measured against the inventory position rather than the position plus this
        order, because a replenishment placed today only arrives after the lead time.
        """
        return float(max(self.expected_demand_protection - self.inventory_position, 0.0))

    @property
    def projected_stockout_cost(self) -> float:
        """Expected cost of the shortfall, weighted by its estimated probability."""
        return float(self.stockout_probability * self.expected_shortfall_units * self.stockout_cost_per_unit)

    @property
    def projected_waste_cost(self) -> float:
        """Cost of the units projected not to be used within their shelf life."""
        return float(self.projected_waste_units * self.waste_cost_per_unit)

    @property
    def risk_level(self) -> RiskLevel:
        """The more severe of the stockout and waste risks."""
        order = {"low": 0, "medium": 1, "high": 2}
        return max((self.stockout_risk, self.waste_risk), key=lambda r: order[r])

    def to_row(self) -> dict:
        """Flat representation for tables and the SQLite recommendations table."""
        return {
            "sku": self.sku,
            "as_of": pd.Timestamp(self.as_of),
            "on_hand": round(self.on_hand, 1),
            "on_order": round(self.on_order, 1),
            "inventory_position": round(self.inventory_position, 1),
            "forecast_demand_horizon": round(self.expected_demand_horizon, 1),
            "forecast_demand_protection": round(self.expected_demand_protection, 1),
            "safety_stock": round(self.safety_stock, 1),
            "target_stock": round(self.target_stock, 1),
            "recommended_order": round(self.recommended_order_quantity, 1),
            "order_value": round(self.order_value, 2),
            "days_of_cover": round(self.days_of_cover, 2),
            "stockout_probability": round(self.stockout_probability, 3),
            "projected_waste_units": round(self.projected_waste_units, 1),
            "risk": self.risk_level,
            "model": self.model_name,
            "policy": self.policy_name,
        }

    def to_dict(self) -> dict:
        """Full dictionary form, with the forecast as a list."""
        data = asdict(self)
        data["daily_forecast"] = [float(v) for v in self.daily_forecast]
        data["as_of"] = str(pd.Timestamp(self.as_of).date())
        return data


def round_to_pack(quantity: float, pack_size: int, mode: str = "nearest") -> float:
    """Round an order to whole supplier cases.

    Args:
        quantity: Raw quantity in units.
        pack_size: Units per case. A pack size of 1 leaves the quantity unchanged.
        mode: ``"nearest"``, ``"up"`` or ``"none"``.

    Returns:
        The rounded quantity. ``"nearest"`` can legitimately return zero when less than
        half a case is needed; the caller decides whether that means "no order".

    Raises:
        ValueError: If the pack size is not positive or the mode is unknown.
    """
    if pack_size <= 0:
        raise ValueError(f"Pack size must be positive, got {pack_size}.")
    if quantity <= 0:
        return 0.0
    match mode:
        case "none":
            return float(quantity)
        case "up":
            return float(math.ceil(quantity / pack_size) * pack_size)
        case "nearest":
            return float(round(quantity / pack_size) * pack_size)
    raise ValueError(f"Unknown pack rounding mode '{mode}'. Use 'nearest', 'up' or 'none'.")


def apply_order_constraints(
    raw_quantity: float,
    inputs: PolicyInputs,
    economics: SkuEconomics,
    config: AppConfig,
    *,
    enforce_shelf_life: bool | None = None,
) -> tuple[float, list[ConstraintAdjustment]]:
    """Apply shelf-life, sizing and packaging constraints in a fixed order.

    The sequence is deliberate:

    1. cap by what can be sold before the delivery expires (perishability);
    2. cap by the configured maximum days of supply (guards against a runaway forecast);
    3. round to supplier case size;
    4. raise a non-zero order to the minimum order quantity.

    Args:
        raw_quantity: ``max(0, target stock - inventory position)``.
        inputs: The policy inputs that produced the raw quantity.
        economics: Resolved per-SKU commercial parameters.
        config: Application configuration.
        enforce_shelf_life: Override the configured shelf-life cap. The manual baseline
            passes ``False``, because a par level set from average usage carries no
            explicit perishability logic.

    Returns:
        ``(final_quantity, adjustments)`` where ``adjustments`` records every change.
    """
    adjustments: list[ConstraintAdjustment] = []
    quantity = float(max(raw_quantity, 0.0))
    if quantity <= 0:
        return 0.0, adjustments

    shelf_life_enabled = (
        config.inventory.enforce_shelf_life_cap if enforce_shelf_life is None else enforce_shelf_life
    )
    if shelf_life_enabled:
        cap = shelf_life_order_cap(
            inputs.daily_forecast,
            economics.shelf_life_days,
            config.inventory.shelf_life_usable_fraction,
            inputs.position,
            inputs.lead_time_days,
        )
        if quantity > cap:
            usable = usable_shelf_life_days(economics.shelf_life_days, config.inventory.shelf_life_usable_fraction)
            adjustments.append(
                ConstraintAdjustment(
                    "shelf_life_cap", quantity, cap,
                    f"Reduced to what can be used within the {usable:.0f}-day usable shelf life "
                    f"after delivery; ordering more would expire before it is sold.",
                )
            )
            quantity = cap

    max_units = config.inventory.max_order_days_of_supply * inputs.mean_daily_forecast
    if max_units > 0 and quantity > max_units:
        adjustments.append(
            ConstraintAdjustment(
                "max_days_of_supply", quantity, max_units,
                f"Capped at {config.inventory.max_order_days_of_supply:.0f} days of expected demand "
                "as a guardrail against an over-stated forecast.",
            )
        )
        quantity = max_units

    if config.inventory.pack_rounding != "none" and economics.pack_size > 1:
        rounded = round_to_pack(quantity, economics.pack_size, config.inventory.pack_rounding)
        if abs(rounded - quantity) > 1e-9:
            adjustments.append(
                ConstraintAdjustment(
                    "pack_rounding", quantity, rounded,
                    f"Rounded to whole cases of {economics.pack_size} "
                    f"({config.inventory.pack_rounding} rounding).",
                )
            )
            quantity = rounded

    if 0 < quantity < economics.min_order_quantity:
        adjustments.append(
            ConstraintAdjustment(
                "min_order_quantity", quantity, float(economics.min_order_quantity),
                f"Raised to the supplier minimum order quantity of {economics.min_order_quantity}.",
            )
        )
        quantity = float(economics.min_order_quantity)

    return float(max(quantity, 0.0)), adjustments


def _classify(value: float, thresholds: tuple[float, float]) -> RiskLevel:
    low, high = thresholds
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def recommend_order(
    inputs: PolicyInputs,
    economics: SkuEconomics,
    config: AppConfig,
    *,
    as_of: pd.Timestamp,
    policy: InventoryPolicy | None = None,
    model_name: str = "",
    metadata: dict | None = None,
    enforce_shelf_life: bool | None = None,
) -> OrderRecommendation:
    """Produce a full order recommendation for one SKU.

    Args:
        inputs: Demand forecast, uncertainty and stock position.
        economics: Resolved per-SKU costs, pack size and shelf life.
        config: Application configuration.
        as_of: The decision date.
        policy: Inventory policy; built from configuration when omitted.
        model_name: Name of the forecasting model, recorded for the audit trail.
        metadata: Extra information to carry through (segment, scenario label, ...).
        enforce_shelf_life: Override the configured shelf-life cap.

    Returns:
        An :class:`OrderRecommendation` containing the recommended quantity, every
        intermediate value behind it and the risk assessment.
    """
    policy = policy or build_policy(config.inventory.policy, config.inventory.s_S_reorder_multiplier)
    raw_quantity, targets = policy.raw_order_quantity(inputs)
    quantity, adjustments = apply_order_constraints(
        raw_quantity, inputs, economics, config, enforce_shelf_life=enforce_shelf_life
    )

    if not targets.should_order and quantity > 0:  # defensive: (s, S) band respected
        quantity = 0.0

    horizon_demand = float(np.sum(inputs.daily_forecast))
    mean_daily = inputs.mean_daily_forecast
    days_of_cover = float(inputs.on_hand / mean_daily) if mean_daily > 0 else float("inf")

    risk_p = stockout_probability(
        inputs.position, targets.expected_demand_protection, inputs.sigma_daily, targets.protection_period_days
    )
    usable = usable_shelf_life_days(economics.shelf_life_days, config.inventory.shelf_life_usable_fraction)
    demand_before_expiry = expected_demand_over(inputs.daily_forecast, inputs.lead_time_days + usable)
    projected_stock = inputs.position + quantity
    projected_waste = float(max(projected_stock - demand_before_expiry, 0.0))
    waste_share = projected_waste / projected_stock if projected_stock > 0 else 0.0

    return OrderRecommendation(
        sku=inputs.sku,
        as_of=pd.Timestamp(as_of),
        policy_name=targets.policy_name,
        model_name=model_name,
        on_hand=float(inputs.on_hand),
        on_order=float(inputs.on_order),
        backorders=float(inputs.backorders),
        inventory_position=float(inputs.position),
        lead_time_days=int(inputs.lead_time_days),
        review_period_days=int(inputs.review_period_days),
        protection_period_days=int(targets.protection_period_days),
        service_level=float(inputs.service_level),
        sigma_daily=float(inputs.sigma_daily),
        daily_forecast=np.asarray(inputs.daily_forecast, dtype=float),
        expected_demand_protection=float(targets.expected_demand_protection),
        expected_demand_horizon=horizon_demand,
        safety_stock=float(targets.safety_stock),
        target_stock=float(targets.target_stock),
        reorder_point=targets.reorder_point,
        raw_order_quantity=float(raw_quantity),
        recommended_order_quantity=float(quantity),
        adjustments=adjustments,
        unit_cost=economics.unit_cost,
        pack_size=economics.pack_size,
        min_order_quantity=economics.min_order_quantity,
        shelf_life_days=economics.shelf_life_days,
        order_value=float(quantity * economics.unit_cost),
        waste_cost_per_unit=economics.waste_cost_per_unit,
        stockout_cost_per_unit=economics.stockout_cost_per_unit,
        days_of_cover=days_of_cover,
        stockout_probability=risk_p,
        projected_waste_units=projected_waste,
        stockout_risk=_classify(risk_p, STOCKOUT_RISK_THRESHOLDS),
        waste_risk=_classify(waste_share, WASTE_RISK_THRESHOLDS),
        metadata={**(metadata or {}), "policy_detail": targets.detail},
    )


def recommendations_to_frame(recommendations: list[OrderRecommendation]) -> pd.DataFrame:
    """Tabulate recommendations for the dashboard and the database."""
    if not recommendations:
        return pd.DataFrame(
            columns=["sku", "as_of", "on_hand", "recommended_order", "risk", "model", "policy"]
        )
    return pd.DataFrame([r.to_row() for r in recommendations]).sort_values("sku").reset_index(drop=True)
