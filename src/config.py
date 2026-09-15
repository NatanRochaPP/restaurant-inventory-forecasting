"""Typed application configuration.

All operational assumptions (horizons, service levels, costs, supplier calendar,
per-SKU economics) are declared in ``config/config.yaml`` and validated here with
Pydantic. Modules must read values from an :class:`AppConfig` instance rather than
hard-coding constants, so that every assumption behind a published result is
inspectable in one file.
"""

from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

Probability = Annotated[float, Field(gt=0.0, lt=1.0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
PosInt = Annotated[int, Field(gt=0)]


class _Base(BaseModel):
    """Strict base model: unknown keys are configuration errors, not silent no-ops."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Base):
    sales_csv: str
    database: str
    required_columns: list[str]


class BacktestConfig(_Base):
    n_folds: PosInt
    horizon_days: PosInt


class GradientBoostingConfig(_Base):
    max_iter: PosInt
    learning_rate: float = Field(gt=0.0)
    max_depth: PosInt
    min_samples_leaf: PosInt
    l2_regularization: NonNegFloat


class CrostonConfig(_Base):
    alpha: float = Field(gt=0.0, le=1.0)
    intermittent_zero_share: Probability


class ForecastConfig(_Base):
    horizon_days: PosInt
    random_seed: int
    min_history_days: PosInt
    refit_frequency_days: PosInt
    seasonal_period: PosInt
    interval_service_level: Probability
    backtest: BacktestConfig
    gradient_boosting: GradientBoostingConfig
    croston: CrostonConfig


class FeaturesConfig(_Base):
    lags: list[PosInt]
    rolling_windows: list[PosInt]
    assume_promotions_known: bool
    assume_calendar_known: bool
    zero_demand_on_closed_days: bool = True


class WeatherConfig(_Base):
    perfect_foresight: bool
    forecast_noise_std_c: NonNegFloat
    noise_growth_per_day_c: NonNegFloat
    random_seed: int


class InventoryConfig(_Base):
    policy: Literal["base_stock", "s_S"]
    review_period_days: PosInt
    service_level: Probability
    s_S_reorder_multiplier: NonNegFloat
    max_order_days_of_supply: NonNegFloat
    enforce_shelf_life_cap: bool
    shelf_life_usable_fraction: float = Field(gt=0.0, le=1.0)
    pack_rounding: Literal["nearest", "up", "none"]
    allow_backorders: bool


class SupplierConfig(_Base):
    lead_time_days: int = Field(ge=0)
    order_days: list[int]
    ordering_cost_per_order: NonNegFloat

    @field_validator("order_days")
    @classmethod
    def _valid_weekdays(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("supplier.order_days must contain at least one weekday")
        if any(d < 0 or d > 6 for d in v):
            raise ValueError("supplier.order_days must be weekday integers 0 (Mon) to 6 (Sun)")
        return sorted(set(v))


class CostsConfig(_Base):
    holding_rate_annual: NonNegFloat
    holding_cost_per_unit_per_day: float | None = None
    waste_cost_multiplier: NonNegFloat
    waste_disposal_cost_per_unit: NonNegFloat
    stockout_cost_multiplier: NonNegFloat
    stockout_cost_per_unit: float | None = None


class SimulationConfig(_Base):
    start_date: date
    end_date: date
    warmup_days: int = Field(ge=0)
    initial_stock_days_of_supply: NonNegFloat
    random_seed: int

    @model_validator(mode="after")
    def _ordered_dates(self) -> "SimulationConfig":
        if self.end_date <= self.start_date:
            raise ValueError("simulation.end_date must be after simulation.start_date")
        return self


class BaselineConfig(_Base):
    name: str
    lookback_days: PosInt
    buffer_pct: NonNegFloat
    cover_days: int | None = None
    apply_shelf_life_cap: bool = False


class SkuDefaults(_Base):
    unit_cost: float = Field(gt=0.0)
    pack_size: PosInt
    min_order_quantity: int = Field(ge=0)


class SkuOverride(_Base):
    unit_cost: float | None = Field(default=None, gt=0.0)
    pack_size: PosInt | None = None
    min_order_quantity: int | None = Field(default=None, ge=0)
    shelf_life_days: PosInt | None = None


class SegmentationConfig(_Base):
    abc_thresholds: list[float]
    xyz_thresholds: list[float]
    lookback_days: PosInt

    @field_validator("abc_thresholds")
    @classmethod
    def _abc_ordered(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or not (0 < v[0] < v[1] < 1):
            raise ValueError("segmentation.abc_thresholds must be two increasing values in (0, 1)")
        return v

    @field_validator("xyz_thresholds")
    @classmethod
    def _xyz_ordered(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or not (0 < v[0] < v[1]):
            raise ValueError("segmentation.xyz_thresholds must be two increasing positive values")
        return v


class SkuEconomics(_Base):
    """Fully resolved commercial parameters for one SKU.

    Produced by :meth:`AppConfig.sku_economics`, which merges the per-SKU block of the
    configuration with the shelf life carried by the dataset.
    """

    sku: str
    unit_cost: float = Field(gt=0.0)
    pack_size: PosInt
    min_order_quantity: int = Field(ge=0)
    shelf_life_days: PosInt
    holding_cost_per_unit_per_day: NonNegFloat
    waste_cost_per_unit: NonNegFloat
    stockout_cost_per_unit: NonNegFloat


class AppConfig(_Base):
    """Root configuration object."""

    data: DataConfig
    forecast: ForecastConfig
    features: FeaturesConfig
    weather: WeatherConfig
    inventory: InventoryConfig
    supplier: SupplierConfig
    costs: CostsConfig
    simulation: SimulationConfig
    baseline: BaselineConfig
    sku_defaults: SkuDefaults
    skus: dict[str, SkuOverride] = Field(default_factory=dict)
    segmentation: SegmentationConfig

    # -- derived helpers ---------------------------------------------------------------

    @property
    def protection_period_days(self) -> int:
        """Lead time plus review period - the window a replenishment must cover."""
        return self.supplier.lead_time_days + self.inventory.review_period_days

    def path(self, relative: str) -> Path:
        """Resolve a configured relative path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def sku_economics(self, sku: str, shelf_life_days: int) -> SkuEconomics:
        """Resolve the commercial parameters for ``sku``.

        Args:
            sku: SKU name as it appears in the sales data.
            shelf_life_days: Shelf life carried by the dataset, used unless the
                configuration overrides it.

        Returns:
            A fully populated :class:`SkuEconomics`. Holding, waste and stockout costs
            are derived from ``unit_cost`` unless absolute overrides are configured.
        """
        override = self.skus.get(sku, SkuOverride())
        unit_cost = override.unit_cost if override.unit_cost is not None else self.sku_defaults.unit_cost
        pack_size = override.pack_size or self.sku_defaults.pack_size
        moq = override.min_order_quantity if override.min_order_quantity is not None else self.sku_defaults.min_order_quantity
        shelf_life = override.shelf_life_days or int(shelf_life_days)

        c = self.costs
        holding = (
            c.holding_cost_per_unit_per_day
            if c.holding_cost_per_unit_per_day is not None
            else unit_cost * c.holding_rate_annual / 365.0
        )
        waste = unit_cost * c.waste_cost_multiplier + c.waste_disposal_cost_per_unit
        stockout = (
            c.stockout_cost_per_unit
            if c.stockout_cost_per_unit is not None
            else unit_cost * c.stockout_cost_multiplier
        )
        return SkuEconomics(
            sku=sku,
            unit_cost=unit_cost,
            pack_size=pack_size,
            min_order_quantity=moq,
            shelf_life_days=shelf_life,
            holding_cost_per_unit_per_day=holding,
            waste_cost_per_unit=waste,
            stockout_cost_per_unit=stockout,
        )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the YAML configuration.

    Args:
        path: Optional path to a configuration file. Defaults to ``config/config.yaml``.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        pydantic.ValidationError: If any value is missing or out of range.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    config = AppConfig.model_validate(raw)
    logger.debug("Loaded configuration from %s", cfg_path)
    return config


@lru_cache(maxsize=4)
def get_config(path: str | None = None) -> AppConfig:
    """Cached accessor for the application configuration."""
    return load_config(path)
