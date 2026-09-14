"""Safety stock calculation.

Safety stock buys availability with cash and, for perishable food, with waste. It is
sized from *forecast error* rather than raw demand variability: a SKU whose demand swings
predictably with the weekend does not need cover for that swing, because the forecast
already anticipates it. Using raw demand standard deviation would systematically
over-order on exactly the fast-moving fresh items where waste is most expensive.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.forecasting.base import z_score

logger = logging.getLogger(__name__)


def safety_stock(
    sigma_daily: float,
    protection_period_days: float,
    service_level: float,
) -> float:
    """Compute safety stock for a deterministic lead time.

    Uses the standard normal approximation

        SS = z(service level) x sigma_daily x sqrt(protection period)

    where ``sigma_daily`` is the standard deviation of the one-day-ahead forecast error
    and errors are assumed independent across days.

    Args:
        sigma_daily: One-day forecast error standard deviation, in units.
        protection_period_days: Lead time plus review period, in days.
        service_level: Target cycle service level in (0, 1).

    Returns:
        Safety stock in units (never negative).

    Raises:
        ValueError: If the protection period is negative or sigma is negative.
    """
    if protection_period_days < 0:
        raise ValueError(f"Protection period must be non-negative, got {protection_period_days}.")
    if sigma_daily < 0:
        raise ValueError(f"Demand uncertainty must be non-negative, got {sigma_daily}.")
    z = z_score(service_level)
    return float(max(z * sigma_daily * math.sqrt(protection_period_days), 0.0))


def safety_stock_with_lead_time_variability(
    sigma_daily: float,
    mean_daily_demand: float,
    protection_period_days: float,
    service_level: float,
    lead_time_sigma_days: float = 0.0,
) -> float:
    """Safety stock when the supplier lead time is itself uncertain.

    Combines demand and lead-time variance:

        SS = z x sqrt(P x sigma_d^2 + mu_d^2 x sigma_L^2)

    The project's supplier lead times are deterministic, so ``lead_time_sigma_days``
    defaults to zero and the result reduces to :func:`safety_stock`. The parameter exists
    so that a sensitivity analysis on unreliable suppliers can be run from configuration.
    """
    if lead_time_sigma_days < 0:
        raise ValueError("Lead-time variability must be non-negative.")
    z = z_score(service_level)
    variance = protection_period_days * sigma_daily**2 + (mean_daily_demand * lead_time_sigma_days) ** 2
    return float(max(z * math.sqrt(max(variance, 0.0)), 0.0))


def expected_demand_over(daily_forecast: np.ndarray | list[float], days: float) -> float:
    """Total expected demand over the first ``days`` of a daily forecast.

    Fractional days are interpolated. When the requested window is longer than the
    forecast, the mean of the forecast is used to extrapolate the remainder, which is
    the honest option: the model has no information beyond its horizon.

    Args:
        daily_forecast: Point forecast per day, ordered from the decision date.
        days: Length of the window in days.

    Returns:
        Expected demand in units.
    """
    values = np.asarray(daily_forecast, dtype=float)
    if days < 0:
        raise ValueError(f"Demand window must be non-negative, got {days}.")
    if values.size == 0:
        return 0.0
    if days == 0:
        return 0.0

    whole = int(math.floor(days))
    fraction = days - whole
    if whole <= values.size:
        total = float(values[:whole].sum())
        if fraction > 0:
            nxt = values[whole] if whole < values.size else float(values.mean())
            total += fraction * float(nxt)
        return total

    # Beyond the horizon: extend with the average forecast day.
    mean_day = float(values.mean())
    extra_days = days - values.size
    logger.debug(
        "Requested %.1f days of expected demand but the forecast covers %d; "
        "extrapolating %.1f days at the forecast mean.", days, values.size, extra_days,
    )
    return float(values.sum() + extra_days * mean_day)


def stockout_probability(
    inventory_position: float,
    expected_demand: float,
    sigma_daily: float,
    protection_period_days: float,
) -> float:
    """Probability that demand over the protection period exceeds available stock.

    Uses the same normal approximation as :func:`safety_stock`. Returned as a
    probability in [0, 1]; with zero uncertainty it degenerates to a hard comparison.
    """
    from scipy import stats

    sigma_period = sigma_daily * math.sqrt(max(protection_period_days, 0.0))
    if sigma_period <= 0:
        return 1.0 if inventory_position < expected_demand else 0.0
    return float(1.0 - stats.norm.cdf((inventory_position - expected_demand) / sigma_period))
