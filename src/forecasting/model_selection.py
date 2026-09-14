"""Per-SKU model selection with an auditable reason for every choice.

No single forecasting method suits all 20 SKUs. A gradient-boosting model trained on
calendar and weather features is strong on smooth, fast-moving items but produces a
small positive forecast on every day for an item that only sells twice a week; a
Croston-family method handles the latter properly but cannot represent a weekend uplift.

Two strategies are available:

``rule``
    Fast, deterministic routing from the ABC/XYZ segment and the share of zero-demand
    days. Used inside the simulator, where selection is repeated many times.

``validated``
    Runs a short rolling-origin evaluation on history *before* the decision date and
    picks the model with the lowest WAPE per SKU. Slower, and used when preparing the
    model registry.

Both record a :class:`ModelChoice` naming the model, the segment and the reason, so the
dissertation can state exactly which method produced each SKU's numbers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import AppConfig
from src.data.preprocessing import history_as_of
from src.forecasting.base import ForecastModel
from src.forecasting.croston import CrostonForecast
from src.forecasting.ets import ETSForecast
from src.forecasting.naive import NaiveForecast
from src.forecasting.seasonal_naive import SeasonalNaiveForecast
from src.segmentation import SkuSegment, segment_skus, to_segments

logger = logging.getLogger(__name__)

Strategy = Literal["rule", "validated"]

#: Model names this selector can route to. The gradient-boosting model is global and is
#: instantiated by the forecasting service rather than per SKU.
GRADIENT_BOOSTING = "Gradient Boosting"
CANDIDATE_MODELS = (GRADIENT_BOOSTING, "ETS (Holt-Winters)", "Seasonal Naive", "Croston (SBA)")


@dataclass(frozen=True)
class ModelChoice:
    """The model chosen for one SKU and the justification for it."""

    sku: str
    model_name: str
    reason: str
    segment: str
    strategy: str
    selected_at: str
    candidate_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ModelSelector:
    """Chooses a forecasting model per SKU and records why.

    Args:
        config: Application configuration.
        strategy: ``"rule"`` for segment-based routing, ``"validated"`` to compare
            candidate models on a held-out tail of history.
        validation_folds: Number of rolling-origin folds used by the validated strategy.
    """

    def __init__(self, config: AppConfig, strategy: Strategy = "rule", validation_folds: int = 3) -> None:
        if strategy not in ("rule", "validated"):
            raise ValueError(f"Unknown selection strategy '{strategy}'.")
        self.config = config
        self.strategy = strategy
        self.validation_folds = int(validation_folds)

    # -- public API --------------------------------------------------------------------

    def select(
        self,
        sales: pd.DataFrame,
        sku_master: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
    ) -> dict[str, ModelChoice]:
        """Select a model for every SKU using data available before ``as_of``."""
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(sales["date"].max()) + pd.Timedelta(days=1)
        segments = to_segments(segment_skus(sales, sku_master, self.config, as_of=as_of))
        if self.strategy == "rule":
            return {sku: self._rule_choice(seg, sales, as_of) for sku, seg in segments.items()}
        return self._validated_choices(sales, segments, as_of)

    def build(self, choice: ModelChoice) -> ForecastModel | None:
        """Instantiate the univariate model named by ``choice``.

        Returns:
            A fitted-ready model instance, or ``None`` for the global gradient-boosting
            model, which the forecasting service owns and shares across SKUs.
        """
        cfg = self.config
        sl = cfg.forecast.interval_service_level
        match choice.model_name:
            case "Gradient Boosting":
                return None
            case "ETS (Holt-Winters)":
                return ETSForecast(cfg.forecast.seasonal_period, service_level=sl)
            case "Seasonal Naive":
                return SeasonalNaiveForecast(cfg.forecast.seasonal_period, service_level=sl)
            case "Croston (SBA)":
                return CrostonForecast(cfg.forecast.croston.alpha, "sba", service_level=sl)
            case "Croston":
                return CrostonForecast(cfg.forecast.croston.alpha, "classic", service_level=sl)
            case "Naive":
                return NaiveForecast(service_level=sl)
        raise ValueError(f"Cannot build unknown model '{choice.model_name}' for SKU '{choice.sku}'.")

    # -- strategies --------------------------------------------------------------------

    def _rule_choice(self, segment: SkuSegment, sales: pd.DataFrame, as_of: pd.Timestamp) -> ModelChoice:
        """Route a SKU to a model from its segment and intermittency."""
        threshold = self.config.forecast.croston.intermittent_zero_share
        n_days = int((as_of - pd.Timestamp(sales["date"].min())).days)
        zero_pct = segment.zero_day_share * 100

        if segment.zero_day_share >= threshold:
            model = "Croston (SBA)"
            reason = (
                f"{zero_pct:.0f}% of days have no demand, above the {threshold:.0%} "
                "intermittency threshold. A Croston-family method forecasts the demand "
                "rate instead of fitting a curve through the zeros; the SBA variant "
                "removes the known upward bias of classic Croston."
            )
        elif n_days < self.config.forecast.min_history_days:
            model = "Seasonal Naive"
            reason = (
                f"Only {n_days} days of history are available, fewer than the "
                f"{self.config.forecast.min_history_days} required to train the machine "
                "learning model. The previous week is repeated until more data exists."
            )
        else:
            model = GRADIENT_BOOSTING
            reason = (
                f"Demand is {'stable' if segment.xyz == 'X' else 'variable but continuous'} "
                f"(coefficient of variation {segment.demand_cv:.2f}, {zero_pct:.0f}% zero-demand "
                "days). The gradient-boosting model can use day of week, holidays, "
                "weather and recent demand for this SKU."
            )
        return ModelChoice(
            sku=segment.sku,
            model_name=model,
            reason=reason,
            segment=segment.segment,
            strategy="rule",
            selected_at=str(as_of.date()),
        )

    def _validated_choices(
        self,
        sales: pd.DataFrame,
        segments: dict[str, SkuSegment],
        as_of: pd.Timestamp,
    ) -> dict[str, ModelChoice]:
        """Compare candidate models on a rolling-origin tail of history before ``as_of``."""
        from src.evaluation.backtest import DEFAULT_UNIVARIATE_MODELS, run_backtest

        history = history_as_of(sales, as_of)
        candidates = {
            k: v for k, v in DEFAULT_UNIVARIATE_MODELS.items()
            if k in ("ETS (Holt-Winters)", "Seasonal Naive", "Croston (SBA)")
        }
        result = run_backtest(
            history,
            self.config,
            n_folds=self.validation_folds,
            horizon=self.config.forecast.horizon_days,
            univariate_models=candidates,
            include_gradient_boosting=True,
            estimate_sigma=False,
        )
        scores = (
            result.metrics.groupby(["sku", "model"], observed=True)["WAPE"].mean().unstack("model")
        )

        choices: dict[str, ModelChoice] = {}
        for sku, segment in segments.items():
            if sku not in scores.index:
                choices[sku] = self._rule_choice(segment, sales, as_of)
                continue
            row = scores.loc[sku].dropna()
            if row.empty:
                choices[sku] = self._rule_choice(segment, sales, as_of)
                continue
            best = str(row.idxmin())
            runner_up = row.drop(index=best)
            margin = (
                f", {(runner_up.min() - row[best]):.1f} percentage points better than "
                f"the next best ({runner_up.idxmin()})" if not runner_up.empty else ""
            )
            choices[sku] = ModelChoice(
                sku=sku,
                model_name=best,
                reason=(
                    f"Selected by a {self.validation_folds}-fold rolling-origin comparison on "
                    f"history before {as_of.date()}: lowest WAPE at {row[best]:.1f}%{margin}. "
                    f"Segment {segment.segment} ({segment.describe().split(': ', 1)[1]})."
                ),
                segment=segment.segment,
                strategy="validated",
                selected_at=str(as_of.date()),
                candidate_scores={str(k): round(float(v), 2) for k, v in row.items()},
            )
        return choices

    # -- persistence -------------------------------------------------------------------

    @staticmethod
    def to_frame(choices: dict[str, ModelChoice]) -> pd.DataFrame:
        """Tabulate the selection for display and for the model-metadata table."""
        return pd.DataFrame([c.to_dict() for c in choices.values()]).sort_values("sku").reset_index(drop=True)

    @staticmethod
    def save(choices: dict[str, ModelChoice], path: str | Path) -> None:
        """Write the selection to JSON so training and inference stay separate."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {sku: choice.to_dict() for sku, choice in choices.items()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved model selection for %d SKUs to %s", len(payload), path)

    @staticmethod
    def load(path: str | Path) -> dict[str, ModelChoice]:
        """Load a previously saved selection.

        Raises:
            FileNotFoundError: If the registry file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model selection found at {path}. Run 'python scripts/train_models.py' first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {sku: ModelChoice(**data) for sku, data in payload.items()}
