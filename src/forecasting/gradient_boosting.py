"""Global gradient-boosting demand model.

One model is trained across all SKUs with the SKU identity as a categorical feature, so
that slow movers borrow strength from the calendar and weather patterns learned on fast
movers. This reproduces the AE1 baseline configuration
(``HistGradientBoostingRegressor``, 300 iterations, learning rate 0.05, depth 6, seed
42) but fixes the ``roll28`` construction and freezes features at the forecast origin.

Forecast uncertainty is estimated on a held-out tail of the training window rather than
in-sample, because in-sample residuals of a boosted model badly understate error and
would produce safety stock that is too small.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import AppConfig
from src.features.demand_features import (
    SkuEncoder,
    engineer_features,
    feature_columns,
    prepare_feature_matrix,
)
from src.forecasting.base import ForecastModel, ForecastResult, z_score

logger = logging.getLogger(__name__)

#: Features treated as categorical by the booster.
CATEGORICAL_FEATURES: tuple[str, ...] = ("sku_code", "dow")


class GradientBoostingForecast(ForecastModel):
    """Feature-based global model for smooth, fast-moving SKUs.

    Args:
        config: Application configuration (hyper-parameters, seed, feature definitions).
        encoder: SKU encoder; fitted from the training panel when omitted.
        service_level: Coverage of the reported prediction interval.
        sigma_validation_days: Length of the held-out tail used to estimate forecast
            error. Set to 0 to fall back to in-sample residuals (faster, optimistic).
    """

    name = "Gradient Boosting"
    is_global = True

    def __init__(
        self,
        config: AppConfig,
        encoder: SkuEncoder | None = None,
        service_level: float | None = None,
        sigma_validation_days: int = 28,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.service_level = service_level if service_level is not None else config.forecast.interval_service_level
        self._z = z_score(self.service_level)
        self.sigma_validation_days = int(sigma_validation_days)
        self.sigma_by_sku_: dict[str, float] = {}
        self.model_: HistGradientBoostingRegressor | None = None
        self.feature_names_: list[str] = feature_columns(config)
        self.train_end_: pd.Timestamp | None = None
        self.n_train_rows_: int = 0

    # -- fitting -----------------------------------------------------------------------

    def _make_estimator(self) -> HistGradientBoostingRegressor:
        gb = self.config.forecast.gradient_boosting
        categorical = [c for c in CATEGORICAL_FEATURES if c in self.feature_names_]
        return HistGradientBoostingRegressor(
            max_iter=gb.max_iter,
            learning_rate=gb.learning_rate,
            max_depth=gb.max_depth,
            min_samples_leaf=gb.min_samples_leaf,
            l2_regularization=gb.l2_regularization,
            random_state=self.config.forecast.random_seed,
            categorical_features=categorical,
        )

    def fit(self, history: pd.DataFrame) -> "GradientBoostingForecast":
        """Train on a multi-SKU panel of observed demand.

        Args:
            history: Tidy sales frame (``date``, ``sku``, ``units_sold``, ``temp_c``,
                ``promo``). Features are engineered internally.
        """
        if history.empty:
            raise ValueError("Cannot fit the gradient-boosting model on an empty history.")
        self.encoder = self.encoder or SkuEncoder.fit([str(s) for s in history["sku"].unique()])
        engineered = engineer_features(history, self.config, self.encoder)
        X, y = prepare_feature_matrix(engineered, self.config, dropna=True)
        if y is None or X.empty:
            raise ValueError(
                "No usable training rows after dropping undefined lag/rolling features. "
                f"History spans {len(history)} rows; the longest window needs "
                f"{max(self.config.features.rolling_windows, default=0)} days per SKU."
            )
        self.model_ = self._make_estimator().fit(X, y)
        self.train_end_ = pd.Timestamp(history["date"].max())
        self.n_train_rows_ = len(X)
        self._estimate_sigma(engineered)
        self._fitted = True
        logger.info(
            "Fitted %s on %d rows across %d SKUs (history to %s)",
            self.name, self.n_train_rows_, history["sku"].nunique(), self.train_end_.date(),
        )
        return self

    def _estimate_sigma(self, engineered: pd.DataFrame) -> None:
        """Estimate per-SKU forecast error on a held-out tail of the training window."""
        self.sigma_by_sku_ = {}
        cutoff = None
        if self.sigma_validation_days > 0:
            cutoff = engineered["date"].max() - pd.Timedelta(days=self.sigma_validation_days)
            train = engineered[engineered["date"] < cutoff]
            valid = engineered[engineered["date"] >= cutoff]
            X_tr, y_tr = prepare_feature_matrix(train, self.config, dropna=True)
            X_va, y_va = prepare_feature_matrix(valid, self.config, dropna=True)
            if not X_tr.empty and not X_va.empty and y_tr is not None and y_va is not None:
                probe = self._make_estimator().fit(X_tr, y_tr)
                resid = y_va.to_numpy() - np.clip(probe.predict(X_va), 0.0, None)
                frame = valid.dropna(subset=self.feature_names_).assign(_resid=resid)
                self.sigma_by_sku_ = {
                    str(sku): float(np.std(g["_resid"].to_numpy(), ddof=1)) if len(g) > 1 else 0.0
                    for sku, g in frame.groupby("sku", observed=True)
                }
                return
            logger.debug("Validation tail too small for out-of-sample sigma; using in-sample residuals.")

        X, y = prepare_feature_matrix(engineered, self.config, dropna=True)
        if y is None or X.empty or self.model_ is None:
            return
        resid = y.to_numpy() - np.clip(self.model_.predict(X), 0.0, None)
        frame = engineered.dropna(subset=self.feature_names_).assign(_resid=resid)
        self.sigma_by_sku_ = {
            str(sku): float(np.std(g["_resid"].to_numpy(), ddof=1)) if len(g) > 1 else 0.0
            for sku, g in frame.groupby("sku", observed=True)
        }

    # -- prediction --------------------------------------------------------------------

    def _validate_future(self, future: pd.DataFrame) -> None:
        self._check_fitted()
        if "units_sold" in future.columns:
            raise ValueError(
                "Future frame carries 'units_sold'; realised demand must not be visible "
                "when predicting."
            )
        missing = [c for c in self.feature_names_ if c not in future.columns]
        if missing:
            raise ValueError(
                f"Future frame is missing feature column(s) {missing}. Build it with "
                "features.demand_features.build_future_features."
            )
        if self.train_end_ is not None and (future["date"].min() <= self.train_end_):
            logger.debug(
                "Predicting dates at or before the training cutoff (%s); this is only "
                "valid for in-sample diagnostics.", self.train_end_.date(),
            )

    def predict(self, future: pd.DataFrame) -> ForecastResult:
        """Forecast a single SKU over the rows supplied in ``future``."""
        self._validate_future(future)
        skus = future["sku"].unique()
        if len(skus) != 1:
            raise ValueError(
                f"predict() handles one SKU at a time; received {len(skus)}. "
                "Use predict_panel() for a multi-SKU frame."
            )
        return self._predict_one(future.sort_values("date", kind="stable"))

    def predict_panel(self, future: pd.DataFrame) -> list[ForecastResult]:
        """Forecast every SKU present in ``future`` with a single model pass."""
        self._validate_future(future)
        ordered = future.sort_values(["sku", "date"], kind="stable")
        preds = np.clip(self.model_.predict(ordered[self.feature_names_].astype("float64")), 0.0, None)
        ordered = ordered.assign(_pred=preds)
        return [self._assemble(str(sku), g) for sku, g in ordered.groupby("sku", observed=True)]

    def _predict_one(self, future: pd.DataFrame) -> ForecastResult:
        preds = np.clip(self.model_.predict(future[self.feature_names_].astype("float64")), 0.0, None)
        return self._assemble(str(future["sku"].iloc[0]), future.assign(_pred=preds))

    def _assemble(self, sku: str, frame: pd.DataFrame) -> ForecastResult:
        dates = pd.DatetimeIndex(frame["date"].to_numpy())
        point = frame["_pred"].to_numpy(dtype=float)
        sigma = float(self.sigma_by_sku_.get(sku, 0.0))
        steps = np.arange(1, len(point) + 1, dtype=float)
        width = self._z * sigma * np.sqrt(steps)
        return ForecastResult(
            sku=sku,
            model_name=self.name,
            dates=dates,
            predicted=point,
            lower=np.clip(point - width, 0.0, None),
            upper=point + width,
            sigma=sigma,
            metadata={
                "n_train_rows": self.n_train_rows_,
                "train_end": str(self.train_end_.date()) if self.train_end_ is not None else None,
                "sigma_estimated_on": "validation_tail" if self.sigma_validation_days > 0 else "in_sample",
            },
        )
