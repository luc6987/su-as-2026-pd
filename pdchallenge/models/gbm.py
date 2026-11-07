"""Gradient boosting over the engineered features.

Boosted trees are here to test one specific hypothesis: that once the de-biasing has
linearised the problem, what remains is a nonlinear interaction between the curve and
the visit's context (which score was observed, how many visits the patient has, how far
into the study this visit sits). If a GBM cannot beat the two-parameter affine
calibration in :mod:`pdchallenge.models.hierarchical`, that hypothesis is wrong and the
ensemble should say so rather than paper over it.

Both backends handle missing values natively, which matters because the design matrix
keeps NaNs for the raw score columns by design.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..evaluation.metrics import clip_to_valid
from ..features.matrix import build_matrix
from .base import BaseModel
from .levelnorm import normalise, patient_scale
from .registry import register

DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
}


@register
class GBMModel(BaseModel):
    """XGBoost or LightGBM over the numeric design matrix."""

    name = "gbm"

    def __init__(
        self,
        backend: str = "xgboost",
        params: dict[str, Any] | None = None,
        seed: int = 0,
        clip: bool = True,
        columns: list[str] | None = None,
        tta: bool = False,
        level_norm: bool = False,
    ) -> None:
        self.backend = backend
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self.clip = clip
        self.columns = columns
        # Scale test-time augmentation (`evaluation/tta.py`): off unless a config asks for it,
        # so every existing run predicts exactly as before.
        self.tta = tta
        # Level normalisation (`levelnorm.py`): score columns as fractions of the patient's own level,
        # target y / s_p with weight s_p². Off unless a config asks for it, so every existing run fits
        # exactly as before.
        self.level_norm = level_norm

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        m = build_matrix(X)
        return m[self.columns] if self.columns else m

    def _design(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
        """The design matrix and, with ``level_norm``, the per-row patient scale it was divided by."""
        Xm = self._matrix(X)
        if not self.level_norm:
            return Xm, None
        s = patient_scale(X)
        return normalise(Xm, s), s

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        Xm, scale = self._design(X)
        yv = np.asarray(y, dtype=float)

        if self.backend == "xgboost":
            from xgboost import XGBRegressor

            self.estimator_ = XGBRegressor(
                random_state=self.seed,
                objective="reg:squarederror",
                n_jobs=-1,
                **self.params,
            )
        elif self.backend == "lightgbm":
            from lightgbm import LGBMRegressor

            params = {k: v for k, v in self.params.items() if k != "min_child_weight"}
            self.estimator_ = LGBMRegressor(
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
                min_child_samples=int(self.params.get("min_child_weight", 5) * 4),
                **params,
            )
        else:
            raise ValueError(f"unknown gbm backend {self.backend!r}; use 'xgboost' or 'lightgbm'")

        if scale is None:
            self.estimator_.fit(Xm, yv)
        else:
            # y / s_p weighted by s_p² is exactly the squared error on the score scale
            self.estimator_.fit(Xm, yv / scale, sample_weight=scale**2)
        self.columns_ = list(Xm.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xm, scale = self._design(X)
        pred = np.asarray(self.estimator_.predict(Xm), dtype=float)
        if scale is not None:
            pred = pred * scale
        return clip_to_valid(pred) if self.clip else pred

    def predict_tta(self, frames: dict[float, pd.DataFrame]) -> np.ndarray:
        """With ``tta``, the mean over scales of ``predict(frame_s) / s``.

        Dividing by ``s`` maps each scaled patient's prediction back to the real one's scale —
        the observations are multiplicative views of the target, so ``s``× the scores is
        ``s``× the target to a good approximation. Each scale is clipped before averaging, as
        the plain prediction is.
        """
        if not self.tta:
            return self.predict(frames[1.0])
        return np.mean([self.predict(f) / s for s, f in frames.items()], axis=0)

    def feature_importance(self, X: pd.DataFrame) -> pd.Series:
        """Gain-based importance, named. SHAP in ``viz.interpret`` is the deeper view."""
        names = getattr(self, "columns_", list(self._matrix(X).columns))
        if self.backend == "xgboost":
            booster = self.estimator_.get_booster()
            score = booster.get_score(importance_type="gain")
            # A booster fitted from a DataFrame — which `_matrix` always builds — names its
            # features after the columns; one fitted from an ndarray names them f0, f1, …
            # Filtering for the positional form alone dropped every entry and returned an
            # empty Series, which reads as "no feature matters" rather than as a bug.
            s = pd.Series({
                (k if k in names else names[int(k[1:])]): v
                for k, v in score.items()
                if k in names or (k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(names))
            })
        else:
            s = pd.Series(self.estimator_.feature_importances_, index=names)
        return s.sort_values(ascending=False)

    def get_params(self) -> dict[str, Any]:
        out = {"backend": self.backend, **self.params}
        if self.tta:
            out["tta"] = True
        if self.level_norm:
            out["level_norm"] = True
        return out
