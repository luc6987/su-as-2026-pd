"""A plain ridge on the engineered features — the sanity-check model.

If a gradient-boosted ensemble cannot beat this, the extra machinery is not earning its
keep. Imputation and scaling live inside the estimator so they are fitted per fold and
never see validation rows.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation.metrics import clip_to_valid
from ..features.matrix import build_matrix
from .base import BaseModel
from .registry import register


@register
class RidgeModel(BaseModel):
    """Standardise, impute, ridge."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, clip: bool = True, columns: list[str] | None = None) -> None:
        self.alpha = alpha
        self.clip = clip
        self.columns = columns

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        m = build_matrix(X)
        return m[self.columns] if self.columns else m

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        self.pipeline_ = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.pipeline_.fit(self._matrix(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.pipeline_.predict(self._matrix(X))
        return clip_to_valid(pred) if self.clip else pred

    def coefficients(self, X: pd.DataFrame) -> pd.Series:
        """Fitted coefficients against feature names — the cheapest interpretability there is."""
        names = self._matrix(X).columns
        return pd.Series(self.pipeline_["ridge"].coef_, index=names).sort_values(key=np.abs, ascending=False)

    def get_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha}
