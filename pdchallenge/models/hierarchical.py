"""The trajectory model: each patient's own shrunk curve, with a learned correction.

The heavy lifting happens upstream in
:class:`~pdchallenge.features.longitudinal.PatientCurveFeatures`, which fits and shrinks
one linear curve per patient and exposes it as ``curve_pred``. What is left for the model
is the residual calibration: the de-biasing that produced those curves is estimated from
aggregate bias, so it leaves a small systematic offset that a two-parameter affine fit
(``y ≈ alpha + beta · curve_pred``) cleans up honestly on training data.

That keeps this model tiny and interpretable — ``beta`` near 1 with a small ``alpha``
means "the curve is right and we nudged the level"; a ``beta`` far from 1 would mean the
curve's *slope* is systematically off, which is a diagnosis worth seeing.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..evaluation.metrics import clip_to_valid
from .base import BaseModel
from .registry import register


@register
class HierarchicalCurveModel(BaseModel):
    """Affine calibration of the per-patient shrunk curve."""

    name = "hierarchical"

    def __init__(
        self,
        pred_col: str = "curve_pred",
        fallback_col: str = "pseudo_target",
        clip: bool = True,
    ) -> None:
        self.pred_col = pred_col
        self.fallback_col = fallback_col
        self.clip = clip

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        if self.pred_col in X:
            p = X[self.pred_col].to_numpy(dtype=float)
        elif self.fallback_col in X:
            p = X[self.fallback_col].to_numpy(dtype=float)
        else:
            raise KeyError(f"neither {self.pred_col!r} nor {self.fallback_col!r} in the frame")
        # a patient with no usable visit falls back to the population curve, which is
        # already stored, so NaN here would mean an upstream bug — be explicit about it
        if np.isnan(p).any():
            raise ValueError(f"{self.pred_col!r} contains NaN for {int(np.isnan(p).sum())} rows")
        return p

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        raw = self._raw(X)
        target = np.asarray(y, dtype=float)
        ok = ~np.isnan(target)
        if ok.sum() < 2:
            raise ValueError("not enough labelled rows to calibrate")

        x = raw[ok]
        t = target[ok]
        xbar, tbar = x.mean(), t.mean()
        sxx = float(((x - xbar) ** 2).sum())
        # With no spread in the curve there is no slope to learn; fall back to a shift.
        self.beta_ = float(((x - xbar) * (t - tbar)).sum() / sxx) if sxx > 1e-9 else 1.0
        self.alpha_ = float(tbar - self.beta_ * xbar)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.alpha_ + self.beta_ * self._raw(X)
        return clip_to_valid(pred) if self.clip else pred

    def get_params(self) -> dict[str, Any]:
        return {"alpha": round(self.alpha_, 3), "beta": round(self.beta_, 4)}
