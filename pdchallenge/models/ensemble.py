"""Combining models, with the leakage guarded against.

A stacking ensemble learns how much to trust each member from their *out-of-fold*
predictions. Those folds **must be patient-disjoint**, for the same reason the outer CV
is: a member's in-fold prediction for a patient it trained on is far too good, and a
meta-learner trained on such predictions will put almost all its weight on whichever
member overfits hardest. That failure is silent and flattering, so the fold logic here
reuses :func:`pdchallenge.data.splits.group_kfold` rather than rolling its own.

Because it subclasses :class:`BaseModel`, an ensemble is itself a model and can be a
member of another ensemble.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.splits import group_kfold
from ..evaluation.metrics import clip_to_valid
from .base import BaseModel
from .registry import build, register, spec_name  # re-exported: it used to live here


@register
class EnsembleModel(BaseModel):
    """Average or stack a list of member models."""

    name = "ensemble"

    METHODS = ("mean", "weighted", "ridge_stack", "nnls_stack")

    def __init__(
        self,
        members: list[Any],
        method: str = "ridge_stack",
        n_splits: int = 5,
        seed: int = 0,
        clip: bool = True,
        weights: list[float] | None = None,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {self.METHODS}")
        # `weighted` takes the weights as given — fixed in the config, chosen out of fold on the
        # dev mining patients (docs/evaluation-protocol.md §7) — rather than learning them here:
        # an in-fit stacker would re-choose them on every training fold.
        if method == "weighted":
            if weights is None or len(weights) != len(members):
                raise ValueError("method 'weighted' needs one weight per member")
            if any(w < 0 for w in weights):
                raise ValueError(f"weights must be non-negative, got {weights}")
        elif weights is not None:
            raise ValueError(f"weights are only used by method 'weighted', not {method!r}")
        self.members = list(members)
        self.method = method
        self.n_splits = n_splits
        self.seed = seed
        self.clip = clip
        self.fixed_weights = None if weights is None else [float(w) for w in weights]

    # -- fitting --------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        yv = np.asarray(y, dtype=float)
        # Positionally aligned with X, and handed straight to the members: a member that
        # does its own resampling (another ensemble, or `segmented`) needs the same
        # patient grouping this fit was given. Dropping it here degrades such a member
        # silently — it falls back to averaging fitted members — which is invisible in
        # the score and only shows up as a stacker weight that looks oddly even.
        g = np.asarray(groups) if groups is not None else None

        if self.method == "weighted":
            self.members_ = [build(spec).fit(X, yv, groups=g) for spec in self.members]
            self.weights_ = np.asarray(self.fixed_weights, dtype=float)
            self.stack_ = None
            return self

        if g is None or self.method == "mean":
            # No patient grouping available: fitting each member on its own in-fold
            # predictions is impossible to do safely, so fall back to a plain average
            # over members fitted on the full training set.
            self.members_ = [build(spec).fit(X, yv, groups=g) for spec in self.members]
            self.weights_ = np.full(len(self.members_), 1.0 / len(self.members_))
            self.stack_ = None
            return self

        oof = np.full((len(X), len(self.members)), np.nan)
        for k, spec in enumerate(self.members):
            for train_idx, valid_idx in group_kfold(g, self.n_splits, seed=self.seed):
                model = build(spec).fit(X.iloc[train_idx], yv[train_idx], groups=g[train_idx])
                oof[valid_idx, k] = model.predict(X.iloc[valid_idx])

        rows = ~np.isnan(oof).any(axis=1)
        if rows.sum() < 50:
            raise ValueError("not enough out-of-fold rows to fit a stacker")

        self.members_ = [build(spec).fit(X, yv, groups=g) for spec in self.members]
        self._fit_stack(oof[rows], yv[rows])
        return self

    def _fit_stack(self, Z: np.ndarray, y: np.ndarray) -> None:
        if self.method == "ridge_stack":
            from sklearn.linear_model import Ridge

            self.stack_ = Ridge(alpha=1.0, positive=False, fit_intercept=True).fit(Z, y)
        elif self.method == "nnls_stack":
            from scipy.optimize import nnls

            # weight members non-negatively; no intercept, so it cannot drift the level
            coef, _ = nnls(Z, y)
            self.stack_ = ("nnls", coef)
        else:
            raise ValueError(self.method)
        self.oof_ = Z

    # -- prediction -----------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = np.column_stack([m.predict(X) for m in self.members_])
        pred = self._combine(Z)
        return clip_to_valid(pred) if self.clip else pred

    def predict_tta(self, frames: dict[float, pd.DataFrame]) -> np.ndarray:
        """Each member decides for itself whether it averages over the scaled frames."""
        Z = np.column_stack([m.predict_tta(frames) for m in self.members_])
        pred = self._combine(Z)
        return clip_to_valid(pred) if self.clip else pred

    def _combine(self, Z: np.ndarray) -> np.ndarray:
        if self.stack_ is None:
            return Z @ self.weights_
        if isinstance(self.stack_, tuple):
            return Z @ self.stack_[1]
        return self.stack_.predict(Z)

    # -- introspection --------------------------------------------------------------
    def member_names(self) -> list[str]:
        return [spec_name(s) for s in self.members]

    def weights(self) -> pd.Series:
        """How much weight the stacker gave each member. The headline diagnostic:
        a member carrying ~0 weight is not contributing and should be dropped."""
        names = self.member_names()
        if self.stack_ is None:
            return pd.Series(self.weights_, index=names)
        if isinstance(self.stack_, tuple):
            return pd.Series(self.stack_[1], index=names)
        return pd.Series(self.stack_.coef_, index=names)

    def get_params(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "members": self.member_names(),
            "weights": {k: round(float(v), 4) for k, v in self.weights().items()},
        }
