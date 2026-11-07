"""Cross-validation that rebuilds the whole chain inside every fold.

The feature pipeline is refitted per fold, not fitted once and reused. That matters here
because the de-biasing curves and the per-patient shrinkage prior are *learned from the
training target* — fitting them once on all the data would leak the validation patients'
targets into their own predictions, and would do so in the flattering direction.

Folds are patient-disjoint (see :mod:`pdchallenge.data.splits`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.splits import assert_disjoint, group_kfold
from ..features.pipeline import build_pipeline
from ..models.registry import build as build_model
from .metrics import decompose_mse, summarize
from .tta import predict_with_tta


@dataclass
class CVResult:
    """Out-of-fold predictions plus per-fold bookkeeping."""

    oof: np.ndarray
    fold_metrics: pd.DataFrame
    summary: dict[str, float]
    fold_params: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mse(self) -> float:
        return self.summary["mse"]


def cross_validate(
    feature_specs: list[Any],
    model_spec: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | pd.Series,
    n_splits: int = 5,
    seed: int = 0,
    verbose: bool = False,
    tta_scales: list[float] | None = None,
) -> CVResult:
    """Run grouped K-fold and return out-of-fold predictions for every row.

    ``tta_scales`` averages each validation prediction over input-score scales through the fold's
    own fitted chain (:mod:`.tta`); ``None`` is the plain prediction.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y, dtype=float))
    groups = np.asarray(groups)

    oof = np.full(len(X), np.nan)
    rows: list[dict[str, Any]] = []
    fold_params: list[dict[str, Any]] = []

    for fold, (train_idx, valid_idx) in enumerate(group_kfold(groups, n_splits, seed=seed)):
        assert_disjoint(groups, train_idx, valid_idx)

        pipeline = build_pipeline(feature_specs)
        X_tr = pipeline.fit_transform(X.iloc[train_idx], y.iloc[train_idx])
        X_va = pipeline.transform(X.iloc[valid_idx])

        model = build_model(model_spec)
        model.fit(X_tr, y.iloc[train_idx], groups=groups[train_idx])
        pred = predict_with_tta(pipeline, model, X.iloc[valid_idx], tta_scales, frame=X_va)
        oof[valid_idx] = pred

        s = summarize(y.iloc[valid_idx], pred)
        s["fold"] = fold
        s["n_patients"] = int(len(np.unique(groups[valid_idx])))
        rows.append(s)
        fold_params.append({"features": pipeline.get_params(), "model": model.get_params()})

        if verbose:
            print(f"  fold {fold}: n={s['n']:6d}  mse={s['mse']:.4f}  bias={s['bias']:+.3f}")

    summary = summarize(y.to_numpy(), oof)
    summary.update(decompose_mse(y.to_numpy(), oof, groups))
    return CVResult(
        oof=oof,
        fold_metrics=pd.DataFrame(rows),
        summary=summary,
        fold_params=fold_params,
    )


def fit_full(
    feature_specs: list[Any],
    model_spec: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | pd.Series | None = None,
) -> tuple[Any, Any]:
    """Fit the chain on all training rows, returning ``(pipeline, model)``.

    Used for the test-set predictions, once CV has told us the configuration is sound.

    The index normalisation mirrors :func:`cross_validate` on purpose. The two paths have to
    hand the model the same *kind* of frame or a difference between them shows up as a
    difference between the CV score and the deployed one — which is the one comparison the
    whole project is built on.
    """
    groups = np.asarray(groups) if groups is not None else None
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y, dtype=float))
    pipeline = build_pipeline(feature_specs)
    Xf = pipeline.fit_transform(X, y)
    model = build_model(model_spec)
    model.fit(Xf, y, groups=groups)
    return pipeline, model
