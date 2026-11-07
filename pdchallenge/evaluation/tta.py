"""Scale test-time augmentation: predict a patient at several overall score scales and average.

The observations are multiplicative views of the latent score (``off ≈ T·m``, ``on ≈ T·m·ρ``), so a
patient whose two scores are all multiplied by ``s`` is, to a good approximation, the same patient
with ``s·T``. A booster is piecewise constant along that direction and is noticeably noisy on it:
transforming the raw scores at ``s ∈ {0.9, 0.95, 1, 1.05, 1.1}``, predicting, dividing by ``s`` and
averaging is bagging along the generator's own symmetry. On v6's chain and bag5 it is −0.10 MSE on
dev (3 CV seeds pooled), every one of 15 folds better (`reports/paradigm-2026-10-08/README.md` §4).

The scaled copies go through the **same fitted chain** as the real rows — nothing is refitted and no
target is read — so this is a pure prediction-time change. Which members of a model are averaged is
the model's own decision (:meth:`pdchallenge.models.base.BaseModel.predict_tta`): the boosters opt in
with ``tta: true``; the set-transformer does not (its own TTA measured worse).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

SCORE_COLS = ("off", "on")


def scale_scores(X: pd.DataFrame, s: float) -> pd.DataFrame:
    """``X`` with both scores multiplied by ``s`` and re-rounded to the integer grid.

    A recorded score stays recorded and a missing one stays missing; a positive score is never
    rounded down to 0 (a zero is a qualitatively different observation for the chain's ratio
    model), and the result stays on the scale's 0–132 range.
    """
    out = X.copy()
    for col in SCORE_COLS:
        if col not in out.columns:
            continue
        v = out[col].to_numpy(dtype=float)
        new = np.clip(np.round(v * s), 0, 132)
        new = np.where((v >= 1) & (new < 1), 1.0, new)
        out[col] = np.where(np.isnan(v), np.nan, new)
    return out


def check_scales(scales: Sequence[float] | None) -> list[float] | None:
    """Validate a scale list: positive, unique, and containing 1 (the unaugmented prediction)."""
    if scales is None:
        return None
    sc = [float(s) for s in scales]
    if not sc:
        return None
    if any(s <= 0 for s in sc):
        raise ValueError(f"tta scales must be positive, got {sc}")
    if len(set(sc)) != len(sc):
        raise ValueError(f"tta scales must be unique, got {sc}")
    if 1.0 not in sc:
        raise ValueError(f"tta scales must include 1.0 (the frame the rest of the run uses), got {sc}")
    return sc


def predict_with_tta(
    pipeline: Any,
    model: Any,
    X_raw: pd.DataFrame,
    scales: Sequence[float] | None,
    frame: pd.DataFrame | None = None,
) -> np.ndarray:
    """``model``'s prediction for ``X_raw``, scale-averaged when ``scales`` is given.

    ``frame`` is ``pipeline.transform(X_raw)`` if the caller already has it (the unscaled frame is
    the one every other output of a run is built from, so it is passed rather than recomputed).
    Without ``scales`` this is exactly ``model.predict(frame)``.
    """
    sc = check_scales(scales)
    frame = pipeline.transform(X_raw) if frame is None else frame
    if sc is None:
        return np.asarray(model.predict(frame), dtype=float)
    frames: dict[float, pd.DataFrame] = {}
    for s in sc:
        f = frame if s == 1.0 else pipeline.transform(scale_scores(X_raw, s))
        # `predict_tta` lines the scaled frames up with the unscaled one by position; the chain
        # must therefore keep the row order, which is checked here rather than assumed.
        if "Index" in f.columns and not np.array_equal(f["Index"].to_numpy(), frame["Index"].to_numpy()):
            raise AssertionError(f"the feature chain reordered rows at tta scale {s}")
        frames[s] = f
    return np.asarray(model.predict_tta(frames), dtype=float)
