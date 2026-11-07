"""Level normalisation: express the score-scale columns relative to a target-free patient scale.

The observations are multiplicative views of the latent score, so a learner that sees every score as
a fraction of the patient's own level has the generator's (approximate) scale symmetry built in
instead of having to learn it at every level separately. ``s_p`` is the patient mean of the chain's
own posterior curve (``lik_pred``, ``env_pred`` where it is missing) — no target involved — floored at
3 points; a patient with neither curve gets 30. Every column on the score scale is divided by
``s_p``, the two log-level columns are shifted by ``−log s_p``, ratios / log-slopes / times /
covariates are left alone, and ``s_p`` itself is appended (``level_scale``) so level-dependent
departures from the symmetry stay learnable.

Used by :class:`~pdchallenge.models.gbm.GBMModel` (``level_norm: true``: fit ``y / s_p`` with sample
weight ``s_p²`` — exactly the squared error on the score scale — and predict ``s_p · f``) and by the
set models (``level_norm: true``). Ported unchanged from the probe that measured it
(`reports/paradigm-2026-10-08/levelnorm.py`; evidence `reports/setnn-2026-10-09/README.md`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import PATIENT

SCORE_EXACT = frozenset({
    "on", "off", "pseudo_target", "bias_off_hat", "bias_on_hat", "ldr_hat", "curve_a", "curve_b",
    "curve_pred", "curve_anchor_value", "curve_pred_anchored", "on_eq", "obs", "on_eq_pk", "obs_pk",
    "obs_max_p", "obs_q75_p", "obs_mean_p", "off_max_p", "env_pred", "obs_minus_env", "off_minus_env",
    "lik_pred", "obs_minus_lik",
})
SCORE_PREFIX = ("obs_minus_lik_", "obs_minus_env_", "obs_pk_prev", "obs_pk_next", "fl_", "edge_cf_",
                "edge_lik_minus")
SCORE_EXCLUDE = frozenset({"fl_floor1_age_gap"})  # a time, not a score
LOG_LEVEL = frozenset({"env_level_log", "lik_level_log"})
SCALE_FLOOR = 3.0
SCALE_MISSING = 30.0


def patient_scale(frame: pd.DataFrame) -> np.ndarray:
    """Per row: max(patient mean of ``lik_pred`` (``env_pred`` where missing), 3); 30 if neither exists."""
    lp = frame["lik_pred"].to_numpy(dtype=float) if "lik_pred" in frame else np.full(len(frame), np.nan)
    if "env_pred" in frame:
        lp = np.where(np.isfinite(lp), lp, frame["env_pred"].to_numpy(dtype=float))
    s = pd.Series(lp).groupby(frame[PATIENT].to_numpy()).transform("mean").to_numpy()
    return np.maximum(np.nan_to_num(s, nan=SCALE_MISSING), SCALE_FLOOR)


def normalise(M: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """``M`` (a design matrix) with its score-scale columns divided by ``s`` and ``level_scale`` added."""
    out = M.copy()
    for c in out.columns:
        if c in SCORE_EXCLUDE:
            continue
        if c in SCORE_EXACT or c.startswith(SCORE_PREFIX):
            out[c] = out[c].to_numpy(dtype=float) / s
        elif c in LOG_LEVEL:
            out[c] = out[c].to_numpy(dtype=float) - np.log(s)
    out["level_scale"] = s
    return out
