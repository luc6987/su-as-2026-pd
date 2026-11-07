"""Calibrated running lower bounds on ``T``: per-visit conditional quantiles, carried forward.

`floor` bounds ``T`` with the patient's earlier OFF scores divided by a fixed edge of ``m``
(1.00 at the first visit, 1.05 after). The edge is not fixed. Measured against the true target
on dev (`reports/mining-2026-10-02/off-edge/a1_law.py`): the share of OFF visits with
``m = off/T > 1.1`` is 5–10% at middle visits and ~0–4% at the last one; an exact
``off = round(T)`` point mass sits on 37% of the first visits of four-visit patients and 9% of
twelve-visit ones; cohort B carries more of it; the edge narrows with ``T``; and on an ON-only
visit the converted observation's spread depends on the intake time, the dose, and how well the
patient's own response is pinned. A fixed divisor cannot follow any of that.

So this step learns the conditional law instead. For every visit with an observation
(``off`` where present, else ``on_eq_pk``, the ON score on the OFF scale), LightGBM quantile
models predict the ``a``-quantile of ``z = log((T + 1)/(obs + 1))`` from the visit's own
observable context — position (``visit_idx``, ``n_visits``, visits to the end), cohort, whether
the patient is treated, intake times, the OFF visit's rank and the gap since the previous one,
and on the ON side the dose, the response amplitude and its support — one model per quantile and
per observation kind. ``L_a = (obs + 1)·exp(q_a) − 1`` is then a per-visit ``a``-quantile of
``T``, and because ``T`` never decreases inside a patient, the running maximum of ``L_a`` over
this and earlier visits bounds ``T`` here at least as well. Six columns: the running maxima at
``a`` = 0.05, 0.25, 0.5 and ``lik_pred`` minus each.

**The training rows get cross-fitted values.** ``fit`` reads the target (the training fold's
only), and a quantile model that has seen a row's own target hands the booster a bound that is
tighter on its training rows than it can ever be on a new patient — the booster would learn to
trust it too much. ``fit`` therefore also computes the training frame's per-visit quantiles out
of fold (patient-grouped inner folds), and ``transform`` returns those whenever it is handed the
frame it was fitted on — recognised by its ``Index`` and patient columns, which is how
``FeaturePipeline.fit_transform`` calls it. Any other frame (a validation fold, ``X_test``, a
forecast's known visits) gets the model fitted on the whole training frame.

**Measured** (dev, single booster + smoothing, against v4; `reports/mining-2026-10-02/`):
pooled over CV seeds 0/1/2 **−0.2434 [−0.3257, −0.1661]**, 15/15 folds, both between- and
within-patient terms falling. Most of it comes from the ON-only visits' bounds (seed 0: −0.19 of
the −0.22), and a level-only version — the same quantile models without the position and
covariates — gives nothing: the gain is the conditioning, not the quantile idea.

What it is not: causal in the strict sense. The running maximum only looks backwards, but the
covariates ``n_visits``, visits-to-the-end and the patient's OFF-visit count describe the whole
trajectory, exactly as `derived`'s ``n_visits`` already does for every other step. The forecast
path rebuilds the columns from ``known visits + this row`` (`evaluation/forward.py`
``_rebuild_rowwise``), where those counts are the known ones.

**The ``timing`` option** (default off) appends four treatment-timing covariates — ``since_on``,
``years_since_on``, ``pre_on_n`` and ``on_start_rel`` (`timing_columns`): the position of the
patient's first ON-scored visit — to both kinds' quantile models. Like ``n_visits`` and
``from_end`` they describe the whole trajectory, so the step stays transductive, not causal.
Measured on the v6 chain at the booster layer (bag5, three seeds; mining −0.0448
[−0.0822, −0.0052], confirmation −0.0546; `reports/round-2026-10-12/C/README.md`, task 05's R1
section), which is what the v9C chain turns on. Off must leave this step's output bit-identical
to the shipped one (G3 of `docs/prompts-2026-10-12/00-RULES.md`).
"""
from __future__ import annotations

import hashlib
from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT
from .base import Transformer

EDGE_COLS = [
    "edge_cf_lo05",
    "edge_cf_lo25",
    "edge_cf_med",
    "edge_lik_minus_cf_lo05",
    "edge_lik_minus_cf_lo25",
    "edge_lik_minus_cf_med",
]
_QUANTILES = (0.05, 0.25, 0.5)
_KIND_OFF, _KIND_ON = 0, 1

# The covariates each observation kind's quantile models see. `lobs` is the observation itself
# (log1p); the ON side adds the raw ON score and the pieces of its conversion.
OFF_COVARIATES = ["lobs", "vidx", "nv", "from_end", "is_B", "ledd_na", "dt_off", "gap_prev_off",
                  "off_rank", "n_off_p", "age_rel", "has_on"]
ON_COVARIATES = ["lobs", "lon", "vidx", "nv", "from_end", "is_B", "ledd_na", "ledd", "dt_on", "pk_n_p",
                 "pk_amp_p", "rho_hat_p", "n_off_p", "age_rel"]
# The four columns ``timing=True`` appends to both lists, in this order (see `timing_columns`).
TIMING_COVARIATES = ["since_on", "years_since_on", "pre_on_n", "on_start_rel"]

_REQUIRED = [PATIENT, "visit_idx", "n_visits", "age", "cohort", "ledd", "time_since_intake_on",
             "time_since_intake_off", "on", "off", "on_eq_pk", "pk_n_p", "pk_amp_p", "rho_hat_p", "lik_pred"]
_PRODUCED_BY = {
    "visit_idx": "derived", "n_visits": "derived", "on_eq_pk": "pk_response", "pk_n_p": "pk_response",
    "pk_amp_p": "pk_response", "rho_hat_p": "pk_response", "lik_pred": "likelihood",
}


def visit_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Observable per-visit context. ``df`` must be sorted by patient, then time."""
    pid = df[PATIENT]
    nv = df["n_visits"].to_numpy(dtype=float)
    vidx = df["visit_idx"].to_numpy(dtype=float)
    has_off = df["off"].notna()
    off_rank = (has_off.groupby(pid, sort=False).cumsum() - has_off.astype(int)).to_numpy(dtype=float)
    last_off_age = df["age"].where(has_off).groupby(pid, sort=False).transform(lambda s: s.ffill().shift(1))
    first_age = df.groupby(pid, sort=False)["age"].transform("min")
    n_off_p = has_off.groupby(pid, sort=False).transform("sum").to_numpy(dtype=float)
    off = df["off"].to_numpy(dtype=float)
    on_eq = df["on_eq_pk"].to_numpy(dtype=float)
    obs = np.where(np.isfinite(off), off, on_eq)
    cov = pd.DataFrame({
        "vidx": vidx,
        "nv": nv,
        "from_end": nv - 1 - vidx,
        "is_B": (df["cohort"].astype(str) == "B").to_numpy(dtype=float),
        "ledd_na": df["ledd"].isna().to_numpy(dtype=float),
        "ledd": df["ledd"].to_numpy(dtype=float),
        "dt_off": df["time_since_intake_off"].to_numpy(dtype=float),
        "dt_on": df["time_since_intake_on"].to_numpy(dtype=float),
        "gap_prev_off": (df["age"] - last_off_age).to_numpy(dtype=float),
        "off_rank": off_rank,
        "n_off_p": n_off_p,
        "age_rel": (df["age"] - first_age).to_numpy(dtype=float),
        "has_on": df["on"].notna().to_numpy(dtype=float),
    }, index=df.index)
    cov["lobs"] = np.log1p(np.maximum(obs, 0.0))
    cov["lon"] = np.log1p(np.maximum(df["on"].to_numpy(dtype=float), 0.0))
    for c in ("pk_n_p", "pk_amp_p", "rho_hat_p"):
        cov[c] = df[c].to_numpy(dtype=float)
    cov["_obs"] = obs
    cov["_kind"] = np.where(np.isfinite(off), _KIND_OFF, _KIND_ON)
    return cov


def timing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The four treatment-timing covariates of ``timing=True``, from the patient's first ON visit.

    ``since_on`` is ``visit_idx`` relative to that visit, ``years_since_on`` the same in age,
    ``pre_on_n`` its ``visit_idx`` (how many visits came before it) and ``on_start_rel`` its age
    relative to the patient's first visit. A patient with no ON score at all gets NaN in all four.
    Copy of the R1 probe's ``timing_cols`` (`reports/round-2026-10-12/C/r1_timing.py`); the parity
    of the two on the cached frames is what `C/07_edge_parity.py` checks.
    """
    pid = df[PATIENT]
    vidx = df["visit_idx"].astype(float)
    age = df["age"].astype(float)
    has_on = df["on"].notna()
    first_on_v = vidx.where(has_on).groupby(pid).transform("min")
    first_on_age = age.where(has_on).groupby(pid).transform("min")
    first_age = age.groupby(pid).transform("min")
    return pd.DataFrame({"since_on": vidx - first_on_v, "years_since_on": age - first_on_age,
                         "pre_on_n": first_on_v, "on_start_rel": first_on_age - first_age}, index=df.index)


def _frame_key(df: pd.DataFrame) -> tuple[int, str]:
    """Row identity of a frame: its length and a digest of its ``Index`` and patient columns."""
    h = hashlib.sha1()
    for col in ("Index", PATIENT):
        if col in df.columns:
            h.update(pd.util.hash_pandas_object(df[col], index=False).to_numpy().tobytes())
    if "Index" not in df.columns:  # unit fixtures: fall back on the time column as well
        h.update(pd.util.hash_pandas_object(df["age"], index=False).to_numpy().tobytes())
    return len(df), h.hexdigest()


class ConditionalEdgeFeatures(Transformer):
    """Running maxima of per-visit conditional quantiles of ``T`` (see the module docstring)."""

    name = "edge"

    # Measured (tests/test_properties.py): `fit` reads the training fold's target, and the
    # step is *not* causal in the strict sense — the covariates describe the whole
    # trajectory, exactly as the module docstring says ("What it is not: causal...").
    reads_target = True
    causal = False

    def __init__(
        self,
        seed: int = 0,
        inner_folds: int = 4,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.9,
        n_jobs: int = 2,
        timing: bool = False,
    ) -> None:
        if inner_folds < 2:
            raise ValueError(f"inner_folds must be >= 2, got {inner_folds}")
        self.seed = seed
        # Patient-grouped folds for the training frame's own values; 4 keeps each inner model
        # on 75% of the training patients, close to the full model the other frames get.
        self.inner_folds = inner_folds
        # The treatment-timing covariates (`timing_columns`) appended to both kinds' model inputs.
        # Off by default, and bit-identical to the shipped step when off (G3 of 00-RULES).
        self.timing = timing
        # A small, heavily regularised booster per quantile: the conditional law has a dozen
        # inputs and tens of thousands of rows, and a bound that chases noise is worse than none.
        self.lgb_params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                               min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1,
                               colsample_bytree=colsample_bytree, n_jobs=n_jobs, verbose=-1)

    @property
    def columns(self) -> list[str]:
        """The new column names, in the order ``transform`` appends them."""
        return list(EDGE_COLS)

    # -- fit ------------------------------------------------------------------------------
    def _fit_models(self, cov: pd.DataFrame, y: np.ndarray) -> dict[tuple[int, float], Any]:
        from lightgbm import LGBMRegressor

        obs, kind = cov["_obs"].to_numpy(dtype=float), cov["_kind"].to_numpy()
        z = np.log((np.asarray(y, dtype=float) + 1.0) / (np.maximum(obs, 0.0) + 1.0))
        models: dict[tuple[int, float], Any] = {}
        for k, cols in ((_KIND_OFF, self._cols(_KIND_OFF)), (_KIND_ON, self._cols(_KIND_ON))):
            sel = (kind == k) & np.isfinite(obs)
            for a in _QUANTILES:
                m = LGBMRegressor(objective="quantile", alpha=a, random_state=self.seed, **self.lgb_params)
                m.fit(cov.loc[sel, cols].to_numpy(dtype=float), z[sel])
                models[(k, a)] = m
        return models

    def _per_visit(self, models: dict[tuple[int, float], Any], cov: pd.DataFrame) -> np.ndarray:
        """``L_a`` for each quantile, shape ``(rows, len(_QUANTILES))``; NaN without an observation."""
        obs, kind = cov["_obs"].to_numpy(dtype=float), cov["_kind"].to_numpy()
        out = np.full((len(cov), len(_QUANTILES)), np.nan)
        for j, a in enumerate(_QUANTILES):
            q = np.full(len(cov), np.nan)
            for k, cols in ((_KIND_OFF, self._cols(_KIND_OFF)), (_KIND_ON, self._cols(_KIND_ON))):
                sel = (kind == k) & np.isfinite(obs)
                if sel.any():
                    q[sel] = models[(k, a)].predict(cov.loc[sel, cols].to_numpy(dtype=float))
            out[:, j] = (np.maximum(obs, 0.0) + 1.0) * np.exp(q) - 1.0
        return out

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if y is None:
            raise ValueError("edge reads the training fold's target to fit its quantile models")
        self._require(df)
        from ..data.splits import group_kfold

        work, order = self._sorted(df)
        yv = np.asarray(y, dtype=float)[order]
        cov = self._covariates(work)
        self.models_ = self._fit_models(cov, yv)
        # Out-of-fold values for the training frame itself (see the module docstring).
        groups = work[PATIENT].to_numpy()
        oof = np.full((len(work), len(_QUANTILES)), np.nan)
        for itr, iva in group_kfold(groups, self.inner_folds, seed=100 + self.seed):
            inner = self._fit_models(cov.iloc[itr], yv[itr])
            oof[iva] = self._per_visit(inner, cov.iloc[iva])
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        self.fit_per_visit_ = oof[inverse]  # back in the caller's row order
        self.fit_key_ = _frame_key(df)
        self.n_fit_ = int(len(df))
        return self

    # -- transform ----------------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require(df)
        out = df.copy()
        work, order = self._sorted(df)
        if _frame_key(df) == getattr(self, "fit_key_", None):
            per_visit = self.fit_per_visit_[order]
        else:
            per_visit = self._per_visit(self.models_, self._covariates(work))
        pid = work[PATIENT].to_numpy()
        lik = work["lik_pred"].to_numpy(dtype=float)
        running = []
        for j in range(len(_QUANTILES)):
            s = pd.Series(per_visit[:, j], index=work.index)
            r = s.groupby(pid, sort=False).cummax().groupby(pid, sort=False).ffill()
            running.append(r.to_numpy(dtype=float))
        new = dict(zip(EDGE_COLS[:3], running))
        for name, r in zip(EDGE_COLS[3:], running):
            new[name] = lik - r
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        for name in EDGE_COLS:
            out[name] = new[name][inverse]
        return self.preserve_index(out, df)

    # -- helpers -----------------------------------------------------------------------------
    def _covariates(self, work: pd.DataFrame) -> pd.DataFrame:
        """`visit_covariates`, plus the four `TIMING_COVARIATES` appended when ``timing`` is on."""
        cov = visit_covariates(work)
        if self.timing:
            t = timing_columns(work)
            for c in TIMING_COVARIATES:
                cov[c] = t[c].to_numpy(dtype=float)
        return cov

    def _cols(self, kind: int) -> list[str]:
        """The covariate list one observation kind's models read; with ``timing``, the four extra
        columns last, in `TIMING_COVARIATES` order. Off it returns the shipped list unchanged."""
        cols = OFF_COVARIATES if kind == _KIND_OFF else ON_COVARIATES
        return cols + TIMING_COVARIATES if self.timing else cols

    @staticmethod
    def _sorted(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """``df`` ordered by patient (first appearance) then age, and the permutation used."""
        codes, _ = pd.factorize(df[PATIENT], sort=False)
        order = np.lexsort((np.arange(len(df)), df["age"].to_numpy(dtype=float), codes))
        return df.iloc[order], order

    def _require(self, df: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            named = [f"{c} (written by the `{_PRODUCED_BY[c]}` step)" if c in _PRODUCED_BY else c
                     for c in missing]
            raise KeyError(f"edge needs columns the chain has not produced yet: {named}. It reads the "
                           f"`pk_response` conversion and the `likelihood` curve, so it runs after them.")

    def get_params(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "inner_folds": self.inner_folds,
            "timing": self.timing,
            "quantiles": list(_QUANTILES),
            "lgb": {k: v for k, v in self.lgb_params.items() if k != "verbose"},
            "n_fit": getattr(self, "n_fit_", 0),
            "columns": list(EDGE_COLS),
        }


__all__ = ["ConditionalEdgeFeatures", "EDGE_COLS", "TIMING_COVARIATES", "timing_columns", "visit_covariates"]
