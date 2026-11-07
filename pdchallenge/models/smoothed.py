"""Post-hoc per-patient smoothing, wrapped around any model.

The latent target is a smooth curve inside each patient — an exact cubic in age for 61-73%
of them (`reports/review-2026-09-26/generator_structure.py`) — while a booster predicts
each visit on its own and lets part of that visit's own observation noise through. The
observation model says how much noise there is to let through: ``off ≈ T · m_v`` with a
per-visit ``m_v`` of SD 0.21 that is pure noise (CLAUDE.md §5). Projecting a patient's
predictions onto a low-degree polynomial in a time column and blending the projection back
with the raw prediction removes that jitter without reading the target: it uses only
predictions and the time column, so it runs unchanged on the test set.

Measured on dev against the consolidated chain with a single booster
(`reports/review-2026-09-26/smoothing.py`): −0.176 / −0.177 / −0.173 MSE on seeds 0/1/2,
i.e. −0.175 pooled, and it moves the within-patient term only — a least-squares fit with an
intercept preserves each patient's mean by construction.

**It is a patient-transductive lever, and the forecast must not use it.** Smoothing a
patient's rows together is legitimate on the leaderboard (a test visit carries its own
scores, and so do the patient's other visits), but under the forecast framing the rows being
predicted are all *future* visits, and blending them would describe each one by the others'
predictions — which come from their own observations. :mod:`pdchallenge.evaluation.forward`
therefore calls :meth:`SmoothedModel.predict_raw`, so in that framing this wrapper *is* its
inner model. Nothing about the class changes there; the framing picks the method.

Configs that use it must set ``explain: false``: SHAP attributes a prediction to the matrix
that produced it, and a post-processing step outside the matrix has no features to attribute
to. See `reports/smoothed/README.md` for the measurements.

**Put it outermost.** Wrapping a model in `smoothed` is the supported form
(``{"name": "smoothed", "base": <any spec>}``); a `smoothed` *member inside an ensemble* is
not. `EnsembleModel.predict` calls each member's `predict`, so such a member would smooth
under the forecast framing too — where it must not — and nothing in that path can see the
framing to refuse. The nesting direction is the one that keeps the guarantee.

**``family='pinned_cubic'`` (off by default) smooths onto the cubic the data is made of.**
61–73% of patients carry an exact cubic trajectory, so instead of projecting each patient
with ≥ 4 visits onto a free quadratic, the projection is onto
``c0 + c1·u + c2·u² + f(c0, c1, c2)·u³``: four fixed-point iterations between a cubic term
the map ``f`` pins and the least-squares (c0, c1, c2) of what is left. ``f`` is a
`HistGradientBoostingRegressor` fitted **per CV fold on the training fold's** "exact-cubic"
patients — with ≥ 6 visits, a non-degenerate time span and a degree-3 fit whose residuals
stay ≤ 0.1 points — a criterion that reads the target but no scan label
(`reports/round-2026-10-12/planner/q1_pinned_cubic_v8.py`, measured as R6: D107, D125;
reweighted as P1 on v9: D138). It rides on top of a blend: 0.75 · projection + 0.25 · raw,
the same ``w`` as the quadratic family. The 2026-10-14 round measured it inside v10M at
−0.0458 MSE on all dev patients (−0.0387 mining, three seeds; `reports/round-2026-10-14/M/03_step1.txt`).
The quadratic family is untouched bit for bit: every option added here is off by default.

**``continuation='g2'`` (off by default; pinned family only) continues the fit the way the
generator does.** The generator's target ``T`` is the G2 continuation of the latent cubic ``c``:
while ``c`` rises, ``T`` follows it, and once ``c`` stops rising, each visit's ``T`` is the
previous one × 1.02 — ``T`` never falls (`reports/mining-2026-09-29/trajectory-generator/genlib.py`'s
``R_GEOM``; a frozen constant, not a knob). R6 projects onto ``c`` alone, so where the pinned cubic
turns down at the tail while the target keeps its 1.02-per-visit floor, the projection is pulled
below the target there. ``continuation='g2'`` runs that rule on the patient's *fitted* values, in
stable age order, before the blend: ``g_0 = max(x_0, 0)`` and ``g_j = x_j`` if ``x_j > g_{j-1}``
else ``1.02 · g_{j-1}``, sorted back to the row order. It can only raise the fit, so the smoothed
prediction rises row by row (before the clip). The probe that defines it is
`reports/round-2026-10-16/P/g2lib.py` (= the planner's `p2_g2_checks.py`); the planning screen read
−0.0373 MSE on the mining patients (D151), and this round's pre-registered package read (D155)
gives mining Δ −0.0373 [−0.0543, −0.0214] and confirm Δ −0.0355 [−0.0649, −0.0084] against v10
(`reports/round-2026-10-16/P/03_step.txt`). With the default ``'none'`` every v7–v10 output is
unchanged bit for bit.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..data.schema import PATIENT
from ..evaluation.metrics import clip_to_valid
from .base import BaseModel
from .registry import build, register, spec_name

# The time axis the projection is fitted on. `age` rather than `time_since_diagnosis`
# because 5.1% of rows have no `age_at_diagnosis` (and so no time since diagnosis at all)
# while every row has an age; the curve is a function of time and the two axes differ by a
# per-patient constant, which a polynomial of the same degree absorbs exactly.
DEFAULT_TIME_COL = "age"


def _smooth_patient(t: np.ndarray, pred: np.ndarray, deg: int, w: float) -> np.ndarray:
    """One patient's blended predictions: ``w · (polynomial fit) + (1 − w) · pred``.

    The fit is centred on the patient's own mean time (``u = t − mean(t)``), which is what
    makes the basis well conditioned — ages are ~65, so ``vander`` on the raw values would
    be a near-singular system at degree 2, and centring is exact rather than a
    reparameterisation only up to floating point.

    ``min(deg, n − 1)``: a patient with one visit gets degree 0, whose fit is that visit's
    own prediction, so the output is the input. Two visits get a straight line through both,
    which is again the input. Smoothing can therefore only ever start at three visits, which
    is the correct behaviour — with two points any curve through them is an interpolation.
    """
    k = min(int(deg), len(pred) - 1)
    u = t - t.mean()
    V = np.vander(u, k + 1)
    c, *_ = np.linalg.lstsq(V, pred, rcond=None)
    return w * (V @ c) + (1.0 - w) * pred


# --- the pinned cubic (family='pinned_cubic') -------------------------------------------
#
# Ported from the probe that defines R6, `reports/round-2026-10-12/planner/q1_pinned_cubic_v8.py`
# (kept verbatim in `reports/round-2026-10-14/M/flib.py` as `pin_*`): same criterion for an
# exact cubic, same map and settings, same four fixed-point iterations, same blend. The
# constants (0.75, 4 iterations, the regressor's hyper-parameters) come with that file.

_CUBIC_MIN_VISITS = 6
_CUBIC_TOL = 0.1
_CUBIC_MIN_PATIENTS = 50
_PIN_ITERS = 4

# The generator's G2 constant (`R_GEOM` in `reports/mining-2026-09-29/trajectory-generator/genlib.py`):
# once the cubic stops rising, the target multiplies the last value by this per visit. Frozen — it is
# the generated data's constant, not a lever, so it is not addressable from a config.
_G2_RATE = 1.02
_CONTINUATIONS = ("none", "g2")


def _centred(t: np.ndarray) -> np.ndarray:
    """A patient's time column centred on its own mean, with the incumbent's row-ordinal fallback.

    The centred age is what makes the cubic basis well conditioned (see `_smooth_patient`), and
    the fallback keeps a patient without a time axis smoothable instead of raising. The probe's
    `pin_centred` is this function to the line.
    """
    if np.isnan(t).any():
        t = np.arange(len(t), dtype=float)
    return t - t.mean()


def exact_cubics(frame: pd.DataFrame, y: np.ndarray | pd.Series, time_col: str) -> np.ndarray:
    """One row of ``(c0, c1, c2, c3)`` per patient whose target *is* an exact cubic in ``time_col``.

    A patient qualifies with ≥ 6 visits, a strictly nonzero time span, and a degree-3 fit whose
    largest absolute residual is ≤ 0.1 points (the target has one decimal, so an exact cubic sits
    far below that). The criterion reads the training target but no scan label, which is what lets
    it live in the package. Patients come in order of first appearance in ``frame`` (the same
    `pd.factorize(..., sort=False)` grouping `_smooth` uses), coefficients in the patient's
    centred-time basis ``u = t − mean(t)`` — the probe's `pin_exact_cubics` re-expressed on a frame.
    """
    pid = frame[PATIENT].to_numpy()
    t_all = frame[time_col].to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    codes = pd.factorize(pid, sort=False)[0]
    order = np.argsort(codes, kind="stable")
    rows = []
    for idx in np.split(order, np.flatnonzero(np.diff(codes[order])) + 1):
        if len(idx) < _CUBIC_MIN_VISITS:
            continue
        u = _centred(t_all[idx])
        if np.ptp(u) <= 1e-6:
            continue
        c = np.polyfit(u, yv[idx], 3)  # highest power first: [c3, c2, c1, c0]
        if np.max(np.abs(np.polyval(c, u) - yv[idx])) <= _CUBIC_TOL:
            c3, c2, c1, c0 = c
            rows.append((c0, c1, c2, c3))
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def fit_cubic_map(
    frame: pd.DataFrame, y: np.ndarray | pd.Series, time_col: str
) -> HistGradientBoostingRegressor:
    """The map ``c3 ≈ f(c0, c1, c2)``, fitted on the frame's exact-cubic patients (the probe's model
    and settings, written down rather than tuned).

    Fit it on the **training** rows only — the model's `fit` runs per CV fold, so the harness gives
    this exactly the probe's per-fold map. Fewer than 50 exact-cubic patients in the frame is the
    sign the criterion, the time column or the target is not what the caller meant, so it raises
    instead of fitting a map from too little data.
    """
    C = exact_cubics(frame, y, time_col)
    if len(C) < _CUBIC_MIN_PATIENTS:
        raise ValueError(
            f"only {len(C)} exact-cubic patients in the training frame; the cubic map needs at "
            f"least {_CUBIC_MIN_PATIENTS}"
        )
    f = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, min_samples_leaf=30, random_state=0
    )
    f.fit(C[:, :3], C[:, 3])
    return f


def _g2(x: np.ndarray, rate: float = _G2_RATE) -> np.ndarray:
    """The generator's G2 rule along one patient's visits (in visit order): T never falls.

    ``g[0] = max(x[0], 0)``; for ``j >= 1``, ``g[j] = x[j]`` if ``x[j] > g[j-1]`` else
    ``rate * g[j-1]``. Why: the generator's target is ``G2(c)`` — while the latent cubic ``c``
    rises, T follows it; once ``c`` stops rising, each visit's T is the previous one × 1.02. R6
    projects only onto ``c``, so a pinned-cubic fit that bends down at the tail is pulled below
    the target there; running this rule on the fit is what R7 does (`rate` is the generator's
    constant, exposed for tests only).
    """
    g = np.empty(len(x), dtype=float)
    if len(x) == 0:
        return g
    g[0] = max(float(x[0]), 0.0)
    for j in range(1, len(x)):
        g[j] = x[j] if x[j] > g[j - 1] else rate * g[j - 1]
    return g


def _pinned_patient(
    u: np.ndarray,
    pred: np.ndarray,
    f: HistGradientBoostingRegressor,
    w: float,
    iters: int = _PIN_ITERS,
    continuation: str = "none",
) -> np.ndarray:
    """One patient's pinned-cubic projection, blended with the raw prediction.

    ``u`` is the patient's centred time column and ``pred`` their current predictions. The probe's
    `pin_project` does this for every patient of a fold at once; here it is one patient, so the
    batch loop is a scalar — initialise ``c3 = 0``, then per iteration take the least-squares
    (c2, c1, c0) of ``pred − c3·u³`` and ask the map for the cubic term they pin; the projection is
    the cubic through that (c0, c1, c2, c3), returned as ``w · projection + (1 − w) · pred``.

    ``continuation='g2'`` runs the generator's G2 rule (R7, module docstring) on the projection
    before the blend, in stable ``u`` order — ``u`` is the centred age, so ordering by it is
    ordering by age, and the stable sort keeps equal ages in row order, as the probe does.
    """
    if continuation not in _CONTINUATIONS:
        raise ValueError(f"unknown continuation {continuation!r}; use one of {_CONTINUATIONS}")
    # The probe's projection operator: `pinv` of the (n, 3) Vandermonde [u², u, 1] — exact on a
    # cubic, least squares otherwise, and one matrix product per iteration (the probe's `pin_project`).
    P = np.linalg.pinv(np.vander(u, 3))  # rows: c2, c1, c0
    c3 = 0.0
    C = np.zeros(3)
    for _ in range(iters):
        C = P @ (pred - c3 * u ** 3)
        c3 = float(f.predict(C[np.newaxis, [2, 1, 0]])[0])
    fit = np.vander(u, 3) @ C + c3 * u ** 3
    if continuation == "g2":
        o = np.argsort(u, kind="stable")
        g = np.empty_like(fit)
        g[o] = _g2(fit[o])
        fit = g
    return w * fit + (1.0 - w) * pred


@register
class SmoothedModel(BaseModel):
    """Any model, with each patient's predictions projected onto a polynomial and blended.

    ``base`` is any model spec — a string, a dict, or a built model — so a smoothing layer
    can wrap an ensemble as easily as a single booster::

        {"name": "smoothed", "base": {"name": "gbm", "params": {"max_depth": 8}}}
    """

    name = "smoothed"

    def __init__(
        self,
        base: Any,
        deg: int = 2,
        w: float = 0.75,
        time_col: str = DEFAULT_TIME_COL,
        clip: bool = True,
        family: str = "quadratic",
        continuation: str = "none",
    ) -> None:
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {w}")
        if deg < 0:
            raise ValueError(f"deg must be >= 0, got {deg}")
        if family not in ("quadratic", "pinned_cubic"):
            raise ValueError(f"unknown smoothing family {family!r}; use 'quadratic' or 'pinned_cubic'")
        if continuation not in _CONTINUATIONS:
            raise ValueError(
                f"unknown continuation {continuation!r}; use one of 'none' or 'g2'"
            )
        if continuation != "none" and family != "pinned_cubic":
            # 'g2' runs on the pinned family's fitted values; there is no such object under the
            # quadratic family, and silently ignoring the option would run the wrong model.
            raise ValueError(
                f"continuation needs family='pinned_cubic', got family={family!r} "
                f"with continuation={continuation!r}"
            )
        if family == "pinned_cubic" and deg != 2:
            # The pinned family *is* a cubic whose quadratic part is the usual degree-2 projection;
            # a different `deg` would silently mean something else, so it is refused.
            raise ValueError(f"family='pinned_cubic' needs deg=2 (the projection's base degree), got {deg}")
        self.base = base
        self.deg = int(deg)
        self.w = float(w)
        self.time_col = time_col
        self.clip = clip
        # 'pinned_cubic' projects onto c0 + c1·u + c2·u² + f(c0, c1, c2)·u³ with the map fitted in
        # `fit`; the default 'quadratic' is the incumbent family, unchanged bit for bit.
        self.family = family
        # 'g2' is R7 (the generator's T-never-falls rule on the pinned fit); 'none' is the
        # incumbent path, unchanged bit for bit.
        self.continuation = continuation

    # -- fitting --------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        # `groups` is handed straight through: a nested model that resamples internally
        # (an ensemble, `segmented`) has to keep the outer patient grouping, and dropping it
        # here would degrade it silently into a plain average over fitted members.
        self.base_ = build(self.base).fit(X, y, groups=groups)
        if self.family == "pinned_cubic":
            # The harness calls `fit` once per CV fold on that fold's training rows, so this is the
            # probe's per-fold map: the validation fold's target is never read here, and `predict`
            # only reads the stored map.
            self.cubic_map_ = fit_cubic_map(X, y, self.time_col)
        return self

    # -- prediction -----------------------------------------------------------------
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """The inner model's own prediction, before any smoothing.

        Separate from :meth:`predict` because the forecast framing needs exactly this: see
        the module docstring and :mod:`pdchallenge.evaluation.forward`.
        """
        return np.asarray(self.base_.predict(X), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._smooth(X, self.predict_raw(X))

    def predict_tta(self, frames: dict[float, pd.DataFrame]) -> np.ndarray:
        """The inner model's scale-averaged prediction, smoothed **once** on the unscaled frame.

        Smoothing is a per-patient linear projection, so smoothing the average equals averaging
        the smoothed predictions up to the final clip; doing it once is cheaper. (The probe that
        measured v7 smoothed the bag and the set-transformer separately and then blended them —
        the same projection, so the two agree up to that clip; `reports/paradigm-2026-10-08/`.)
        """
        return self._smooth(frames[1.0], np.asarray(self.base_.predict_tta(frames), dtype=float))

    def _smooth(self, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        if self.w == 0.0 or len(X) == 0:
            return clip_to_valid(pred) if self.clip else pred

        if PATIENT not in X.columns:
            raise KeyError(
                f"{PATIENT!r} is not in the frame handed to {self.name!r}; smoothing is "
                "per patient, so it needs the column (the pipeline frame has it, the "
                "design matrix does not)"
            )
        if self.time_col not in X.columns:
            # A typo'd `time_col` must not degrade into the row-ordinal fallback for everyone:
            # that would score a plausible number against the wrong axis and say nothing.
            raise KeyError(
                f"time column {self.time_col!r} is not in the frame handed to {self.name!r}; "
                f"available: {sorted(map(str, X.columns))[:20]}…"
            )
        t_all = X[self.time_col].to_numpy(dtype=float)

        out = pred.astype(float, copy=True)
        # Patients in order of first appearance, rows in frame order inside each block (the
        # stable sort is what keeps a patient's rows in the order they were handed over —
        # the order the fallback below means, and the order a shuffled frame would change).
        codes, _ = pd.factorize(X[PATIENT].to_numpy(), sort=False)
        order = np.argsort(codes, kind="stable")
        for idx in np.split(order, np.flatnonzero(np.diff(codes[order])) + 1):
            t = t_all[idx]
            if np.isnan(t).any():
                # A patient without a fully observed time axis has no curve to project onto,
                # and `lstsq` on a NaN basis raises rather than degrading. Fall back to the
                # row ordinal so the output stays total — the visit *order* is the only
                # ordering left. Unlike the time path this is sensitive to row order, which
                # is worth knowing and is not reachable on this cohort: every row has an
                # `age`, and the 5.1% with no `time_since_diagnosis` are missing it for the
                # whole trajectory, but `envelope_curve`'s `time_col` is not this default.
                t = np.arange(len(idx), dtype=float)
            if self.family == "pinned_cubic":
                # The probe applies R6 only to patients with >= 4 visits; shorter series keep
                # their prediction (`pin_project`'s `out = r.copy()`).
                if len(idx) >= 4:
                    out[idx] = _pinned_patient(
                        t - t.mean(), pred[idx], self.cubic_map_, self.w,
                        continuation=self.continuation,
                    )
            else:
                out[idx] = _smooth_patient(t, pred[idx], self.deg, self.w)

        return clip_to_valid(out) if self.clip else out

    # -- introspection --------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        out = {
            "base": spec_name(self.base),
            "base_params": self.base_.get_params(),
            "deg": self.deg,
            "w": self.w,
            "time_col": self.time_col,
        }
        if self.family != "quadratic":
            out["family"] = self.family
        if self.continuation != "none":
            out["continuation"] = self.continuation
        return out
