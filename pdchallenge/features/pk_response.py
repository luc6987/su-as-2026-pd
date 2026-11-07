"""The per-patient levodopa response, on top of the population curve.

:class:`~pdchallenge.features.ratio.OnRatioTransformer` converts an ON score onto the OFF
scale by dividing by ``rho_hat = E[on/off | ledd, time_since_intake_on]`` — one curve for
the whole cohort. The data has a patient-level term on top of it, and until now it was
unused (``docs/techniques.md`` §10 item 3b). Splitting the ratio into

    log(on/off) = log rho_hat(ledd, dt) + [patient effect] + [visit noise]

on the 15,442 both-score visits with ``off >= 5`` (``reports/pk-response/e1_reliability.py``,
fold-wise ``rho_hat`` so no visit sees its own row) gives a residual ``e0`` with **SD 0.185,
of which 0.169 is between patients** — the opposite of the OFF side, where the per-visit
factor ``m_v`` is pure noise (split-half reliability 0.048, §2.3 of the diagnosis). The
split-half reliability of this patient mean is **0.841**, and it holds out of sample: an
amplitude estimated on a patient's odd visits cuts the residual on their even visits from
0.187 to 0.132 (−29%), and the conversion error at a both-score visit — ``sd(on/rho − off)``,
the quantity the envelope actually consumes — from 6.995 to 5.518 (−21%).

Measured, and the reason this transformer carries **one** random effect and not three:

* **amplitude** (the patient's mean residual, log scale) — reliability 0.841, worth the
  whole gain above;
* **a per-patient slope on ``time_since_intake_on``** — reliability 0.274, and a
  prior-shrunk per-patient ridge on it never beat the amplitude alone at any penalty
  (0.1310 at its best against 0.1289);
* **a per-patient slope on ``ledd``** — reliability −0.001, i.e. nothing. Even ridge-penalised
  it made the held-out residual 0.23–2.24 against 0.129.

So ``rho_hat_p = rho_hat · exp(a_p)``, with ``a_p`` the patient's mean ``e0`` shrunk by
``n/(n + k)`` toward the population. That is the "residual amplitude" random effect the
literature says is the identifiable half of the PK/PD pair (``reports/literature/evidence/pkpd-levodopa.md``
§药代动力学与浓度-效应): the full per-patient concentration curve is not identifiable, but
the amplitude is.

Two things worth knowing when reading a run that uses this:

* **``a_p`` is not predictable from covariates by a global regression** (OOF R² −0.04
  against dose, intake time, level, visit count, sex) — **but it is pinned by
  ``age_at_diagnosis`` locally** (next section). It is estimated from the patient's *own*
  both-score visits, which the test set carries as well — the same transductive-within-patient
  framing as the envelope curve, and the reason it helps here while the OFF-side "patient
  constant" helped nowhere.
* **``a_p`` is fitted with no target**, like the ratio itself, so ``fit`` ignores ``y``.

**The amplitude is pinned by (age_at_diagnosis, sex) — ``prior="aad_knn"``.** Two patients
with the same age at diagnosis (on its 0.1-year grid) and the same sex have amplitudes that
correlate at 0.72 (0.59 across sexes), falling to ~0 by a 0.7-year gap; nothing else about
the patient is pinned that way (T's level and shape, ``m``, dose, intake times, gene, cohort
all ≈ 0). A leave-one-out mean over same-sex neighbours within 0.1 year explains 78% of the
amplitude's latent between-patient variance on dev, and **75% for test patients against a
dev-only table** — the same map spans the released split (``docs/mining-2026-09-29.md``
§4.1, ``reports/mining-2026-09-29/pinning/amp_verify.py``). The global regression above could
not see it because the map is local at the 0.1-year scale. So the default shrink-to-zero is
the wrong prior for a patient with few both-score visits: ``prior="aad_knn"`` shrinks toward
the n-weighted mean of the neighbours' raw amplitudes instead, with that prior's own variance
``tau²`` measured leave-one-out on the fit frame (≈0.069² against the population's ≈0.147²),
and falls back to the population prior for a patient with no ``age_at_diagnosis`` or no
neighbour. Through the whole v3 chain on dev (single booster + smoothing, seeds 0/1/2):
6.9484 / 6.9005 / 6.8899 → 6.6290 / 6.6327 / 6.6012, pooled −0.2919 [−0.4017, −0.1785],
almost all of it on ON-only visits of patients with 0–1 both-score visits.
``knn_pool="fit+transform"`` also pools the transformed frame's own patients into the
neighbour table (target-free; a patient never counts for itself) — the test patients at
deployment, the validation fold in CV (seed 0: 6.6030).

**``prior="aad_krig"`` reads the same map with a kriging predictor instead of a window
mean** (`docs/mining-2026-10-02.md`, `reports/mining-2026-10-02/pinning-scan/`). The pair
covariance of the centred amplitudes falls off almost linearly in the age-at-diagnosis gap and is
shared, at a lower level, across the sexes, so the hard same-sex ±0.1-year window throws away
most of what the neighbourhood knows. The kriging prior fits that covariance on the fit frame
(a triangle in the gap, one level for the other sex and a higher one for the same sex, sex-specific
means, each neighbour's own sampling variance on the diagonal) and takes the best linear predictor
from every neighbour within ``krig_window`` grid steps (0.7 year), both sexes. Target-free, and
the transfer holds: latent amplitude variance explained 0.786 → 0.833 on dev (leave-one-out) and
0.764 → 0.800 for the test patients against a dev-only table (`pinning-scan/amp_blup.csv`).
``dt_slope=True`` adds the one visit-level term that is pinned the same way: a patient's ON
residual drifts with the intake time, and the pooled within-patient slope of the patient's
(age_at_diagnosis) neighbours — both sexes, ±0.2 year, the patient itself never counted —
explains ~8% of the within-patient variance of the log ratio on dev and on test alike
(`pinning-scan/dtslope.csv`). It enters as ``a_v = a_p + β·s_N(p)·(dt_on − mean dt_on of p)``,
``β`` calibrated leave-one-out on the fit frame; a visit with no intake time gets ``a_p``.

**``effect_loading`` / ``timing`` (task 02 of the 2026-10-03 pack, off by default) read the
patient term as phase-dependent — off, the transform is byte-identical to the chain above.**
The split-half criterion of the mining round (`reports/mining-2026-10-02/on-conversion/a3c_timing.py`)
reads a patient's residual as a *shift along the curve* rather than a pure level: with the
plateau loading ``ℓ_P = E[(1−ρ̂)/ρ̂]`` over dt ∈ [2, 4.5) h, ``effect_loading`` scales the
amplitude by ``λ_v = clip(((1−ρ̂_v)/ρ̂_v)/ℓ_P, 0.02, 1.4)`` (1 where the intake time is
missing), and ``timing`` adds a patient-level absorption delay ``δ_p`` that moves the visit by
``s_v = ∂ log ρ̂/∂dt`` (finite differences ±0.25 h; 0 where the intake time is missing). When
either is on, the per-patient posterior becomes a GLS on
``z_v = λ_v·a_p − s_v·δ_p + ε_v`` with the amplitude's usual prior (neighbours or the
population) and ``δ_p ~ N(0, timing_tau²)``; the per-visit ratio is
``rho_hat_p = clip(ρ̂ · exp(λ_v·a_p − s_v·δ_p))``. Both read only the two scores, like the
amplitude; ``s_v`` comes from a copy of the ratio step's booster refit inside ``fit`` and
asserted to reproduce the chain's ``rho_hat`` column, so the slope is taken off the curve
the chain actually uses.

``obs_pk`` (``off`` where present, else ``on_eq_pk``) is a drop-in replacement for ``obs``
and is consumed by ``patient_aggregates`` and ``envelope_curve`` through their column
parameters. ``obs_pk_weight`` carries the relative precision: at a both-score visit the two
views of the same ``T·m_v`` are paired, so their relative errors give the weight directly —
0.78 for a patient-corrected ``on_eq`` against 0.70 for the population one and 1.0 for
``off`` itself. That 0.78 is a real measurement and ``on_weight`` can be set to it, but it
does not survive contact with the model: see the parameter's comment.

**The gain is transductive, and the forecast pays for it.** In the main CV a visit's own
both-score visits are part of the amplitude it is converted with, which is the leaderboard's
own framing (test visits carry their scores too) and is how the envelope already works. In
the forecast framing — fit on complete trajectories, predict a later visit from a patient's
first k — the amplitude has to come from those k visits alone, and the same chain is *worse*
than no correction at all (k=3: 38.7 against 36.2 MSE; k=5: 42.6 against 41.5). The
amplitude does not drift along the trajectory (temporal split-half 0.897 against 0.902 for a
parity split, within-patient slope on time ≈ 0), so this is not a stale estimate; it is the
price of a quantity estimated from few visits being applied to many. Both numbers are in
`docs/pk-response-2026-09-23.md`.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT
from .base import Transformer
from .ratio import OBS, OBS_WEIGHT, RATIO_INPUTS, RHO_HAT

PK_AMP = "pk_amp_p"  # the patient's residual ON-response amplitude, shrunk (log ratio)
PK_AMP_RAW = "pk_amp_raw_p"  # the same before shrinkage, for introspection
PK_N = "pk_n_p"  # both-score visits behind the amplitude
PK_RHO = "rho_hat_p"  # the patient-specific on/off ratio for this visit
PK_ON_EQ = "on_eq_pk"  # the ON score divided by PK_RHO
PK_OBS = "obs_pk"  # one OFF-scale observation per visit, the patient-corrected `obs`
PK_DELAY = "pk_delay_p"  # the patient's absorption delay δ_p, hours (written when `timing`)
# Relative precision of `obs_pk`. At the default `on_weight` (0.5, the same value the ratio
# step uses) this column is numerically identical to `obs_weight` — which is why the booster
# gives it zero importance. It is emitted anyway so the chain states which weight goes with
# which observation, and so a config that sets `on_weight` moves the right column.
PK_WEIGHT = "obs_pk_weight"
# The (age_at_diagnosis, sex) neighbour prior of the amplitude — written only when
# `prior="aad_knn"`, so the shipped chain's columns do not change.
PK_AMP_PRIOR = "pk_amp_prior_p"

PK_COLS = [PK_AMP, PK_AMP_RAW, PK_N, PK_RHO, PK_ON_EQ, PK_OBS, PK_WEIGHT]

PRIORS = ("population", "aad_knn", "aad_krig")
NEIGHBOUR_PRIORS = ("aad_knn", "aad_krig")
KNN_POOLS = ("fit", "fit+transform")

# The kriging prior's covariance table: one entry per 0.1-year grid step of the gap, up to
# 1.5 years (the measured pair covariance is ~0 beyond 0.7). The triangle's width is searched
# over 0.2–1.5 year on the first ten steps, where the pair counts are large.
_KRIG_GMAX = 15
_KRIG_FIT_STEPS = 11

# Floors on the method-of-moments variances, so a cohort with no patient effect at all
# (the multiplicative test fixture is one) cannot divide by zero or flip the sign of `k`.
_MIN_SD_A = 0.02
_MIN_SD_E = 0.02
_MIN_FIT_ROWS = 40


def pk_amplitude(
    e0: pd.Series, patients: pd.Series, shrink_k: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-patient shrunk amplitude of the residual ``e0``.

    Returns ``(raw mean, shrunk amplitude, n)`` indexed by patient. The shrinkage is
    ``n/(n + k)``: at the fitted ``k = 0.55`` a patient with one both-score visit keeps
    about two thirds of their residual, one with five keeps nine tenths, and a patient with
    none is left on the population curve.
    """
    g = e0.groupby(patients, sort=False)
    raw = g.mean()
    n = g.size().astype(float)
    shrunk = raw * n / (n + shrink_k)
    return raw, shrunk, n


def shrink_constant(e0: pd.Series, patients: pd.Series) -> tuple[float, float, float]:
    """``k = var(visit noise) / var(patient effect)`` by method of moments.

    The variance of the observed patient means is the patient variance plus the mean
    sampling variance of a mean of ``n`` visits; subtracting the latter isolates the
    former. Both are read off the training fold, so ``k`` carries no constant that would
    have to be re-tuned when the cohort changes.
    """
    df = pd.DataFrame({"p": patients.to_numpy(), "e": e0.to_numpy()}).dropna()
    g = df.groupby("p", sort=False).e
    n = g.size().astype(float)
    means = g.mean()
    n_multi = n[n >= 2]
    if len(n_multi) < 5:
        return float("nan"), float("nan"), float("nan")
    within = df[df.p.isin(n_multi.index)].copy()
    within["dev"] = within.e - within.p.map(means)
    dof = float((n_multi - 1).sum())
    sd_e = float(np.sqrt((within.dev**2).sum() / dof))
    var_means = float(means.var(ddof=1))
    sd_a = float(np.sqrt(max(var_means - sd_e**2 * float((1.0 / n).mean()), 0.0)))
    sd_a = max(sd_a, _MIN_SD_A)
    sd_e = max(sd_e, _MIN_SD_E)
    return sd_e**2 / sd_a**2, sd_a, sd_e


def krig_kernel(grid: np.ndarray, sex: np.ndarray, value: np.ndarray) -> tuple[float, float, float]:
    """Fit the kriging prior's covariance: ``(W, A_other, A_diff)`` in grid steps.

    Pair cross-products of the centred amplitudes are averaged by (gap in 0.1-year steps, same
    sex or not); a triangle ``tri(g; W) = max(1 − g/W, 0)`` is then fitted to both curves with one
    shared width, ``c_other = A_other·tri`` and ``c_same = (A_other + A_diff)·tri``. A triangle is
    positive definite in one dimension, so every neighbourhood's covariance matrix can be solved.
    """
    order = np.argsort(grid, kind="stable")
    k, s, v = grid[order], sex[order], value[order]
    acc = np.zeros((_KRIG_GMAX + 1, 2))
    cnt = np.zeros((_KRIG_GMAX + 1, 2))
    for i in range(len(k)):
        lo = i + 1
        hi = np.searchsorted(k, k[i] + 10, side="right")
        if hi <= lo:
            continue
        gp = k[lo:hi] - k[i]
        ss = (s[lo:hi] == s[i]).astype(int)
        np.add.at(acc, (gp, ss), v[i] * v[lo:hi])
        np.add.at(cnt, (gp, ss), 1)
    cov = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)[:_KRIG_FIT_STEPS]
    g = np.arange(_KRIG_FIT_STEPS, dtype=float)
    best: tuple[float, float, float, float] | None = None
    for width in np.arange(2.0, 15.01, 0.25):
        t = np.clip(1 - g / width, 0, None)
        tt = float(t @ t)
        a_other = max(float(t @ cov[:, 0]) / tt, 0.0)
        a_same = max(float(t @ cov[:, 1]) / tt, a_other)
        sse = float(((cov[:, 1] - a_same * t) ** 2).sum() + ((cov[:, 0] - a_other * t) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(width), a_other, a_same - a_other)
    assert best is not None
    return best[1], best[2], best[3]


def apply_pk(
    df: pd.DataFrame,
    amplitude: np.ndarray | pd.Series,
    rho_floor: float,
    rho_ceiling: float,
    on_weight: float,
    conversion_sds: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Add the patient-corrected observation columns to a frame.

    Shared with the forecast path (``evaluation/forward.py``), which calls this with an
    amplitude carried from a patient's *known* visits rather than one estimated on the
    frame in hand — the same rule ``curve_pred_anchored`` follows.

    ``conversion_sds`` is ``(sd_known, sd_na)`` — the per-class conversion noise of
    ``log(on/off) − log(rho_hat)`` for intake-time-known vs -missing visits, fitted by
    :meth:`PatientPKResponse._fit_conversion_sds`. When given, an ON-derived observation
    whose own intake time is missing is downweighted by ``min(1, σ²_known/σ²_na)``:
    the measured pair (0.080, 0.144) turns the default 0.5 into ≈0.15 there, because the
    conversion is ~3× noisier without the intake time. A visit with an OFF score is
    unaffected (its weight is 1.0 either way), and so is a frame with no such visits.
    """
    out = df.copy()
    rho = np.clip(out[RHO_HAT].to_numpy(dtype=float) * np.exp(np.asarray(amplitude, dtype=float)),
                  rho_floor, rho_ceiling)
    on = out["on"].to_numpy(dtype=float)
    off = out["off"].to_numpy(dtype=float)
    has_off = ~np.isnan(off)
    on_eq = on / rho
    out[PK_RHO] = rho
    out[PK_ON_EQ] = on_eq
    out[PK_OBS] = np.where(has_off, off, on_eq)
    out[PK_WEIGHT] = np.where(has_off, 1.0, on_weight)
    if conversion_sds is not None and "time_since_intake_on" in out.columns:
        sd_known, sd_na = conversion_sds
        if np.isfinite(sd_known) and np.isfinite(sd_na) and sd_known > 0.0 and sd_na > 0.0:
            scale = min(1.0, (sd_known / sd_na) ** 2)
            # Only rows converted from ON, and only those whose intake time is missing —
            # an OFF-only row has no intake time either but never enters this branch.
            dt_na = out["time_since_intake_on"].isna().to_numpy()
            out.loc[~has_off & dt_na, PK_WEIGHT] = on_weight * scale
    return out


class PatientPKResponse(Transformer):
    """One per-patient random effect on the ON curve: the residual response amplitude.

    ``fit`` needs ``rho_hat``, so ``on_ratio`` must run first. It reads no target — the
    amplitude comes from the ratio of a patient's own two scores — so unlike the envelope
    it can be fitted on any frame that carries the raw scores.
    """

    name = "pk_response"

    # Measured (tests/test_properties.py): `fit` ignores y; `transform` estimates each
    # patient's amplitude on the frame it is handed (and `knn_pool="fit+transform"` pools
    # that frame's patients), so a row's value depends on the patient's later visits — the
    # transductive framing the module docstring explains, and why `patient_level_cols`
    # is what the forecast carries.
    reads_target = False
    causal = False

    #: Columns the forecast validation must carry from a patient's known visits
    #: (`forward.py` skips any that the chain does not produce).
    patient_level_cols = [PK_AMP, PK_N, PK_AMP_PRIOR, PK_DELAY]

    def __init__(
        self,
        min_off: float = 5.0,
        shrink_k: float | None = None,
        on_weight: float = 0.5,
        rho_floor: float = 0.15,
        rho_ceiling: float = 1.5,
        prior: str = "population",
        knn_max_gap: float = 0.1,
        knn_same_sex: bool = True,
        knn_pool: str = "fit",
        krig_window: int = 7,
        dt_slope: bool = False,
        dt_tol: int = 2,
        dt_lambda: float = 1.0,
        conversion_weight: bool = False,
        effect_loading: bool = False,
        timing: bool = False,
        timing_tau: float = 0.4,
    ) -> None:
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, got {prior!r}")
        if knn_pool not in KNN_POOLS:
            raise ValueError(f"knn_pool must be one of {KNN_POOLS}, got {knn_pool!r}")
        if dt_slope and prior not in NEIGHBOUR_PRIORS:
            raise ValueError("dt_slope needs a neighbour prior ('aad_knn' or 'aad_krig')")
        if timing_tau <= 0.0:
            raise ValueError(f"timing_tau must be positive (it is the delay prior's SD), got {timing_tau!r}")
        # `aad_krig`: neighbours within `krig_window` grid steps of age at diagnosis (0.1 year
        # each), both sexes — 0.5, 0.7 and 1.0 year explain the same (0.833 / 0.831 / 0.829),
        # 0.3 year is too narrow for the fitted triangle and collapses.
        self.krig_window = krig_window
        # The neighbour dt slope: ±`dt_tol` grid steps, both sexes, a ridge `dt_lambda` on the
        # pooled Sxx so a neighbourhood with almost no spread in intake time says ~nothing.
        self.dt_slope = dt_slope
        self.dt_tol = dt_tol
        self.dt_lambda = dt_lambda
        # `population` shrinks toward 0 (the shipped behaviour, output unchanged);
        # `aad_knn` shrinks toward the (age_at_diagnosis, sex) neighbours — see the module
        # docstring. 0.1 year is the grid step of age_at_diagnosis, and the measured sweep
        # (0 / 0.1 / 0.15 / 0.25) peaks there; matching gene or cohort as well leaves too few
        # neighbours and explains less (62–74% against 78%, `pinning/knn_refine.py`).
        self.prior = prior
        self.knn_max_gap = knn_max_gap
        self.knn_same_sex = knn_same_sex
        self.knn_pool = knn_pool
        # Same floor as the ratio fit: below it on/off is dominated by the integer
        # rounding of a near-zero denominator.
        self.min_off = min_off
        # None -> method of moments on the training rows (see `shrink_constant`), which
        # gives 0.5475 on the real cohort; the odd/even held-out test prefers 0.2-1.0 and is
        # flat over it (0.1334 / 0.1320 / 0.1339), so the two agree.
        self.shrink_k = shrink_k
        # Relative precision of a corrected ON-derived observation against a real OFF one.
        # The paired within-visit measurement says 0.78 (0.70 for the uncorrected `on_eq`),
        # but end-to-end that is a tie on the main CV (10.5489 at 0.5 against 10.5478 at
        # 0.8) and *worse* on the forecast (k=3 38.7 against 39.0; k=5 42.6 against 45.1,
        # `reports/pk-response/e4_forward.py`), so the default is left where the ratio step
        # has it. The parameter stays because the measurement is real; it is just not
        # measurable through the model.
        self.on_weight = on_weight
        self.rho_floor = rho_floor
        self.rho_ceiling = rho_ceiling
        # Downweight an ON-derived observation whose intake time is missing, by the measured
        # per-class conversion noise (task 01 of the 2026-10-03 pack; the missing class is
        # ~3× noisier: SD 0.1441 against 0.0800, `reports/v6-dtna/logs/facts.log`). Off, the
        # weight column is exactly the shipped one — `apply_pk` takes no conversion SDs.
        self.conversion_weight = conversion_weight
        # Task 02 of the 2026-10-03 pack (`reports/v6-timing/`): the per-visit effect loading
        # λ_v and the patient-level absorption delay δ_p of the module docstring. Both default
        # off, and off means the shipped transform bit for bit (the flags only add branches).
        self.effect_loading = effect_loading
        self.timing = timing
        self.timing_tau = timing_tau

    # -- fit ------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        needed = [PATIENT, "on", "off", RHO_HAT]
        if self.conversion_weight:
            needed.append("time_since_intake_on")
        if self.effect_loading or self.timing:
            needed.extend(RATIO_INPUTS)
        self.require(df, needed)
        both = df["on"].notna() & df["off"].notna() & (df["off"] >= self.min_off)
        n_both = int(both.sum())
        if n_both < _MIN_FIT_ROWS:
            raise ValueError(
                f"only {n_both} visits carry both scores with off >= {self.min_off}; "
                "cannot estimate a patient-level response"
            )
        e0 = (np.log(df.loc[both, "on"] / df.loc[both, "off"])
              - np.log(df.loc[both, RHO_HAT]))
        self.n_fit_ = n_both
        self.sd_e_, self.sd_a_ = float("nan"), float("nan")
        if self.shrink_k is None:
            k, sd_a, sd_e = shrink_constant(e0, df.loc[both, PATIENT])
            if not np.isfinite(k):
                # Too few patients with two both-score visits to separate the two
                # variances. Shrinking hard is the honest fallback: it leaves the
                # population curve in place rather than acting on noise.
                k = 10.0
            self.shrink_k_ = float(np.clip(k, 0.02, 50.0))
            self.sd_a_, self.sd_e_ = sd_a, sd_e
        else:
            self.shrink_k_ = float(self.shrink_k)
        self.shrink_prior_ = "method_of_moments" if self.shrink_k is None else "fixed"
        if self.effect_loading or self.timing:
            self._fit_rho_model(df, both)
        if self.prior in NEIGHBOUR_PRIORS:
            e0_full = pd.Series(np.nan, index=df.index, dtype=float)
            e0_full[both] = e0
            self._fit_knn(df, e0_full)
        elif self.effect_loading or self.timing:
            # `_fit_knn` normally leaves `sig2_`/`pop_var_` behind; the GLS needs them on
            # the population-prior path too (they are the amplitude prior's variance).
            sd_e = self.sd_e_
            if not np.isfinite(sd_e):  # a fixed `shrink_k` skips the moments; the prior needs them
                _, _, sd_e = shrink_constant(e0, df.loc[both, PATIENT])
            self.sig2_ = float(sd_e ** 2) if np.isfinite(sd_e) else float(self.shrink_k_ * _MIN_SD_A ** 2)
            self.pop_var_ = self.sig2_ / self.shrink_k_
        if self.conversion_weight:
            self._fit_conversion_sds(df, e0, both)
        return self

    def _fit_conversion_sds(self, df: pd.DataFrame, e0: pd.Series, both: np.ndarray) -> None:
        """Per-class conversion noise of ``log(on/off) − log(rho_hat)``, dt known vs missing.

        The patient mean is removed over *all* of a patient's both-score visits — the
        amplitude is shared across the two classes (patient-mean correlation 0.661,
        `reports/v6-dtna/logs/facts.log`) — then ``σ²_class = Σ dev² / Σ (1 − 1/n_p)`` over
        that class's visits, the unbiased within-patient variance with ``n_p`` the patient's
        both-score visit count. Patients with a single both-score visit carry no
        within-patient information and are left out, as in `shrink_constant`. A class with
        no visits in the fit frame gets NaN, which `apply_pk` reads as "no downweight".
        Target-free: ``e0`` comes from the ratio of a patient's own two scores.
        """
        d = pd.DataFrame({
            "p": df.loc[both, PATIENT].to_numpy(),
            "e": e0.to_numpy(dtype=float),
            "k": df.loc[both, "time_since_intake_on"].notna().to_numpy(),
        })
        g = d.groupby("p", sort=False)
        d = d.assign(dev=d["e"] - g["e"].transform("mean"), n_p=g["e"].transform("size"))
        d = d[d["n_p"] >= 2]
        self.sd_conv_known_ = float("nan")
        self.sd_conv_na_ = float("nan")
        for cls, attr in ((True, "sd_conv_known_"), (False, "sd_conv_na_")):
            m = d["k"].to_numpy() == cls
            if m.sum() == 0:
                continue
            num = float((d.loc[m, "dev"] ** 2).sum())
            den = float((1.0 - 1.0 / d.loc[m, "n_p"]).sum())
            setattr(self, attr, float(np.sqrt(max(num / den, 0.0))))

    # -- the two-effect GLS (`effect_loading` / `timing`, both off by default) ----------
    def _fit_rho_model(self, df: pd.DataFrame, both: np.ndarray) -> None:
        """Refit the ratio step's booster on the fit frame and measure the plateau loading.

        The GLS design reads the population curve twice — the loading ``λ_v`` from the
        plateau level ``ℓ_P = E[(1−ρ̂)/ρ̂]`` over dt ∈ [2, 4.5) h, and the delay sensitivity
        ``s_v`` from the curve's log-slope at the visit's intake time. Both are read off a
        copy of the ratio model refit here (same hyperparameters and rows, so it must
        reproduce the chain's ``rho_hat`` column on the fit rows); if it does not — the two
        steps' `min_off` disagree, say — the step raises instead of quietly designing
        against a different curve.
        """
        self.require(df, [*RATIO_INPUTS])
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = df.loc[both, RATIO_INPUTS].to_numpy(dtype=float)
        ratio = (df.loc[both, "on"] / df.loc[both, "off"]).to_numpy(dtype=float)
        self.rho_model_ = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
            random_state=0,
        ).fit(X, ratio)
        pred = np.clip(self.rho_model_.predict(X), self.rho_floor, self.rho_ceiling)
        shipped = df.loc[both, RHO_HAT].to_numpy(dtype=float)
        err = float(np.max(np.abs(pred - shipped)))
        if err > 1e-12:
            raise ValueError(
                f"the refit ratio model does not reproduce the chain's rho_hat column "
                f"(max |diff| = {err:.3g}); the two steps' min_off settings must agree"
            )
        dt = df.loc[both, "time_since_intake_on"].to_numpy(dtype=float)
        plateau = (dt >= 2.0) & (dt < 4.5)
        if not plateau.any():
            raise ValueError(
                "no both-score visit in the fit frame has an intake time in [2.0, 4.5) h; "
                "cannot measure the plateau loading"
            )
        self.ell_p_ = float(np.mean((1.0 - shipped[plateau]) / shipped[plateau]))

    def _visit_design(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per visit: the loading ``λ_v`` and the delay sensitivity ``s_v`` of the ON conversion.

        ``λ_v = clip(((1−ρ̂_v)/ρ̂_v)/ℓ_P, 0.02, 1.4)`` — the visit's own effect scale over the
        plateau's; 1 where the intake time is missing (no phase is known). ``s_v`` is the
        log-slope of the refit curve at the visit's (ledd, dt), finite differences ±0.25 h
        clipped at 0; 0 where the intake time is missing. Both read the shipped ``rho_hat``
        column and the refit model, never the target.
        """
        rho = df[RHO_HAT].to_numpy(dtype=float)
        lam = np.clip(((1.0 - rho) / rho) / self.ell_p_, 0.02, 1.4)
        dt = df["time_since_intake_on"].to_numpy(dtype=float)
        ok = np.isfinite(dt)
        lam[~ok] = 1.0
        s = np.zeros(len(df), dtype=float)
        if ok.any():
            ledd = df["ledd"].to_numpy(dtype=float)[ok]
            up = np.column_stack([ledd, dt[ok] + 0.25])
            dn = np.column_stack([ledd, np.maximum(dt[ok] - 0.25, 0.0)])
            lu = np.log(np.clip(self.rho_model_.predict(up), self.rho_floor, self.rho_ceiling))
            ld = np.log(np.clip(self.rho_model_.predict(dn), self.rho_floor, self.rho_ceiling))
            s[ok] = (lu - ld) / (up[:, 1] - dn[:, 1])
        return lam, s

    def visit_shift(self, frame: pd.DataFrame, amplitude: np.ndarray | pd.Series,
                    delay: np.ndarray | pd.Series) -> np.ndarray:
        """The per-visit log-ratio shift behind ``rho_hat_p``: ``λ_v·a_p − s_v·δ_p``.

        ``apply_pk`` takes the same shift ``transform`` computes from the full posterior;
        public so the forecast path (`evaluation/forward.py`) can rebuild a future visit's
        conversion from the carried patient level and the visit's own λ_v/s_v (the
        neighbour dt-slope term stays uncarried there, as before). The flags are respected:
        off, λ_v ≡ 1 and s_v ≡ 0, and this is exactly the bare amplitude.
        """
        lam, s = self._visit_design(frame)
        if not self.effect_loading:
            lam = np.ones_like(lam)
        if not self.timing:
            s = np.zeros_like(s)
        return lam * np.asarray(amplitude, dtype=float) - s * np.asarray(delay, dtype=float)

    # -- the (age_at_diagnosis, sex) neighbour prior ---------------------------------
    @staticmethod
    def _patient_table(df: pd.DataFrame, e0: pd.Series) -> pd.DataFrame:
        """Per patient: age at diagnosis, sex, raw mean residual and its visit count."""
        g = df.assign(_e0=e0).groupby(PATIENT, sort=False)
        return pd.DataFrame({
            "aad": g["age_at_diagnosis"].first(),
            "sex": g["sexM"].first(),
            "raw": g["_e0"].mean(),
            "n": g["_e0"].count().astype(float),
        })

    def _knn_prior(self, ref: pd.DataFrame, q: pd.DataFrame) -> pd.Series:
        """n-weighted mean raw amplitude of each query patient's neighbours in ``ref``.

        Neighbours: ``|Δ age_at_diagnosis| <= knn_max_gap`` and (with ``knn_same_sex``) the
        same sex. A patient never counts for itself — the ids are compared, so the training
        frame's own patients get a leave-one-out prior. NaN when there is no neighbour or no
        age at diagnosis. With ``prior="aad_krig"`` the kriging predictor is used instead.
        """
        if self.prior == "aad_krig":
            return self._krig_prior(ref, q)
        out = pd.Series(np.nan, index=q.index, dtype=float)
        if ref.empty or q.empty:
            return out
        w_all = (ref["n"] / (ref["n"] + self.shrink_k_)).to_numpy(dtype=float)
        if self.knn_same_sex:
            groups = [(s, (ref["sex"] == s).to_numpy()) for s in pd.unique(ref["sex"].dropna())]
        else:
            groups = [(None, np.ones(len(ref), dtype=bool))]
        eps = 1e-9  # aad sits on a 0.1 grid: keep a gap of exactly 0.1 inside
        for s, mask in groups:
            r = ref[mask]
            order = np.argsort(r["aad"].to_numpy(dtype=float), kind="stable")
            a = r["aad"].to_numpy(dtype=float)[order]
            v = r["raw"].to_numpy(dtype=float)[order]
            w = w_all[mask][order]
            ids = r.index.to_numpy()[order]
            qq = q if s is None else q[q["sex"] == s]
            qq = qq[qq["aad"].notna()]
            if qq.empty:
                continue
            qa = qq["aad"].to_numpy(dtype=float)
            lo = np.searchsorted(a, qa - self.knn_max_gap - eps, side="left")
            hi = np.searchsorted(a, qa + self.knn_max_gap + eps, side="right")
            vals = np.full(len(qq), np.nan)
            for i, (pid, left, right) in enumerate(zip(qq.index, lo, hi)):
                if right <= left:
                    continue
                keep = ids[left:right] != pid
                if not keep.any():
                    continue
                ww = w[left:right][keep]
                vals[i] = float(np.sum(ww * v[left:right][keep]) / np.sum(ww))
            out.loc[qq.index] = vals
        return out

    def _fit_knn(self, df: pd.DataFrame, e0: pd.Series) -> None:
        self.require(df, ["age_at_diagnosis", "sexM"])
        sd_e = self.sd_e_
        if not np.isfinite(sd_e):  # a fixed `shrink_k` skips the moments; the prior needs them
            _, _, sd_e = shrink_constant(e0.dropna(), df.loc[e0.notna(), PATIENT])
        # Visit-noise variance of a residual; the population prior's variance is set so that a
        # patient with no neighbour gets exactly the shipped n/(n + k) shrinkage.
        self.sig2_ = float(sd_e ** 2) if np.isfinite(sd_e) else float(self.shrink_k_ * _MIN_SD_A ** 2)
        self.pop_var_ = self.sig2_ / self.shrink_k_
        table = self._patient_table(df, e0)
        self.knn_ref_ = table[(table["n"] > 0) & table["aad"].notna()].copy()
        if self.prior == "aad_krig":
            self._fit_krig(self.knn_ref_)
        prior = self._knn_prior(self.knn_ref_, self.knn_ref_)
        both_ok = self.knn_ref_.assign(prior=prior).dropna(subset=["prior"])
        both_ok = both_ok[both_ok["n"] >= 2]
        # The prior's own variance, leave-one-out: what is left of a patient's mean after the
        # neighbours' mean, minus that patient mean's own sampling variance.
        if len(both_ok):
            resid = float(((both_ok["raw"] - both_ok["prior"]) ** 2).mean()
                          - (self.sig2_ / both_ok["n"]).mean())
            self.tau2_ = float(max(resid, _MIN_SD_A ** 2))
        else:
            self.tau2_ = self.pop_var_
        self.knn_coverage_ = float(prior.notna().mean()) if len(prior) else 0.0
        if self.dt_slope:
            self._fit_dt(df, e0)

    # -- the kriging variant (`prior="aad_krig"`) -------------------------------------
    def _fit_krig(self, ref: pd.DataFrame) -> None:
        """Sex-specific means, the latent variance and the triangle covariance, on ``ref``."""
        self.mu_sex_ = {int(sx): float(g["raw"].mean()) for sx, g in ref.groupby("sex")}
        v = ref["raw"].to_numpy(float) - ref["sex"].map(self.mu_sex_).to_numpy(float)
        var_lat = float(np.var(v) - np.mean(self.sig2_ / ref["n"].to_numpy(float)))
        width, a_other, a_diff = krig_kernel(np.round(ref["aad"].to_numpy(float) * 10).astype(int),
                                             ref["sex"].to_numpy(int), v)
        tri = np.clip(1 - np.arange(_KRIG_GMAX + 1, dtype=float) / width, 0, None)
        self.krig_cov_ = np.column_stack([a_other * tri, (a_other + a_diff) * tri])  # [:,0] other sex
        # The diagonal must dominate the same-sex covariance at gap 0, or two patients with the
        # same (aad, sex) would be treated as one.
        self.krig_var_lat_ = max(var_lat, 1.02 * float(self.krig_cov_[0, 1]), _MIN_SD_A ** 2)
        self.krig_kernel_ = (width, a_other, a_diff)

    def _krig_prior(self, ref: pd.DataFrame, q: pd.DataFrame) -> pd.Series:
        """Best linear predictor of each query patient's amplitude from its neighbours in ``ref``.

        Neighbours: every patient within ``krig_window`` grid steps of age at diagnosis, either
        sex, the query patient itself excluded (ids compared, as in the window mean). Each
        neighbour's own sampling variance ``σ²/n`` sits on the diagonal, so a neighbour with one
        both-score visit counts for less than one with ten.
        """
        out = pd.Series(np.nan, index=q.index, dtype=float)
        r = ref[ref["aad"].notna() & (ref["n"] > 0)]
        if r.empty or q.empty:
            return out
        rk = np.round(r["aad"].to_numpy(float) * 10).astype(int)
        order = np.argsort(rk, kind="stable")
        rk = rk[order]
        rs = r["sex"].to_numpy(int)[order]
        mu = np.array([self.mu_sex_.get(0, 0.0), self.mu_sex_.get(1, 0.0)])
        ra = r["raw"].to_numpy(float)[order] - mu[rs]
        rn = r["n"].to_numpy(float)[order]
        rid = r.index.to_numpy()[order]
        qk = np.round(q["aad"].to_numpy(float) * 10)
        qs = q["sex"].to_numpy()
        vals = np.full(len(q), np.nan)
        for i, (pid, kk, ss) in enumerate(zip(q.index, qk, qs)):
            if not np.isfinite(kk) or not np.isfinite(ss):
                continue
            lo = np.searchsorted(rk, kk - self.krig_window, side="left")
            hi = np.searchsorted(rk, kk + self.krig_window, side="right")
            idx = np.arange(lo, hi)
            idx = idx[rid[idx] != pid]
            if len(idx) == 0:
                continue
            gk = np.minimum(np.abs(rk[idx] - kk).astype(int), _KRIG_GMAX)
            ci = self.krig_cov_[gk, (rs[idx] == int(ss)).astype(int)]
            if not np.any(ci > 0):
                continue
            gap = np.minimum(np.abs(rk[idx][:, None] - rk[idx][None, :]), _KRIG_GMAX)
            same = (rs[idx][:, None] == rs[idx][None, :]).astype(int)
            cmat = self.krig_cov_[gap, same]
            cmat[np.diag_indices_from(cmat)] = self.krig_var_lat_ + self.sig2_ / rn[idx]
            w = np.linalg.solve(cmat, ci)
            vals[i] = mu[int(ss)] + float(w @ ra[idx])
        out[:] = vals
        return out

    # -- the neighbour dt slope (`dt_slope=True`) ---------------------------------------
    @staticmethod
    def _dt_moments(df: pd.DataFrame, e0: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per patient: within-patient Σ dev·dtc and Σ dtc², mean intake time, aad, sex."""
        d = pd.DataFrame({"p": df[PATIENT].to_numpy(), "e": e0.to_numpy(float),
                          "dt": df["time_since_intake_on"].to_numpy(float),
                          "aad": df["age_at_diagnosis"].to_numpy(float), "sex": df["sexM"].to_numpy(float)})
        d = d[np.isfinite(d["e"]) & np.isfinite(d["dt"])]
        g = d.groupby("p", sort=False)
        d["dtc"] = d["dt"] - g["dt"].transform("mean")
        d["dev"] = d["e"] - g["e"].transform("mean")
        m = d.assign(sxy=d["dev"] * d["dtc"], sxx=d["dtc"] ** 2).groupby("p", sort=False).agg(
            sxy=("sxy", "sum"), sxx=("sxx", "sum"), dtbar=("dt", "mean"), aad=("aad", "first"),
            sex=("sex", "first"))
        return m, d

    def _nbr_slope(self, ref: pd.DataFrame, q: pd.DataFrame) -> pd.Series:
        """Pooled within-patient slope of the ``ref`` patients within ±``dt_tol`` grid steps
        (both sexes), the query patient itself excluded."""
        r = ref[ref["aad"].notna()]
        rk = np.round(r["aad"].to_numpy(float) * 10).astype(int)
        order = np.argsort(rk, kind="stable")
        rk, sxy, sxx = rk[order], r["sxy"].to_numpy(float)[order], r["sxx"].to_numpy(float)[order]
        rid = r.index.to_numpy()[order]
        csy, csx = np.concatenate([[0.0], np.cumsum(sxy)]), np.concatenate([[0.0], np.cumsum(sxx)])
        pos = pd.Series(np.arange(len(rid)), index=rid)
        qk = np.round(q["aad"].to_numpy(float) * 10)
        out = np.full(len(q), np.nan)
        ok = np.isfinite(qk)
        lo = np.searchsorted(rk, qk[ok] - self.dt_tol, side="left")
        hi = np.searchsorted(rk, qk[ok] + self.dt_tol, side="right")
        sy, sx = csy[hi] - csy[lo], csx[hi] - csx[lo]
        self_pos = pos.reindex(q.index[ok]).to_numpy()
        has_self = np.isfinite(self_pos)
        sp = np.where(has_self, self_pos, 0).astype(int)
        sy = sy - np.where(has_self, sxy[sp], 0.0)
        sx = sx - np.where(has_self, sxx[sp], 0.0)
        out[ok] = sy / (sx + self.dt_lambda)
        return pd.Series(out, index=q.index)

    def _fit_dt(self, df: pd.DataFrame, e0: pd.Series) -> None:
        """The neighbour table and ``β``: how much of a neighbour slope a patient's own
        within-patient deviations follow, by least squares over the fit frame (leave-one-out
        slopes), clipped to [0, 1.5]."""
        m, d = self._dt_moments(df, e0)
        self.dt_ref_ = m
        self.dt_pop_ = float(d["dt"].mean()) if len(d) else 0.0
        s = self._nbr_slope(m, m)
        pred = d["p"].map(s).to_numpy(float) * d["dtc"].to_numpy(float)
        ok = np.isfinite(pred)
        num = float(np.sum(d["dev"].to_numpy(float)[ok] * pred[ok]))
        den = float(np.sum(pred[ok] ** 2))
        self.dt_beta_ = float(np.clip(num / den, 0.0, 1.5)) if den > 0 else 0.0

    def _dt_adjustment(self, out: pd.DataFrame, e0: pd.Series) -> np.ndarray:
        """``β·s_N(p)·(dt_on − mean dt_on of p)`` per visit; 0 where the intake time is missing."""
        m, _ = self._dt_moments(out, e0)
        pool = self.dt_ref_
        if self.knn_pool == "fit+transform":
            pool = pd.concat([pool, m[~m.index.isin(pool.index)]])
        pt = out.groupby(PATIENT, sort=False).agg(aad=("age_at_diagnosis", "first"), sex=("sexM", "first"))
        s = self._nbr_slope(pool, pt)
        dtbar = out[PATIENT].map(m["dtbar"]).fillna(self.dt_pop_).to_numpy(float)
        adj = self.dt_beta_ * out[PATIENT].map(s).to_numpy(float) * (
            out["time_since_intake_on"].to_numpy(float) - dtbar)
        return np.where(np.isfinite(adj), adj, 0.0)

    # -- transform ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = [PATIENT, "on", "off", RHO_HAT]
        if self.conversion_weight:
            needed.append("time_since_intake_on")
        if self.effect_loading or self.timing:
            needed.extend(RATIO_INPUTS)
        self.require(df, needed)
        out = df.copy()
        both = out["on"].notna() & out["off"].notna() & (out["off"] >= self.min_off)
        e0 = pd.Series(np.nan, index=out.index, dtype=float)
        e0[both] = (np.log(out.loc[both, "on"] / out.loc[both, "off"])
                    - np.log(out.loc[both, RHO_HAT]))

        if self.effect_loading or self.timing:
            shift, _ = self._transform_gls(out, e0)
        elif self.prior in NEIGHBOUR_PRIORS:
            shift = self._transform_knn(out, e0)
        else:
            # Fitted per patient on whatever frame is handed in — including the test set,
            # whose patients bring their own both-score visits. Patients with none are left
            # on the population curve rather than dropped.
            raw, shrunk, n = pk_amplitude(e0.dropna(), out.loc[e0.notna(), PATIENT], self.shrink_k_)
            shift = out[PATIENT].map(shrunk).fillna(0.0)
            out[PK_AMP] = shift
            out[PK_AMP_RAW] = out[PATIENT].map(raw).fillna(0.0)
            out[PK_N] = out[PATIENT].map(n).fillna(0.0)
        sds = None
        if self.conversion_weight:
            sds = (getattr(self, "sd_conv_known_", float("nan")),
                   getattr(self, "sd_conv_na_", float("nan")))
        return self.preserve_index(
            apply_pk(out, np.asarray(shift, dtype=float), self.rho_floor, self.rho_ceiling,
                     self.on_weight, sds),
            df,
        )

    def _neighbour_prior(self, table: pd.DataFrame) -> pd.Series:
        """Prior mean per patient in ``table`` from the neighbour pool: the fit frame plus,
        with ``knn_pool="fit+transform"``, the transformed frame's own patients (a patient
        never counts for itself)."""
        pool = self.knn_ref_
        if self.knn_pool == "fit+transform":
            extra = table[(table["n"] > 0) & table["aad"].notna() & ~table.index.isin(pool.index)]
            pool = pd.concat([pool, extra])
        return self._knn_prior(pool, table)

    def _transform_knn(self, out: pd.DataFrame, e0: pd.Series) -> pd.Series:
        """Posterior-mean amplitude under the neighbour prior; writes the amplitude columns.

        ``a_p = (raw·n/σ² + prior/τ²) / (n/σ² + 1/τ²)`` — a patient with no both-score visit
        gets the neighbours' amplitude, one with many keeps their own. Without a neighbour the
        prior is the population's (mean 0, variance σ²/k), which is exactly the shipped
        ``n/(n + k)``. ``PK_AMP_PRIOR`` is written after ``PK_N`` and before ``apply_pk`` —
        the column order is part of what ``colsample_bytree`` samples.
        """
        self.require(out, ["age_at_diagnosis", "sexM"])
        table = self._patient_table(out, e0)
        prior = self._neighbour_prior(table)
        has = prior.notna().to_numpy()
        pm = prior.fillna(0.0).to_numpy(dtype=float)
        pv = np.where(has, self.tau2_, self.pop_var_)
        n = table["n"].fillna(0.0).to_numpy(dtype=float)
        raw = table["raw"].fillna(0.0).to_numpy(dtype=float)
        post = pd.Series((raw * n / self.sig2_ + pm / pv) / (n / self.sig2_ + 1.0 / pv), index=table.index)
        amp = out[PATIENT].map(post).fillna(0.0)
        out[PK_AMP] = amp
        out[PK_AMP_RAW] = out[PATIENT].map(table["raw"]).fillna(0.0)
        out[PK_N] = out[PATIENT].map(table["n"]).fillna(0.0)
        out[PK_AMP_PRIOR] = out[PATIENT].map(prior)
        if self.dt_slope:
            # The visit-level term moves `rho_hat_p` and everything downstream of it, but not
            # `pk_amp_p`, which stays the patient's level.
            return amp + self._dt_adjustment(out, e0)
        return amp

    def _transform_gls(self, out: pd.DataFrame, e0: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Per-patient posterior of ``(a_p, δ_p)`` under the two-effect GLS; writes the columns.

        ``z_v = λ_v·a_p − s_v·δ_p + ε_v`` over the patient's both-score visits, ε ~ N(0, σ²),
        with the amplitude's usual prior (neighbours or the population) and
        ``δ_p ~ N(0, timing_tau²)``. ``timing=False`` pins δ_p ≡ 0 (the scalar posterior is
        the amplitude one); ``effect_loading=False`` pins λ_v ≡ 1. Returns the per-visit
        shift ``λ_v·a_p − s_v·δ_p`` (plus the neighbour dt slope when on) and the per-visit
        delay. ``PK_DELAY`` is written after ``PK_N`` (and the prior), before ``apply_pk``.
        """
        lam, s = self._visit_design(out)
        if not self.effect_loading:
            lam = np.ones_like(lam)
        rows = pd.DataFrame({"p": out[PATIENT].to_numpy(), "z": e0.to_numpy(dtype=float),
                             "lam": lam, "s": s})
        rows = rows[np.isfinite(rows["z"])]
        if self.prior in NEIGHBOUR_PRIORS:
            self.require(out, ["age_at_diagnosis", "sexM"])
            table = self._patient_table(out, e0)
            prior = self._neighbour_prior(table)
            has = prior.notna().to_numpy()
            pm = prior.fillna(0.0).to_numpy(dtype=float)
            pv = np.where(has, self.tau2_, self.pop_var_)
        else:
            g = e0.dropna().groupby(out.loc[e0.notna(), PATIENT], sort=False)
            table = pd.DataFrame({"raw": g.mean(), "n": g.size().astype(float)})
            pm = np.zeros(len(table), dtype=float)
            pv = np.full(len(table), self.pop_var_)
        pos = pd.Series(np.arange(len(table)), index=table.index)
        inv_sig2 = 1.0 / self.sig2_
        inv_tau_d2 = 1.0 / (self.timing_tau ** 2) if self.timing else 0.0
        a = pd.Series(np.nan, index=table.index, dtype=float)
        d = pd.Series(np.nan, index=table.index, dtype=float)
        for pid, grp in rows.groupby("p", sort=False):
            i = pos.at[pid]
            z_v = grp["z"].to_numpy(dtype=float)
            Z = np.column_stack([grp["lam"].to_numpy(dtype=float), -grp["s"].to_numpy(dtype=float)])
            gram = (Z.T @ Z) * inv_sig2
            rhs = Z.T @ z_v * inv_sig2
            # (ZᵀZ/σ² + P0) θ = Zᵀz/σ² + P0 θ0, P0 = diag(1/τ_a², 1/τ_d²), θ0 = (m_p, 0).
            if self.timing:
                A = gram + np.array([[1.0 / pv[i], 0.0], [0.0, inv_tau_d2]])
                th = np.linalg.solve(A, rhs + np.array([pm[i] / pv[i], 0.0]))
                a.iloc[i] = th[0]
                d.iloc[i] = th[1]
            else:
                a.iloc[i] = (rhs[0] + pm[i] / pv[i]) / (gram[0, 0] + 1.0 / pv[i])
                d.iloc[i] = 0.0
        no_rows = table["n"].fillna(0.0).to_numpy() == 0
        a.loc[no_rows] = pm[no_rows]
        d.loc[no_rows] = 0.0
        out[PK_AMP] = out[PATIENT].map(a).fillna(0.0)
        out[PK_AMP_RAW] = out[PATIENT].map(table["raw"]).fillna(0.0)
        out[PK_N] = out[PATIENT].map(table["n"]).fillna(0.0)
        if self.prior in NEIGHBOUR_PRIORS:
            out[PK_AMP_PRIOR] = out[PATIENT].map(prior)
        delay = out[PATIENT].map(d).fillna(0.0)
        if self.timing:
            out[PK_DELAY] = delay
        shift = lam * out[PK_AMP].to_numpy(dtype=float) - s * delay.to_numpy(dtype=float)
        if self.dt_slope:
            return shift + self._dt_adjustment(out, e0), delay.to_numpy(dtype=float)
        return shift, delay.to_numpy(dtype=float)

    # -- introspection --------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        params = {
            "n_fit": getattr(self, "n_fit_", 0),
            "shrink_k": round(self.shrink_k_, 4),
            "shrink_prior": self.shrink_prior_,
            "sd_a": round(getattr(self, "sd_a_", float("nan")), 4),
            "sd_e": round(getattr(self, "sd_e_", float("nan")), 4),
            "on_weight": self.on_weight,
            "min_off": self.min_off,
            "columns": [PK_AMP, PK_N, PK_RHO, PK_ON_EQ, PK_OBS, PK_WEIGHT],
        }
        if self.prior in NEIGHBOUR_PRIORS:
            params.update({
                "prior": self.prior,
                "knn_max_gap": self.knn_max_gap,
                "knn_same_sex": self.knn_same_sex,
                "knn_pool": self.knn_pool,
                "tau_knn": round(float(np.sqrt(getattr(self, "tau2_", float("nan")))), 4),
                "sd_pop": round(float(np.sqrt(getattr(self, "pop_var_", float("nan")))), 4),
                "sd_visit": round(float(np.sqrt(getattr(self, "sig2_", float("nan")))), 4),
                "knn_coverage": round(getattr(self, "knn_coverage_", float("nan")), 4),
                "columns": [PK_AMP, PK_N, PK_AMP_PRIOR, PK_RHO, PK_ON_EQ, PK_OBS, PK_WEIGHT],
            })
        if self.prior == "aad_krig" and hasattr(self, "krig_kernel_"):
            width, a_other, a_diff = self.krig_kernel_
            params.update({"krig_window": self.krig_window, "krig_width_years": round(width / 10, 3),
                           "krig_cov_other": round(a_other, 5), "krig_cov_same_extra": round(a_diff, 5),
                           "krig_var_lat": round(self.krig_var_lat_, 5)})
        if self.dt_slope:
            params.update({"dt_slope": True, "dt_tol": self.dt_tol, "dt_lambda": self.dt_lambda,
                           "dt_beta": round(getattr(self, "dt_beta_", float("nan")), 4)})
        if self.conversion_weight:
            params.update({"conversion_weight": True,
                           "sd_conv_known": round(getattr(self, "sd_conv_known_", float("nan")), 4),
                           "sd_conv_na": round(getattr(self, "sd_conv_na_", float("nan")), 4)})
        if self.effect_loading or self.timing:
            cols = [PK_AMP, PK_N, PK_RHO, PK_ON_EQ, PK_OBS, PK_WEIGHT]
            if self.timing:
                cols = [PK_AMP, PK_N, PK_DELAY, PK_RHO, PK_ON_EQ, PK_OBS, PK_WEIGHT]
            params.update({
                "effect_loading": self.effect_loading,
                "timing": self.timing,
                "timing_tau": self.timing_tau,
                "ell_P": round(getattr(self, "ell_p_", float("nan")), 4),
                "columns": cols,
            })
        return params


__all__ = [
    "PatientPKResponse",
    "apply_pk",
    "pk_amplitude",
    "shrink_constant",
    "PK_AMP",
    "PK_AMP_PRIOR",
    "PK_AMP_RAW",
    "PK_DELAY",
    "PRIORS",
    "KNN_POOLS",
    "PK_N",
    "PK_RHO",
    "PK_ON_EQ",
    "PK_OBS",
    "PK_WEIGHT",
    "PK_COLS",
    "OBS",
    "OBS_WEIGHT",
]
