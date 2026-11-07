"""The observation model written as a likelihood, and the per-patient posterior it implies.

``envelope.py`` fits a *quantile* of the observations as a proxy for the one-sided gap, and
``aggregates.py`` hands the booster the upper end of each patient's observations as a proxy
for the same thing. This step does neither: it writes the generative model the diagnosis
measured (``docs/diagnosis-2026-09-18.md``) down as a likelihood and takes the posterior.

    T(t) = exp(a + b·u + c·u²) − k        u = t − (the patient's mean visit time)
    off  = T · m                          on = T · m · ρ(ledd, dt_on) · exp(ε)
    m ~ f(m)    empirical, one-sided: mean 0.82, SD 0.21, hard right edge near 1.1
    ε ~ N(0, σ_ρ²)   the ratio model's own error, measured out of fold

Three things follow from writing it this way rather than as a quantile fit:

* the gap is used through its **measured density**, not through a τ picked by hand. The hard
  right edge is the whole point — it is what makes a high observation informative and a low
  one nearly worthless — and a quantile loss only approximates that shape;
* a both-score visit is **one observation, not two**. The two scores share ``m_v``, so the
  visit is combined as ``sqrt(off · on / ρ̂)``: the ratio model's error on that combination is
  halved rather than doubled. (Averaging the two scores, which the earlier chain did, averages
  nothing away because the shared factor does not cancel.)
* the answer is a **posterior mean**, which is what an RMSE-scored target wants, and it comes
  with the level, slope and curvature it was averaged over — so the forecast validation can
  rebuild the curve at a future visit's time from the known visits alone.

Measured against the true target, one feature at a time, with the same two-parameter affine
read-out the diagnosis doc uses (``reports/likelihood-probe/p1_posterior.py``, all visits):

    env_pred   5.897 RMSE   slope 0.840   ← the quantile curve is still compressed
    lik_pred   5.731 RMSE   slope 0.986   ← essentially on scale already

The grid also yields the exact posterior mean of ``T`` (``E[T] = Σ w·T``), and that measured
*slightly worse* (5.795): the log-normal moment correction over-inflates at the top. So
``lik_pred`` is the curve at the posterior-mean parameters, which has the further advantage of
being exactly rebuildable in the forecast path.

Everything fitted here — the density of ``m``, the level prior, the shape priors, the ratio
model and its error — is fitted on the training fold only. ``transform`` needs no target: it
applies the fitted pieces to a patient's own observations, exactly like the envelope step.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT, TIME_SINCE_DIAGNOSIS
from .base import Transformer
from .envelope import fit_curve_priors
from .pk_response import pk_amplitude, shrink_constant
from .ratio import OBS, RATIO_INPUTS, RHO_HAT

LIK_PRED = "lik_pred"  # the posterior-mean curve at this visit, on the score scale
LIK_LEVEL_LOG = "lik_level_log"  # posterior mean of the log-scale level at the patient centre
LIK_SLOPE_LOG = "lik_slope_log"  # posterior mean of the log-scale slope (per year)
LIK_CURV_LOG = "lik_curv_log"  # posterior mean of the log-scale curvature (per year²)
LIK_TBAR = "lik_tbar"  # the patient's mean visit time — the curve is centred on it
LIK_N_OBS = "lik_n_obs"  # how many observations went into this patient's posterior
OBS_MINUS_LIK = "obs_minus_lik"  # the chain's OFF-scale observation minus the curve

# The patient-constant block: what the forecast path carries forward from the known visits
# and re-evaluates at a future visit's time, like ENV_PARAM_COLS.
LIK_PARAM_COLS = [LIK_LEVEL_LOG, LIK_SLOPE_LOG, LIK_CURV_LOG, LIK_TBAR, LIK_N_OBS]

# Observation kinds, in the order the per-visit density table is stacked.
KIND_OFF, KIND_ON, KIND_BOTH = 0, 1, 2
# The two extra tables `sigma_by_dt` stacks: the same ON and both-score combinations, but
# smeared with the noisier conversion SD measured on intake-time-missing visits.
KIND_ON_NA, KIND_BOTH_NA = 3, 4

# log m grid: m from 0.04 to 2.0. The measured density lives in [0.3, 1.1]; the tails are
# only ever reached by a mis-scaled candidate level, where the floor is what matters.
_DENSITY_LO = -3.2
_DENSITY_HI = 0.7
_DENSITY_STEP = 0.01
_DENSITY_SMOOTH = 2.0  # bins; absorbs the integer rounding of the scores
_DENSITY_FLOOR = 1e-4  # relative to the peak: keeps a zero score from killing the posterior
_OBS_FLOOR = 0.5  # an observation is clipped here before the log (scores are integers ≥ 0)
_T_FLOOR = 0.5  # and a candidate T is clipped here, so log T stays finite
_SD_FLOOR = 0.05  # floor on sigma_rho, so one lucky fold cannot claim a perfect ratio model
_MIN_OFF_FIT = 5.0  # below this the ratio on/off is dominated by the rounding of the denominator
_PRIOR_N = 1200  # points on the global log-level prior grid
_SILVERMAN = 0.9

# How `sigma_rho` is measured when the step reads a per-patient amplitude (`amp_col`).
AMP_SIGMA_IN_SAMPLE = "in_sample"
AMP_SIGMA_HELDOUT = "heldout"
AMP_SIGMA_CHOICES = (AMP_SIGMA_IN_SAMPLE, AMP_SIGMA_HELDOUT)
# The held-out path needs enough scored visits for an SD to mean anything; below this it
# keeps the in-sample value rather than reporting a number read off a handful of rows.
_MIN_HELDOUT_RESID = 20
# The same fallback `pk_response` uses when a half cannot separate the two variances: shrink
# hard, which is the honest direction — it leaves the population curve in place instead of
# acting on noise, and the residual it reports is the larger one.
_HELDOUT_SHRINK_FALLBACK = 10.0


def log_density_of_m(
    off: np.ndarray,
    target: np.ndarray,
    min_target: float,
    smooth: float = _DENSITY_SMOOTH,
) -> np.ndarray:
    """The density of ``log m``, ``m = off / T``, on the fixed log grid.

    Read off the rows where both scores are known. ``log m`` rather than ``m`` because the
    observation enters as ``log obs = log T + log m`` — a convolution in log space, and one
    where the Jacobian of the change of variable is the same for every candidate ``T`` and so
    cancels out of the posterior.
    """
    sel = np.isfinite(off) & np.isfinite(target) & (target >= min_target)
    m = off[sel] / target[sel]
    m = m[m > 0]
    if len(m) < 100:
        raise ValueError(f"only {len(m)} usable rows to estimate the density of m")
    edges = np.arange(_DENSITY_LO, _DENSITY_HI + _DENSITY_STEP, _DENSITY_STEP)
    h, _ = np.histogram(np.log(m), bins=edges, density=True)
    r = int(np.ceil(3 * smooth))
    xs = np.arange(-r, r + 1)
    ker = np.exp(-0.5 * (xs / smooth) ** 2)
    h = np.convolve(h, ker / ker.sum(), mode="same")
    return np.maximum(h, _DENSITY_FLOOR * h.max())


def smear_log_density(h: np.ndarray, sd: float) -> np.ndarray:
    """Convolve a log-scale density with ``N(0, sd²)`` — the ratio model's own error.

    An ON-only visit enters as ``on / ρ̂``, so its per-visit factor is ``m · exp(−ε)``; the
    both-score combination carries ``ε/2``. Convolving here is what stops the step from
    treating a ratio prediction as if it were exact.
    """
    if sd <= 0:
        return h
    nb = sd / _DENSITY_STEP
    r = int(np.ceil(4 * nb))
    xs = np.arange(-r, r + 1)
    ker = np.exp(-0.5 * (xs / nb) ** 2)
    return np.convolve(h, ker / ker.sum(), mode="same")


def heldout_sigma_rho(
    e0: np.ndarray | pd.Series,
    patients: np.ndarray | pd.Series,
    times: np.ndarray | pd.Series,
) -> float:
    """SD of the ratio residual on visits the per-patient amplitude was *not* estimated from.

    ``e0`` is ``log(on/off) − log(ρ̂)`` at the both-score visits and ``times`` are their visit
    times. Each patient's visits are split into an odd and an even half **in time order**, the
    shrunk amplitude is estimated on one half — with its own method-of-moments ``k``, computed
    on that half alone — and subtracted on the other half; then the two halves swap roles. So
    every visit is scored exactly once, by an amplitude that never saw it, and the pooled SD
    is what an ON observation actually carries when the amplitude applied to it came from a
    different set of visits.

    The in-sample alternative (:meth:`LikelihoodFeatures._fit_ratio`) subtracts an amplitude
    estimated on the very rows whose residual it measures, so it reads low by construction —
    the same gap ``reports/pk-response/e1_reliability.py`` measured on the real cohort
    (0.1867 → 0.1320). Which of the two is *right* is not a matter of principle: the
    leaderboard's framing is transductive within a patient, exactly like the in-sample number,
    while the forecast framing is not.

    Asked end to end, the wider sigma is worth **−0.017 / −0.027 / −0.044 MSE** against the
    in-sample arm on the shipped chain (dev CV seeds 0/1/2, better in all three, no interval
    excluding 0) — and *none* of it on the ON-only rows it was meant for (−0.001 / −0.010 /
    −0.068, intervals straddling). The consistent part is on the OFF-only slice. Against the
    chain that uses no amplitude at all the whole thing is a wash, so nothing here is shipped:
    `reports/amp-heldout-sigma/README.md`.

    Returns NaN when the frame cannot support the split (no visit left to score), which the
    caller reads as "keep the in-sample value".
    """
    d = pd.DataFrame(
        {
            "p": np.asarray(patients),
            "e": np.asarray(e0, dtype=float),
            "t": np.asarray(times, dtype=float),
        }
    ).dropna(subset=["e"])
    if d.empty:
        return float("nan")
    # Time order inside each patient. `pos` breaks ties and, because a missing time sorts
    # last, it is also what a patient with no time axis falls back to (the frame's own order).
    d = d.assign(pos=np.arange(len(d)))
    d = d.sort_values(["p", "t", "pos"], kind="mergesort", na_position="last")
    d = d.assign(slot=d.groupby("p", sort=False).cumcount().to_numpy() % 2)

    resid = []
    for est_slot in (0, 1):
        est, eva = d[d["slot"] == est_slot], d[d["slot"] != est_slot]
        if est.empty or eva.empty:
            continue
        k, _, _ = shrink_constant(est["e"], est["p"])
        if not np.isfinite(k):
            k = _HELDOUT_SHRINK_FALLBACK
        _, shrunk, _ = pk_amplitude(est["e"], est["p"], float(np.clip(k, 0.02, 50.0)))
        # A patient with no visit in the estimating half is left on the population curve,
        # exactly as `PatientPKResponse.transform` leaves a patient with no both-score visit.
        amp = eva["p"].map(shrunk).fillna(0.0).to_numpy(dtype=float)
        resid.append(eva["e"].to_numpy(dtype=float) - amp)
    if not resid:
        return float("nan")
    pooled = np.concatenate(resid)
    if len(pooled) < _MIN_HELDOUT_RESID:
        return float("nan")
    return float(pooled.std())


def log_kde(x: np.ndarray, samples: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian KDE in log space, normalised, evaluated at ``x`` (numerically stable)."""
    d = (np.asarray(x, dtype=float)[:, None] - np.asarray(samples, dtype=float)[None, :]) / bw
    peak = (-0.5 * d**2).max(axis=1)
    return peak + np.log(np.exp(-0.5 * d**2 - peak[:, None]).mean(axis=1)) - np.log(bw * np.sqrt(2 * np.pi))


def silverman_bandwidth(samples: np.ndarray) -> float:
    sd = float(np.std(samples, ddof=1))
    iqr = float(np.percentile(samples, 75) - np.percentile(samples, 25)) / 1.349
    return max(_SILVERMAN * min(sd, iqr) * len(samples) ** (-0.2), 1e-3)


def evaluate_likelihood(
    level: np.ndarray | float,
    slope: np.ndarray | float,
    curv: np.ndarray | float,
    tbar: np.ndarray | float,
    t: np.ndarray,
    k_offset: float,
) -> np.ndarray:
    """The posterior-mean curve at time ``t``, back on the score scale.

    The same contract as :func:`~pdchallenge.features.envelope.evaluate_envelope`: the
    forecast validation calls this with parameters carried from a patient's known visits and
    the *future* visit's ``t``, so a curve is never copied forward as a constant. A missing
    ``t`` evaluates at the patient's centre.
    """
    tbar = np.asarray(tbar, dtype=float)
    u = np.where(np.isnan(t), 0.0, np.asarray(t, dtype=float) - tbar)
    log_pred = (
        np.asarray(level, dtype=float)
        + np.asarray(slope, dtype=float) * u
        + np.asarray(curv, dtype=float) * u**2
    )
    return np.exp(log_pred) - k_offset


def _logsumexp_weights(ll: np.ndarray) -> np.ndarray:
    w = np.exp(ll - ll.max())
    return w / w.sum()


class LikelihoodFeatures(Transformer):
    """One grid posterior per patient over ``(level, slope, curvature)``.

    ``fit`` needs the target: the density of ``m``, the prior on the log-level and the priors
    on the shape are all read off the training fold's true trajectories. ``transform`` needs
    no target — it applies those fitted pieces to a patient's own observations, so it runs
    unchanged on the test set.
    """

    name = "likelihood"

    # Measured (tests/test_properties.py): `fit` reads y — the density of m, the level
    # prior and the shape priors all come from the training fold's true trajectories —
    # while `transform` takes each patient's posterior over the whole frame it is handed,
    # so a row's value depends on the patient's later visits (the forecast carries the
    # posterior-mean parameters, `LIK_PARAM_COLS`).
    reads_target = True
    causal = False

    def __init__(
        self,
        k_offset: float = 3.0,
        min_target_density: float = 10.0,
        min_off: float = _MIN_OFF_FIT,
        rho_floor: float = 0.15,
        rho_ceiling: float = 1.5,
        ratio_folds: int = 3,
        density_smooth: float = _DENSITY_SMOOTH,
        a_half: float = 0.7,
        n_a: int = 71,
        n_b: int = 11,
        n_c: int = 5,
        b_span: float = 3.5,
        c_span: float = 2.5,
        quadratic_min_obs: int = 4,
        curv_min_span: float | None = None,
        curv_span_ref: float | None = None,
        time_col: str = TIME_SINCE_DIAGNOSIS,
        seed: int = 0,
        amp_col: str | None = None,
        amp_sigma: str = AMP_SIGMA_IN_SAMPLE,
        sigma_by_dt: bool = False,
    ) -> None:
        # The log offset log(obs + k), the same one the envelope step uses: k > 0 keeps a
        # zero score finite and softens the low end, where the deficit is not a fraction of T.
        self.k_offset = k_offset
        # Below this target the ratio off/T is dominated by the integer rounding of a small
        # denominator, so those rows are left out of the density and of the shape priors.
        self.min_target_density = min_target_density
        self.min_off = min_off
        # The ratio model's predictions are bounded by its training range; the clip is a guard
        # against dividing by a ratio that would blow an observation up.
        self.rho_floor = rho_floor
        self.rho_ceiling = rho_ceiling
        # Folds used to measure the ratio model's own error. Fitted out of fold: an in-sample
        # residual would report a ratio model better than the one actually applied.
        self.ratio_folds = ratio_folds
        self.density_smooth = density_smooth
        # The log-level grid: +/- a_half around the patient's largest observation (a factor of
        # two either way), n_a points. The measured posterior width is ~0.07 in log, so 0.02
        # of resolution is several times finer than the thing being resolved — halving the
        # grid changes lik_pred by less than the 3rd decimal.
        self.a_half = a_half
        self.n_a, self.n_b, self.n_c = n_a, n_b, n_c
        # Grid spans for the shape, in prior SDs.
        self.b_span, self.c_span = b_span, c_span
        # Below this many observations the curvature is not identified and is held at the
        # prior mean — the same rule (and default) the envelope step uses. **The default of 4
        # is measured too low in the forecast framing**: a quadratic fitted to 4–5 visits both
        # widens `sd_c` (0.0244 against 0.0100 on the ≥6-visit pool) and puts a curvature on
        # the five-visit posteriors of `evaluation.forward`, which then carries it to a future
        # visit as `c·u²`. 6 on the forecast's k=5 costs 18.6 MSE less (53.77 → 35.17) and
        # *improves* the dev CV in all three seeds; see `reports/likelihood-extrapolation/`
        # §10 and the shipped `configs/candidates/lik_extrapolation.yaml`. The class default
        # stays 4 because the envelope step shares the rule and because a chain that wants 6
        # should say so — the candidate config does.
        self.quadratic_min_obs = quadratic_min_obs
        # The same decision, keyed on the patient's **time span** instead of a visit count.
        # A visit count is a proxy for "can this trajectory carry a quadratic?", and a poor
        # one: five visits inside two years pin a curvature no better than three spread over
        # eight. That proxy is what the forecast runs into — its evaluation patients are the
        # short-span ones by construction (median span over their first five visits 3.8 years
        # against 6.7 over the training trajectories), while the fitting side keeps full
        # trajectories, so a curvature the five visits cannot support is estimated and then
        # extrapolated over `u = t − lik_tbar` at a future visit, where it enters as `c·u²`
        # (task 04: 67% of the k=5 damage, `reports/likelihood-extrapolation/README.md` §3).
        # `curv_min_span` holds the curvature at zero below that span (the grid collapses to a
        # single point, exactly as it does below `quadratic_min_obs`), and `curv_span_ref`
        # narrows the prior width to `sd_c · min(1, span / curv_span_ref)` — the same quantity
        # `fit_curve_priors` inflates by reading `sd_c` off short-span quadratic fits.
        # Both are in **years** and are shift-invariant, so they mean the same thing on `age`
        # and on `time_since_diagnosis`. None (the default) leaves the step unchanged, bit for
        # bit — `tests/test_features_likelihood.py::test_the_span_knobs_are_off_by_default`
        # pins that against goldens recorded before they existed.
        self.curv_min_span = curv_min_span
        self.curv_span_ref = curv_span_ref
        self.time_col = time_col
        self.seed = seed
        # A per-patient log-amplitude already on the frame (`pk_amp_p` from `pk_response`).
        # When set, the ratio this step applies is `ρ̂ · exp(amp)` — the same patient-corrected
        # ratio `obs_pk` is built with — and `sigma_rho` is measured on the residual that is
        # left after that correction, so the ON-derived observations are smeared by the error
        # they actually carry. None keeps the population ratio (the step's original behaviour).
        self.amp_col = amp_col
        # Which residual `sigma_rho` is read off when `amp_col` is set. `"in_sample"` (the
        # default) subtracts the amplitude the frame already carries, which was estimated on
        # the very rows it is subtracted from — so it is a lower bound on the error an ON
        # observation really carries (0.100 on the dev training folds, against 0.188 for the
        # uncorrected residual). `"heldout"` re-estimates the amplitude on half of each
        # patient's both-score visits and scores the other half (`heldout_sigma_rho`), which
        # reads 0.144 there — 1.43×, and the direction `reports/pk-response/e1_reliability.py`
        # measured (0.1867 → 0.1320). It changes only `sigma_rho` — never ρ̂, the density of
        # `m`, or the amplitude itself — and with `amp_col` unset the two values coincide,
        # because there is no amplitude to hold out. Both are on the fitted step
        # (`sigma_rho_in_sample_` / `sigma_rho_heldout_`).
        if amp_sigma not in AMP_SIGMA_CHOICES:
            raise ValueError(
                f"amp_sigma must be one of {AMP_SIGMA_CHOICES}, got {amp_sigma!r}"
            )
        self.amp_sigma = amp_sigma
        # Task 01 of the 2026-10-03 pack: split the same OOF ratio residual by intake-time
        # known/missing and smear ON-derived observations with their own class's SD. The
        # missing class converts ~3× noisier (SD 0.144 against 0.080,
        # `reports/v6-dtna/logs/facts.log`), so its tables are much wider — an ON-only visit
        # with no intake time is then read as the weak observation it is. Off, `log_g_` is
        # the shipped three tables and the output is bit-identical.
        self.sigma_by_dt = sigma_by_dt

    # -- fit ------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.require(df, [PATIENT, self.time_col, "off", "on", OBS, *RATIO_INPUTS])
        if y is None:
            raise ValueError("LikelihoodFeatures needs the training target to set its priors")
        target = np.asarray(y, dtype=float)
        t_all = df[self.time_col].to_numpy(dtype=float)
        z_true = np.log(np.clip(target, 0.0, None) + self.k_offset)

        # 1. the density of the per-visit factor, from the rows where both scores are known
        off_all = df["off"].to_numpy(dtype=float)
        self.density_ = log_density_of_m(
            off_all, target, self.min_target_density, self.density_smooth
        )
        sel = np.isfinite(off_all) & np.isfinite(target) & (target >= self.min_target_density)
        m = off_all[sel] / target[sel]
        m = m[m > 0]
        self.m_mean_, self.m_sd_ = float(m.mean()), float(m.std())
        self.m_above_one_ = float((m > 1.0).mean())

        # 2. the prior on the log-level: a KDE of the per-patient mean of log(T + k). Built on
        # a global grid and interpolated per patient — evaluating the KDE against every
        # training patient inside the per-patient loop costs more than the posterior does.
        z_patient = pd.Series(z_true).groupby(df[PATIENT].to_numpy(), sort=False).mean()
        self.level_samples_ = z_patient.to_numpy(dtype=float)
        self.level_bw_ = silverman_bandwidth(self.level_samples_)
        self.prior_grid_ = np.linspace(
            np.log(self.k_offset), np.log(132.0 + self.k_offset) + 0.4, _PRIOR_N
        )
        self.log_prior_level_ = log_kde(self.prior_grid_, self.level_samples_, self.level_bw_)

        # 3. the shape priors, shared with the envelope step (fitted on the true trajectories)
        priors = fit_curve_priors(
            t_all, z_true, df.groupby(PATIENT, sort=False).indices.values(), self.quadratic_min_obs
        )
        self.mu_b_, self.sd_b_ = priors.mu_b, priors.sd_b
        self.mu_c_, self.sd_c_ = priors.mu_c, priors.sd_c
        self.slope_prior_, self.curv_prior_ = priors.slope_source, priors.curv_source
        self.n_slope_prior_, self.n_curv_prior_ = priors.n_slope, priors.n_curv

        # 4. the ratio model, and the spread of its error
        self._fit_ratio(df)
        if self.sigma_by_dt:
            # Five tables: OFF keeps the unsmeared density; ON-derived observations are
            # smeared by their own class's SD — ON-only by σ, the both-score combination by
            # σ/2 — and the intake-time-missing class gets the noisier pair (kinds 3/4).
            self.log_g_ = np.log(
                np.stack(
                    [
                        self.density_,
                        smear_log_density(self.density_, self.sigma_rho_known_),
                        smear_log_density(self.density_, self.sigma_rho_known_ / 2.0),
                        smear_log_density(self.density_, self.sigma_rho_na_),
                        smear_log_density(self.density_, self.sigma_rho_na_ / 2.0),
                    ]
                )
            )
        else:
            self.log_g_ = np.log(
                np.stack(
                    [
                        self.density_,
                        smear_log_density(self.density_, self.sigma_rho_),
                        smear_log_density(self.density_, self.sigma_rho_ / 2.0),
                    ]
                )
            )
        return self

    def _fit_ratio(self, df: pd.DataFrame) -> None:
        """``on/off ~ (ledd, time_since_intake_on)``, its out-of-fold error, and the full fit.

        The step fits its own ratio model rather than reading ``on_ratio``'s ``rho_hat``: the
        measurement of ``sigma_rho`` needs out-of-fold residuals, and pairing a ρ̂ from one
        model with an error measured on another would be incoherent. The cost is one extra
        booster fit per fold, against a step that works with or without the rest of the chain.

        With ``amp_col`` set, the amplitude already on the frame is subtracted from the
        residual before ``sigma_rho`` is read off it — and that amplitude was estimated on the
        same rows, so the resulting SD is a lower bound. ``amp_sigma="heldout"`` replaces it
        with :func:`heldout_sigma_rho`. Both are kept on the fitted step
        (``sigma_rho_in_sample_`` / ``sigma_rho_heldout_``) so which one was used is readable
        after the fact; ``sigma_rho_`` is the one that goes into the likelihood table.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import KFold

        both = (
            df["on"].notna() & df["off"].notna() & (df.loc[:, "off"] >= self.min_off)
        )
        Xr = df.loc[both, RATIO_INPUTS].to_numpy(dtype=float)
        ratio = (df.loc[both, "on"] / df.loc[both, "off"]).to_numpy(dtype=float)
        if len(ratio) < 50:
            raise ValueError(f"only {len(ratio)} visits carry both scores — cannot fit the ratio")
        kw = dict(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
            random_state=self.seed,
        )
        folds = max(2, int(self.ratio_folds))
        oof = np.empty(len(ratio))
        for tr, te in KFold(folds, shuffle=True, random_state=self.seed).split(Xr):
            oof[te] = HistGradientBoostingRegressor(**kw).fit(Xr[tr], ratio[tr]).predict(Xr[te])
        base = np.log(ratio) - np.log(np.clip(oof, self.rho_floor, self.rho_ceiling))
        eps = base
        if self.amp_col is not None:
            eps = eps - df.loc[both, self.amp_col].to_numpy(dtype=float)
        # What the amplitude-corrected observation's error is under the two readings; the
        # in-sample one is always computed, so a `"heldout"` fit can report both and fall
        # back to it when the frame is too small to split.
        self.sigma_rho_in_sample_ = max(float(eps.std()), _SD_FLOOR)
        self.sigma_rho_heldout_ = float("nan")
        if self.amp_sigma == AMP_SIGMA_HELDOUT and self.amp_col is not None:
            self.sigma_rho_heldout_ = heldout_sigma_rho(
                base, df.loc[both, PATIENT].to_numpy(), df.loc[both, self.time_col].to_numpy()
            )
        used = self.sigma_rho_in_sample_
        if np.isfinite(self.sigma_rho_heldout_):
            used = float(self.sigma_rho_heldout_)
        self.sigma_rho_ = max(used, _SD_FLOOR)
        self.sigma_rho_known_ = float("nan")
        self.sigma_rho_na_ = float("nan")
        if self.sigma_by_dt:
            # The same OOF residual split by intake-time known/missing (task 01). A class too
            # small for an SD to mean anything falls back to the global sigma — keeping the
            # shipped tables there rather than smearing with a number read off a handful of
            # rows.
            dt_known = df.loc[both, "time_since_intake_on"].notna().to_numpy()
            for attr, m in (("sigma_rho_known_", dt_known), ("sigma_rho_na_", ~dt_known)):
                setattr(self, attr, max(float(eps[m].std()), _SD_FLOOR)
                        if m.sum() >= 50 else float(self.sigma_rho_))
        self.ratio_model_ = HistGradientBoostingRegressor(**kw).fit(Xr, ratio)
        self.n_ratio_fit_ = int(len(ratio))
        self.ratio_mean_ = float(ratio.mean())

    # -- transform ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.require(df, [PATIENT, self.time_col, "off", "on", OBS, *RATIO_INPUTS])
        out = df.copy()
        n = len(out)

        rho = self.ratio_model_.predict(out[RATIO_INPUTS].to_numpy(dtype=float))
        if self.amp_col is not None:
            rho = rho * np.exp(out[self.amp_col].to_numpy(dtype=float))
        rho = np.clip(rho, self.rho_floor, self.rho_ceiling)
        off = out["off"].to_numpy(dtype=float)
        on = out["on"].to_numpy(dtype=float)
        has_off, has_on = ~np.isnan(off), ~np.isnan(on)
        both = has_off & has_on

        # One observation per visit. A both-score visit is the geometric mean of the two
        # OFF-equivalent scores: they share `m_v`, so this does not average the shared factor
        # away, and it halves the ratio model's error on the visit.
        on_eq = on / rho
        y_obs = np.where(both, np.sqrt(np.maximum(off, 0.0) * np.maximum(on_eq, 0.0)), np.where(has_off, off, on_eq))
        kind = np.where(both, KIND_BOTH, np.where(has_off, KIND_OFF, KIND_ON))
        if self.sigma_by_dt:
            # An intake-time-missing visit reads the noisier conversion tables: the ON-only
            # observation is smeared by σ_na, the both-score combination by σ_na/2. OFF visits
            # keep table 0 — their observation is not a conversion.
            dt_na = out["time_since_intake_on"].isna().to_numpy()
            kind = np.where(dt_na & (kind == KIND_BOTH), KIND_BOTH_NA,
                            np.where(dt_na & (kind == KIND_ON), KIND_ON_NA, kind))
        z_obs = np.log(np.maximum(y_obs, _OBS_FLOOR))

        valid_all = np.isfinite(z_obs)
        t_all = out[self.time_col].to_numpy(dtype=float)
        level = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        curv = np.full(n, np.nan)
        tbar = np.full(n, np.nan)
        n_obs = np.zeros(n, dtype=int)

        # `groupby().indices` is positional — index arrays, never `.loc`.
        for idx in out.groupby(PATIENT, sort=False).indices.values():
            t = t_all[idx]
            finite = np.isfinite(t)
            centre = float(t[finite].mean()) if finite.any() else 0.0
            u = (np.where(finite, t, centre) - centre).astype(np.float32)
            zo = z_obs[idx].astype(np.float32)

            ko = kind[idx].astype(np.intp)
            tbar[idx] = centre
            n_obs[idx] = int(valid_all[idx].sum())
            a, b, c = self._grids(u, zo)
            a_bar, b_bar, c_bar = self._posterior(a, b, c, u, zo, ko)
            level[idx], slope[idx], curv[idx] = a_bar, b_bar, c_bar

        out[LIK_LEVEL_LOG] = level
        out[LIK_SLOPE_LOG] = slope
        out[LIK_CURV_LOG] = curv
        out[LIK_TBAR] = tbar
        out[LIK_N_OBS] = n_obs
        out[LIK_PRED] = evaluate_likelihood(level, slope, curv, tbar, t_all, self.k_offset)
        # Against the chain's own OFF-scale observation rather than this step's internal one,
        # so the column is the exact analogue of `obs_minus_env` — and, more practically, so
        # the forecast path can rebuild it from a carried curve (evaluation/forward.py).
        out[OBS_MINUS_LIK] = out[OBS] - out[LIK_PRED]
        return self.preserve_index(out, df)

    def _sd_c_eff(self, span: float) -> float:
        """The curvature prior's width for a trajectory spanning ``span`` years.

        Off (``curv_span_ref is None``) this is the fitted ``sd_c`` unchanged.
        """
        if self.curv_span_ref is None:
            return self.sd_c_
        return self.sd_c_ * min(1.0, span / self.curv_span_ref)

    def _curv_open(self, span: float) -> bool:
        """Whether a trajectory spanning ``span`` years may carry a curvature at all."""
        return self.curv_min_span is None or span >= self.curv_min_span

    def _grids(self, u: np.ndarray, z_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The per-patient parameter grid.

        The level grid is centred on the patient's largest observation, de-trended by the
        prior slope so that "centre" means the level at the patient's own mean visit time.
        Since ``obs ≤ 1.1·T``, the largest observation is a lower anchor for the level; the
        half-width of 0.7 in log covers a factor of two either way around it.
        """
        # A patient with no usable observation at all still gets a grid — around the
        # population's level, since the posterior is then the prior and the data says nothing.
        if np.isfinite(z_obs).any():
            u_at_max = float(u[int(np.nanargmax(z_obs))])
            # `a` is log(T + k) at the centre, so the anchor from an observation has to add k
            # back before taking the log — otherwise the grid sits ~0.07 low at a score of 40.
            a0 = float(np.log(np.exp(np.nanmax(z_obs)) + self.k_offset)) - self.mu_b_ * u_at_max
        else:
            a0 = float(np.mean(self.level_samples_))
        a = np.linspace(a0 - self.a_half, a0 + self.a_half, self.n_a, dtype=np.float32)
        b = np.linspace(
            self.mu_b_ - self.b_span * self.sd_b_,
            self.mu_b_ + self.b_span * self.sd_b_,
            self.n_b,
            dtype=np.float32,
        )
        span = float(np.ptp(u))
        flat = bool(span < 1e-6)
        if len(z_obs) >= self.quadratic_min_obs and not flat and self._curv_open(span):
            sd_c = self._sd_c_eff(span)
            c = np.linspace(
                self.mu_c_ - self.c_span * sd_c,
                self.mu_c_ + self.c_span * sd_c,
                self.n_c,
                dtype=np.float32,
            )
        else:
            c = np.zeros(1, dtype=np.float32)
        return a, b, c

    def _posterior(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        u: np.ndarray,
        z_obs: np.ndarray,
        kind: np.ndarray,
    ) -> tuple[float, float, float]:
        """Posterior mean of ``(a, b, c)`` under the likelihood and the priors.

        A visit enters as ``log obs = log T + log m_eff``, so its log-likelihood is the
        log-density of ``log m_eff`` evaluated at ``log obs − log T``: one gather on the fixed
        log grid, summed over the patient's visits. The gather is nearest-neighbour rather than
        interpolated — at a 0.01 step, and with the density already smoothed over two bins,
        interpolating changes ``lik_pred`` by less than the third decimal.

        The curvature prior is the one place the span enters here: its width is
        :meth:`_sd_c_eff` of this patient's span, matching the grid :meth:`_grids` built. With
        the grid collapsed to a single point the term is skipped, as before.
        """
        # (n_a, n_b, n_c, n_visits) — every visit, so a visit whose own observation is
        # unusable still gets a prediction from the rest of the patient's trajectory.
        z = (
            a[:, None, None, None]
            + b[None, :, None, None] * u[None, None, None, :]
            + c[None, None, :, None] * u[None, None, None, :] ** 2
        )
        t_cand = np.maximum(np.exp(z) - self.k_offset, _T_FLOOR)
        # A visit with no usable score contributes *nothing* — not a floored density, which
        # would still pull, and not a NaN, which would index the grid out of bounds (every
        # real visit carries a score, but the forecast path and a hand-built frame can both
        # present one that does not).
        valid = np.isfinite(z_obs)
        arg = np.where(valid, z_obs, 0.0)[None, None, None, :] - np.log(t_cand)
        i0 = np.clip(
            (arg - _DENSITY_LO) * (1.0 / _DENSITY_STEP), 0, self.log_g_.shape[1] - 1
        ).astype(np.intp)
        # `log_g_` is indexed by observation kind, not by visit: the kind array broadcasts
        # against the grid axes, giving one density row per visit.
        ll = np.where(valid[None, None, None, :], self.log_g_[kind, i0], 0.0).sum(axis=3)

        ll = ll + np.interp(a, self.prior_grid_, self.log_prior_level_)[:, None, None]
        ll = ll + (-0.5 * ((b - self.mu_b_) / self.sd_b_) ** 2)[None, :, None]
        if len(c) > 1:
            sd_c = self._sd_c_eff(float(np.ptp(u)))
            ll = ll + (-0.5 * ((c - self.mu_c_) / sd_c) ** 2)[None, None, :]

        w = _logsumexp_weights(ll)
        return (
            float((w * a[:, None, None]).sum()),
            float((w * b[None, :, None]).sum()),
            float((w * c[None, None, :]).sum()),
        )

    # -- introspection --------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        params = {
            "m_mean": round(getattr(self, "m_mean_", float("nan")), 4),
            "m_sd": round(getattr(self, "m_sd_", float("nan")), 4),
            "m_above_one": round(getattr(self, "m_above_one_", float("nan")), 4),
            "sigma_rho": round(getattr(self, "sigma_rho_", float("nan")), 4),
            "sigma_rho_in_sample": round(getattr(self, "sigma_rho_in_sample_", float("nan")), 4),
            "sigma_rho_heldout": round(getattr(self, "sigma_rho_heldout_", float("nan")), 4),
            "ratio_mean": round(getattr(self, "ratio_mean_", float("nan")), 4),
            "n_ratio_fit": getattr(self, "n_ratio_fit_", 0),
            "level_bw": round(getattr(self, "level_bw_", float("nan")), 4),
            "n_level_prior": int(len(getattr(self, "level_samples_", []))),
            "mu_b": round(getattr(self, "mu_b_", float("nan")), 4),
            "sd_b": round(getattr(self, "sd_b_", float("nan")), 4),
            "mu_c": round(getattr(self, "mu_c_", float("nan")), 4),
            "sd_c": round(getattr(self, "sd_c_", float("nan")), 4),
            "slope_prior": getattr(self, "slope_prior_", "?"),
            "curv_prior": getattr(self, "curv_prior_", "?"),
            "k_offset": self.k_offset,
            "min_target_density": self.min_target_density,
            "min_off": self.min_off,
            "ratio_folds": self.ratio_folds,
            "a_half": self.a_half,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_c": self.n_c,
            "quadratic_min_obs": self.quadratic_min_obs,
            "curv_min_span": self.curv_min_span,
            "curv_span_ref": self.curv_span_ref,
            "time_col": self.time_col,
            "amp_col": self.amp_col,
            "amp_sigma": self.amp_sigma,
        }
        if self.sigma_by_dt:
            params.update({
                "sigma_by_dt": True,
                "sigma_rho_known": round(getattr(self, "sigma_rho_known_", float("nan")), 4),
                "sigma_rho_na": round(getattr(self, "sigma_rho_na_", float("nan")), 4),
            })
        return params
