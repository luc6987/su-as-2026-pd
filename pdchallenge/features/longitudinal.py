"""Per-patient trajectories: a shrunk linear curve through each patient's visits.

The target is, per patient, a smooth near-linear curve in time since diagnosis
(EDA: median slope 2.37 points/year, IQR 1.72–3.08). So each patient's whole trajectory
is described by two numbers, and the job is to estimate them from 4–12 noisy visits.

The problem is that 4 visits spanning ~2 years pin a slope down very loosely. The
literature is blunt about this: baseline variables explain only 9% of between-patient
slope variance out-of-sample, so individual slopes should be pulled hard toward the
population value rather than trusted. That is what :class:`PatientCurveFeatures` does —
a textbook empirical-Bayes / James–Stein shrinkage:

    b_shrunk = (b_hat / var_b + mu_b / sigma_b^2) / (1 / var_b + 1 / sigma_b^2)

where ``var_b`` is the *sampling* variance of that patient's OLS slope. With few visits
``var_b`` is large, the prior term dominates, and the patient gets the population slope;
with many visits the patient's own data wins. The prior itself is estimated from the
data by subtracting the average sampling noise from the observed spread of per-patient
slopes, so it is not a hand-picked constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT, TIME_SINCE_DIAGNOSIS
from .base import Transformer

CURVE_A = "curve_a"  # patient intercept (score at time_since_diagnosis = 0)
CURVE_B = "curve_b"  # patient slope (points per year, at time_since_diagnosis = 0)
CURVE_C = "curve_c"  # patient curvature (points per year²); 0 on the linear fallback
CURVE_PRED = "curve_pred"  # the curve evaluated at this visit
CURVE_N = "curve_n_obs"  # how many observations went into this patient's curve
# The centred parameterization (`centre_time=True`): the same line, described by its level at
# the patient's own mean visit time instead of at diagnosis. `curve_a` is still emitted, as
# `level − b·t̄`, so it keeps meaning "the value at t = 0" — but for prediction and for the
# level prior, `curve_level_c` is the coefficient that is actually estimated.
CURVE_LEVEL_C = "curve_level_c"  # level at the patient's mean visit time (t̄)
CURVE_TBAR = "curve_tbar"  # that mean visit time, on whatever axis the curve was fitted
# Anchored extrapolation: the same slope, but grown from the patient's most recent
# observation instead of from the intercept at diagnosis.
CURVE_ANCHOR_T = "curve_anchor_time"  # time since diagnosis of that observation
CURVE_ANCHOR_V = "curve_anchor_value"  # its de-biased value
CURVE_PRED_ANCHORED = "curve_pred_anchored"

# Floors, so a degenerate prior can never turn into division by ~zero.
_MIN_SD_B = 0.3
_MIN_SD_A = 3.0
_MIN_RESID = 1.0
# The curvature prior width, in points per year². Far below the slope floor because
# curvature is a small correction: the measured population value is ~0.03.
_MIN_SD_C = 0.02
# A quadratic needs three parameters, so it needs at least one degree of freedom left over.
# Below this the linear fit is used, which is also what keeps two- and three-visit patients
# on exactly the behaviour they had before.
_MIN_QUADRATIC_OBS = 4


def _prior_sd(hat: np.ndarray, var_hat: np.ndarray) -> float:
    """Between-patient spread of a coefficient, with the average sampling noise removed."""
    return float(np.sqrt(max(float(np.var(hat, ddof=1)) - float(np.mean(var_hat)), 0.0)))


def _ols_ab_cov(x: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """``(a, b, var_a, var_b, cov_ab, resid_var)`` — ``_ols`` plus the a/b sampling covariance.

    ``cov_ab = -resid_var · x̄ / Sxx``. It is not a small correction: the sampling error of
    an intercept and a slope are nearly perfectly anti-correlated whenever the time origin
    sits well outside the data, which it does here (``x`` is time since diagnosis, and the
    visits cluster years away from 0). For a typical patient here the implied sampling
    correlation is around −0.99.

    That matters because the *observed* spread of per-patient ``(a, b)`` estimates carries
    this term, so the raw correlation between them is mostly an artifact of the estimator
    rather than a fact about patients. Subtracting it is what makes the coupling
    :attr:`PatientCurveFeatures.rho_` measurable at all.
    """
    n = len(x)
    xbar = x.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if n < 2 or sxx <= 1e-9:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    b = float(((x - xbar) * (z - z.mean())).sum() / sxx)
    a = float(z.mean() - b * xbar)
    resid = z - (a + b * x)
    resid_var = float((resid**2).sum() / max(n - 2, 1))
    return (
        a,
        b,
        resid_var * (1.0 / n + xbar**2 / sxx),
        resid_var / sxx,
        -resid_var * xbar / sxx,
        resid_var,
    )


def _shrink(hat: float, var: float, prior_mean: float, inv_prior_var: float, k: float) -> float:
    """Empirical-Bayes posterior mean of one coefficient."""
    var_eff = max(var * k, 1e-9)
    return (hat / var_eff + prior_mean * inv_prior_var) / (1.0 / var_eff + inv_prior_var)


def _ols(x: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return ``(a, b, var_a, var_b, resid_var)`` for ``z = a + b·x``.

    ``var_a``/``var_b`` are the OLS sampling variances under homoscedastic errors.
    """
    n = len(x)
    xbar = x.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if n < 2 or sxx <= 1e-9:
        return float(z.mean()), np.nan, np.nan, np.nan, np.nan
    b = float(((x - xbar) * (z - z.mean())).sum() / sxx)
    a = float(z.mean() - b * xbar)
    resid = z - (a + b * x)
    dof = max(n - 2, 1)
    resid_var = float((resid**2).sum() / dof)
    var_b = resid_var / sxx
    var_a = resid_var * (1.0 / n + xbar**2 / sxx)
    return a, b, var_a, var_b, resid_var


def _ols_centred(x: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float, float]:
    """``(level, b, var_level, var_b, resid_var)`` for ``z = level + b·(x − x̄)``.

    The same line as :func:`_ols` describes, with a different pair of coefficients — and
    the sampling variance of the *level* is what makes the difference. It is ``resid_var/n``
    here and ``resid_var·(1/n + x̄²/Sxx)`` there; with ``x̄`` years away from the origin (this
    is time since diagnosis, and visits cluster around year 6), the second term dominates the
    first several times over. The empirical-Bayes weight below reads that variance, so fitting
    in the raw basis shrinks the level far harder than the evidence supports.

    The slope and its variance are basis-invariant and come out identically either way.
    """
    n = len(x)
    xbar = float(x.mean())
    u = x - xbar
    sxx = float((u**2).sum())
    if n < 2 or sxx <= 1e-9:
        return float(z.mean()), np.nan, np.nan, np.nan, np.nan
    level = float(z.mean())
    b = float((u * (z - level)).sum() / sxx)
    resid = z - (level + b * u)
    resid_var = float((resid**2).sum() / max(n - 2, 1))
    return level, b, resid_var / n, resid_var / sxx, resid_var


@dataclass(frozen=True)
class QuadraticFit:
    """One patient's fitted quadratic, in both the centred and the raw basis.

    ``alpha``/``beta`` are *shrunk* — they are the level at the patient's own mean visit
    time and the slope there, which is the pair that is close to independent. The raw
    ``(a, b, c)`` triple is what gets reported, so that ``curve_a`` keeps meaning "score at
    time since diagnosis 0" and ``curve_b`` keeps meaning "points per year there".
    ``gamma`` is the same number in either basis: curvature does not care where the origin is.
    """

    alpha: float
    beta: float
    gamma: float
    var_alpha: float
    var_beta: float
    var_gamma: float
    resid_var: float
    xbar: float

    @property
    def raw(self) -> tuple[float, float, float]:
        return (
            self.alpha - self.beta * self.xbar + self.gamma * self.xbar**2,
            self.beta - 2.0 * self.gamma * self.xbar,
            self.gamma,
        )

    def at(self, t: np.ndarray) -> np.ndarray:
        """The curve evaluated at ``t`` (in raw time-since-diagnosis units)."""
        u = t - self.xbar
        return self.alpha + self.beta * u + self.gamma * u**2

    def grow(self, t: np.ndarray, t0: float) -> np.ndarray:
        """Grow the curve from ``(t0, v0)``-style differences — the anchored form.

        Expressed as a difference so the same expression serves the anchor: a quadratic
        evaluated at ``t`` minus the same quadratic at ``t0`` is the increment to add to the
        value observed at ``t0``.
        """
        u, u0 = t - self.xbar, t0 - self.xbar
        return self.beta * (u - u0) + self.gamma * (u**2 - u0**2)


def _ols2(x: np.ndarray, z: np.ndarray) -> QuadraticFit | None:
    """Fit ``z = a + b·x + c·x²``, or return ``None`` if it is not determined.

    Fitted in the **centred** basis (``u = x − x̄``) and converted back at the end, because
    the raw normal equations are badly conditioned here: ``x`` runs to decades, so
    ``[1, x, x²]`` is nearly collinear and the raw sampling variances come out inflated.
    That matters beyond numerics — the empirical-Bayes weights below are built from those
    variances, and inflating ``var_a`` would shrink the level *harder* than the linear model
    does, which would cost more on the between-patient term than the curvature could win.
    Centring decouples the level from the curvature, and the variances are carried through
    the change of basis exactly, as ``T·cov·Tᵀ``.
    """
    n = len(x)
    if n < 3:
        return None
    xbar = float(x.mean())
    u = x - xbar
    s2 = float((u**2).sum())
    s3 = float((u**3).sum())
    s4 = float((u**4).sum())
    m = np.array([[n, 0.0, s2], [0.0, s2, s3], [s2, s3, s4]])  # normal matrix, Σu = 0
    if abs(float(np.linalg.det(m))) < 1e-9:
        return None
    rhs = np.array([z.sum(), float((u * z).sum()), float((u**2 * z).sum())])
    alpha, beta, gamma = (float(v) for v in np.linalg.solve(m, rhs))
    if not np.isfinite([alpha, beta, gamma]).all():
        return None
    resid = z - (alpha + beta * u + gamma * u**2)
    dof = max(n - 3, 1)
    resid_var = float((resid**2).sum() / dof)
    cov = resid_var * np.linalg.inv(m)
    t = np.array([[1.0, -xbar, xbar**2], [0.0, 1.0, -2.0 * xbar], [0.0, 0.0, 1.0]])
    var_alpha, var_beta, var_gamma = (float(v) for v in np.diag(t @ cov @ t.T))
    return QuadraticFit(alpha, beta, gamma, var_alpha, var_beta, var_gamma, resid_var, xbar)


class PatientCurveFeatures(Transformer):
    """Fit one shrunk linear curve per patient and emit its parameters and prediction.

    ``value_col`` is what the curve is fitted to. At train time that is the de-biased
    ``pseudo_target``; the population prior is learned from the *true* target instead
    (it is cleaner), falling back to ``value_col`` when no target is passed.
    """

    name = "patient_curve"

    # Measured (tests/test_properties.py): `fit` reads y for the population priors;
    # `transform` fits every patient's curve over the whole frame it is handed, so a row's
    # value depends on the patient's later visits (the forecast rebuilds them from the past).
    reads_target = True
    causal = False

    def __init__(
        self,
        value_col: str = "pseudo_target",
        time_col: str = TIME_SINCE_DIAGNOSIS,
        shrink_k: float = 0.5,
        shrink_intercept: bool = True,
        quadratic: bool = False,
        level_slope: bool = False,
        centre_time: bool = False,
    ) -> None:
        self.value_col = value_col
        self.time_col = time_col
        # Fit each patient's line as `level + b·(t − t̄_p)` instead of `a + b·t`.
        #
        # Both describe the same line, but the *sampling variance* of the coefficient the
        # level prior is applied to is not the same in the two bases, and the shrinkage weight
        # reads it: at t̄_p it is `resid_var/n`, at t = 0 it is `resid_var·(1/n + t̄²/Sxx)`,
        # and t̄ sits years from diagnosis, so the raw basis shrinks the level much harder than
        # the patient's own visits justify. The prior mean moves with it — `mu_level_` is the
        # population level *at the patient's own mean time*, which is the quantity the centred
        # coefficient actually estimates.
        #
        # The second thing it unlocks is `time_col: age`. A patient with no `age_at_diagnosis`
        # has no time since diagnosis, so the raw basis can only give them a population
        # constant; a level at their mean visit age is defined for everyone. Off by default,
        # and while it is off nothing below runs.
        self.centre_time = centre_time
        if centre_time and (quadratic or level_slope):
            # Both are parameterizations of the same (level, slope) pair, and both report
            # through `curve_b`/`rho_` — composing them would silently change what those
            # columns mean rather than add a feature.
            raise ValueError(
                "centre_time=True cannot be combined with quadratic= or level_slope="
                f" (got quadratic={quadratic}, level_slope={level_slope})"
            )
        # Couple the level and slope priors through their measured correlation instead of
        # shrinking them independently. Off by default; at ``rho_ == 0`` the joint
        # posterior below is algebraically the same expression as the independent one, so
        # the control is exact rather than approximate.
        self.level_slope = level_slope
        # On the *true* target a per-patient quadratic leaves 0.332 of residual where a
        # straight line leaves 1.602, so the shape of a trajectory is the one place a line is
        # clearly the wrong functional form. It only applies to patients with at least
        # _MIN_QUADRATIC_OBS usable visits; everyone else keeps the line, curvature 0.
        #
        # Off by default while it is being measured against the straight line on the real
        # CV — turning it on changes the fitted curve for most patients, so it has to earn
        # its place rather than arrive with it. See docs/audit-2026-09-16.md §5.1.
        self.quadratic = quadratic
        # Multiplier on the assumed sampling variance. k=1 is textbook empirical Bayes;
        # k>1 shrinks harder (assume our per-patient estimates are noisier than OLS says).
        # Measured on this data: k>1 clearly hurts (MSE 54.46 at k=1, 62.25 at k=3,
        # 87.61 at k=10), so the default is the mild, textbook setting.
        self.shrink_k = shrink_k
        # Shrinking the intercept is more delicate than shrinking the slope. The
        # between-patient spread of intercepts is large (~15 points) and a patient's own
        # visits really do pin their level down, so a prior there mostly adds bias. Set
        # False to leave the level entirely to the patient's data.
        self.shrink_intercept = shrink_intercept

    # -- fit ------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.require(df, [PATIENT, self.time_col, self.value_col])
        value = pd.Series(np.asarray(y, dtype=float), index=df.index) if y is not None else df[self.value_col]

        # ``groupby().indices`` yields *positional* indices, which do not necessarily
        # match the frame's labels (e.g. after a CV slice keeps the original index).
        # Work on plain arrays so the two can never be confused.
        x_all = df[self.time_col].to_numpy(dtype=float)
        idx_by_patient = df.groupby(PATIENT, sort=False).indices

        def _curves(z_all: np.ndarray) -> list[tuple[float, float, float, float, float, float]]:
            # ``(a, b, var_a, var_b, resid_var, n)``. ``var_a`` is kept rather than dropped:
            # the prior width below subtracts the *intercept's* sampling variance from the
            # observed spread of intercepts, and the residual variance is a different
            # quantity (it describes one observation, not the fitted level).
            out = []
            for idx in idx_by_patient.values():
                x, z = x_all[idx], z_all[idx]
                ok = ~np.isnan(x) & ~np.isnan(z)
                if ok.sum() < 2:
                    continue
                a, b, va, vb, rv = _ols(x[ok], z[ok])
                if np.isfinite(b):
                    out.append((a, b, va, vb, rv, float(ok.sum())))
            return out

        # The prior describes the *latent* curve, so estimate it from the cleanest signal
        # available — the target when we have it.
        ref = _curves(np.asarray(value, dtype=float))
        if len(ref) < 10:
            raise ValueError(f"only {len(ref)} usable patient curves — cannot set a prior")

        a_hat = np.array([r[0] for r in ref])
        b_hat = np.array([r[1] for r in ref])
        var_a = np.array([r[2] for r in ref])
        var_b = np.array([r[3] for r in ref])
        resid_vars = np.array([r[4] for r in ref])

        self.mu_a_ = float(np.mean(a_hat))
        self.mu_b_ = float(np.mean(b_hat))
        # Strip the average sampling noise out of the observed spread, so the prior width
        # reflects real between-patient variation rather than estimation error.
        #
        # Each coefficient subtracts *its own* sampling variance. The intercept used to
        # subtract `resid_vars` instead — the per-observation residual variance, which
        # ``_ols`` also returns and which describes a different quantity. On this data the
        # two happen to be close (mean 3.50 against the correct 4.33) and the observed
        # spread of 224.48 drowns both, so the prior width moves by 0.19%: 14.8374 against
        # the 14.8655 it used to be. Small, but it is the right statistic, and subtracting
        # too little is the direction that under-shrinks (docs/audit-2026-09-16.md §2 P2-2).
        var_a_true = max(float(np.var(a_hat, ddof=1)) - float(np.mean(var_a)), 0.0)
        var_b_true = max(float(np.var(b_hat, ddof=1)) - float(np.mean(var_b)), 0.0)
        self.sd_a_ = max(float(np.sqrt(var_a_true)), _MIN_SD_A)
        self.sd_b_ = max(float(np.sqrt(var_b_true)), _MIN_SD_B)

        # Level/slope coupling. The *raw* correlation between the per-patient ``a`` and
        # ``b`` estimates is dominated by the estimator's own sampling correlation (≈ −0.99
        # here, see _ols_ab_cov), so it is almost entirely an artifact. Subtracting the
        # average sampling covariance — exactly as var_a_true/var_b_true subtract the
        # average sampling variance — is what makes the remainder a fact about patients.
        self.rho_ = 0.0
        self.rho_raw_ = 0.0
        self.cov_samp_mean_ = 0.0
        if self.level_slope:
            cov_samp = []
            for idx in idx_by_patient.values():
                x, z = x_all[idx], np.asarray(value, dtype=float)[idx]
                ok = ~np.isnan(x) & ~np.isnan(z)
                if ok.sum() < 2:
                    continue
                _a, bb, _va, _vb, cov_ab, _rv = _ols_ab_cov(x[ok], z[ok])
                if np.isfinite(bb):
                    cov_samp.append(cov_ab)
            denom = float(np.sqrt(var_a_true) * np.sqrt(var_b_true))
            if len(cov_samp) == len(a_hat) and denom > 0:
                self.cov_samp_mean_ = float(np.mean(cov_samp))
                cov_obs = float(np.cov(a_hat, b_hat, ddof=1)[0, 1])
                self.rho_raw_ = cov_obs / denom
                self.rho_ = float(np.clip((cov_obs - self.cov_samp_mean_) / denom, -0.95, 0.95))

        # The shrinkage weight needs the noise of the series ``transform`` actually fits,
        # which is ``value_col`` — a de-biased observation, and considerably noisier than
        # the target. Borrowing the target's residual here understates the noise, and an
        # understated noise means under-shrinking exactly the patients with few visits.
        if y is not None:
            used = _curves(df[self.value_col].to_numpy(dtype=float))
            resid_used = [r[4] for r in used if np.isfinite(r[4])]
            self.sigma_resid_ = float(
                max(np.sqrt(np.mean(resid_used)), _MIN_RESID) if resid_used else _MIN_RESID
            )
        else:
            self.sigma_resid_ = float(max(np.sqrt(np.mean(resid_vars)), _MIN_RESID))

        # Priors for the quadratic path, built the same way but on the centred coefficients.
        # They are separate because they describe different quantities: the spread of
        # "level at t=0" is inflated by slope variation, while "level at the patient's own
        # mean time" is not. Reusing the linear priors here would shrink the level against
        # the wrong yardstick.
        self.mu_level_, self.sd_level_ = self.mu_a_, self.sd_a_
        self.mu_slope_, self.sd_slope_ = self.mu_b_, self.sd_b_
        self.mu_c_, self.sd_c_ = 0.0, _MIN_SD_C
        self.n_quadratic_ = 0
        if self.centre_time:
            # The level prior, estimated exactly the way the raw one above is — from the
            # cleanest series available (the target when `fit` was given one, `value_col`
            # otherwise) with the average *sampling* variance subtracted out. Skipping that
            # subtraction is not an option here: `var_level ≈ resid_var/n` shrinks with 1/n,
            # so the observed spread of the centred levels carries an estimation-noise term
            # that grows as trajectories get shorter, and a prior width read off the raw
            # spread would be far too wide on exactly the patients the prior has to hold.
            levels, level_vars = [], []
            z_ref = np.asarray(value, dtype=float)
            for idx in idx_by_patient.values():
                x, z = x_all[idx], z_ref[idx]
                ok = ~np.isnan(x) & ~np.isnan(z)
                if ok.sum() < 2:
                    continue
                level, b, var_level, _, _ = _ols_centred(x[ok], z[ok])
                if np.isfinite(b):
                    levels.append(level)
                    level_vars.append(var_level)
            if len(levels) < 10:
                raise ValueError(
                    f"only {len(levels)} usable patient levels — cannot set a level prior"
                )
            self.mu_level_ = float(np.mean(levels))
            self.sd_level_ = max(
                _prior_sd(np.array(levels), np.array(level_vars)), _MIN_SD_A
            )
        if self.quadratic:
            quads = []
            for idx in idx_by_patient.values():
                x, z = x_all[idx], np.asarray(value, dtype=float)[idx]
                ok = ~np.isnan(x) & ~np.isnan(z)
                if ok.sum() < _MIN_QUADRATIC_OBS:
                    continue
                q = _ols2(x[ok], z[ok])
                if q is not None:
                    quads.append(q)
            if len(quads) >= 10:
                lvl = np.array([q.alpha for q in quads])
                slp = np.array([q.beta for q in quads])
                gam = np.array([q.gamma for q in quads])
                self.mu_level_ = float(np.mean(lvl))
                self.mu_slope_ = float(np.mean(slp))
                self.mu_c_ = float(np.mean(gam))
                self.sd_level_ = max(_prior_sd(lvl, np.array([q.var_alpha for q in quads])), _MIN_SD_A)
                self.sd_slope_ = max(_prior_sd(slp, np.array([q.var_beta for q in quads])), _MIN_SD_B)
                self.sd_c_ = max(_prior_sd(gam, np.array([q.var_gamma for q in quads])), _MIN_SD_C)
                self.n_quadratic_ = len(quads)

        self.n_curves_ = len(ref)
        return self

    def _shrink_ab_joint(
        self, a_hat: float, var_a: float, b_hat: float, var_b: float
    ) -> tuple[float, float]:
        """Joint empirical-Bayes posterior mean of ``(a, b)`` under the correlated prior.

        The prior is the bivariate normal with correlation ``rho_``; the likelihood keeps
        the independent (diagonal) form the single-coefficient path uses. At ``rho_ == 0``
        the prior precision is diagonal and this reduces, algebraically and not
        approximately, to the two independent ``_shrink`` calls — which is what makes
        ``level_slope`` an exact A/B.

        Only the *prior* carries the coupling because that is the only place the coupling
        is known: the sampling correlation between ``â`` and ``b̂`` is an artifact of the
        line fit, already removed in ``rho_``, and putting it back into the likelihood
        would reintroduce it.
        """
        rho = self.rho_
        scale = 1.0 / (1.0 - rho**2)
        off = -rho * scale / (self.sd_a_ * self.sd_b_)
        prior_prec = np.array(
            [[scale / self.sd_a_**2, off], [off, scale / self.sd_b_**2]]
        )
        var_a_eff = max(var_a * self.shrink_k, 1e-9)
        var_b_eff = max(var_b * self.shrink_k, 1e-9)
        lik_prec = np.diag([1.0 / var_a_eff, 1.0 / var_b_eff])
        post_prec = prior_prec + lik_prec
        rhs = prior_prec @ np.array([self.mu_a_, self.mu_b_]) + lik_prec @ np.array(
            [a_hat, b_hat]
        )
        a_p, b_p = np.linalg.solve(post_prec, rhs)
        return float(a_p), float(b_p)

    def _centred_coefficients(
        self, x: np.ndarray, z: np.ndarray, n_ok: int
    ) -> tuple[float, float, float]:
        """``(level, slope, t̄)`` for one patient — the `centre_time` path's whole fit.

        ``t̄`` is NaN when the patient has no observation with a time on the fitted axis, and
        the level is then the population one with a flat curve: there is nothing to anchor a
        level *at their mean visit time* to. That is the same degenerate case the raw path
        answers with ``mu_a_`` at t = 0.
        """
        if n_ok == 0:
            return self.mu_level_, self.mu_b_, np.nan
        tbar = float(x.mean())
        level_hat, b_hat, var_level, var_b, _ = _ols_centred(x, z)
        if not np.isfinite(b_hat):
            # One observation, or several at one time: the level is the patient's mean, the
            # slope is the prior, and the sampling variance of that level is the residual
            # variance of the series itself — there is no second time to measure it from, so
            # `sigma_resid_` stands in. Exactly the substitution the raw path makes.
            b_p = self.mu_b_
            if not self.shrink_intercept:
                return float(level_hat), b_p, tbar
            v_obs = max(self.sigma_resid_**2 * self.shrink_k, 1e-6)
            inv_sd_lvl2 = 1.0 / self.sd_level_**2
            w = (1.0 / v_obs) / (1.0 / v_obs + inv_sd_lvl2)
            return float(w * level_hat + (1.0 - w) * self.mu_level_), b_p, tbar
        b_p = _shrink(b_hat, var_b, self.mu_b_, 1.0 / self.sd_b_**2, self.shrink_k)
        if not self.shrink_intercept:
            return level_hat, b_p, tbar
        level_p = _shrink(
            level_hat, var_level, self.mu_level_, 1.0 / self.sd_level_**2, self.shrink_k
        )
        return level_p, b_p, tbar

    def _shrink_level(self, quad: QuadraticFit, inv_sd_lvl2: float) -> float:
        """The centred level coefficient, honouring ``shrink_intercept``."""
        if not self.shrink_intercept:
            return quad.alpha
        return _shrink(quad.alpha, quad.var_alpha, self.mu_level_, inv_sd_lvl2, self.shrink_k)

    # -- transform ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.require(df, [PATIENT, self.time_col, self.value_col])
        out = df.copy()

        a_col = np.full(len(out), np.nan)
        b_col = np.full(len(out), np.nan)
        c_col = np.full(len(out), np.nan)
        pred_col = np.full(len(out), np.nan)
        n_col = np.zeros(len(out), dtype=int)
        anchor_t = np.full(len(out), np.nan)
        anchor_v = np.full(len(out), np.nan)
        anchored_col = np.full(len(out), np.nan)
        level_col = np.full(len(out), np.nan)
        tbar_col = np.full(len(out), np.nan)

        # Shrinkage weights, precomputed once.
        inv_sd_a2 = 1.0 / self.sd_a_**2
        inv_sd_b2 = 1.0 / self.sd_b_**2
        inv_sd_lvl2 = 1.0 / self.sd_level_**2
        inv_sd_slp2 = 1.0 / self.sd_slope_**2
        inv_sd_c2 = 1.0 / self.sd_c_**2

        x_all = out[self.time_col].to_numpy(dtype=float)
        z_all = out[self.value_col].to_numpy(dtype=float)

        for _, idx in out.groupby(PATIENT, sort=False).indices.items():
            x, z = x_all[idx], z_all[idx]
            ok = ~np.isnan(x) & ~np.isnan(z)
            n_ok = int(ok.sum())
            n_col[idx] = n_ok

            # Curvature is only attempted where there is a degree of freedom to estimate it
            # with; below that the patient keeps exactly the straight-line treatment they
            # had before, curvature 0 and all.
            quad = (
                _ols2(x[ok], z[ok])
                if self.quadratic and n_ok >= _MIN_QUADRATIC_OBS
                else None
            )

            if self.centre_time:
                # The whole fit, in the centred basis: the level at the patient's own mean
                # visit time, the slope, and where that mean time is. `curve_a` is still
                # reported — it is the same line read at t = 0 — but it is a *derived*
                # quantity now, and a patient with no time on this axis has no value there at
                # all, so it stays NaN rather than borrowing the population one.
                level_p, b_p, tbar_p = self._centred_coefficients(x[ok], z[ok], n_ok)
                c_p = 0.0
                level_col[idx] = level_p
                tbar_col[idx] = tbar_p
                a_col[idx] = level_p - b_p * tbar_p if np.isfinite(tbar_p) else np.nan
                b_col[idx] = b_p
                c_col[idx] = c_p
                # A missing time evaluates at the patient's own centre, which is what makes
                # the level-only patient a flat curve through their own level.
                u = (np.zeros_like(x) if not np.isfinite(tbar_p)
                     else np.nan_to_num(x - tbar_p, nan=0.0))
                pred_col[idx] = level_p + b_p * u
            elif quad is not None:
                # Shrink the centred triple — level and slope at the patient's own mean
                # time, and curvature — then convert to the raw basis for reporting.
                fit = QuadraticFit(
                    alpha=self._shrink_level(quad, inv_sd_lvl2),
                    beta=_shrink(quad.beta, quad.var_beta, self.mu_slope_, inv_sd_slp2, self.shrink_k),
                    gamma=_shrink(quad.gamma, quad.var_gamma, self.mu_c_, inv_sd_c2, self.shrink_k),
                    var_alpha=quad.var_alpha,
                    var_beta=quad.var_beta,
                    var_gamma=quad.var_gamma,
                    resid_var=quad.resid_var,
                    xbar=quad.xbar,
                )
                a_p, b_p, c_p = fit.raw
            elif n_ok == 0:
                a_p, b_p, c_p = self.mu_a_, self.mu_b_, 0.0
            else:
                c_p = 0.0
                a_hat, b_hat, var_a, var_b, resid_var = _ols(x[ok], z[ok])
                if n_ok < 2 or not np.isfinite(b_hat):
                    # Only one observation: slope is entirely the prior, intercept is
                    # that observation pulled toward the prior mean.
                    b_p = self.mu_b_
                    if self.shrink_intercept:
                        v_obs = max(self.sigma_resid_**2 * self.shrink_k, 1e-6)
                        w = (1.0 / v_obs) / (1.0 / v_obs + inv_sd_a2)
                        a_p = w * a_hat + (1.0 - w) * self.mu_a_
                    else:
                        a_p = a_hat
                elif self.level_slope and self.shrink_intercept:
                    a_p, b_p = self._shrink_ab_joint(a_hat, var_a, b_hat, var_b)
                else:
                    var_b_eff = max(var_b * self.shrink_k, 1e-9)
                    b_p = (b_hat / var_b_eff + self.mu_b_ * inv_sd_b2) / (1.0 / var_b_eff + inv_sd_b2)
                    if self.shrink_intercept:
                        var_a_eff = max(var_a * self.shrink_k, 1e-9)
                        a_p = (a_hat / var_a_eff + self.mu_a_ * inv_sd_a2) / (1.0 / var_a_eff + inv_sd_a2)
                    else:
                        a_p = a_hat

            if not self.centre_time:
                a_col[idx] = a_p
                b_col[idx] = b_p
                c_col[idx] = c_p
                t = np.nan_to_num(x, nan=0.0)
                pred_col[idx] = a_p + b_p * t + c_p * t**2

            # Anchor at the patient's latest *observed* point and grow the same slope from
            # there. Evaluating the curve at t=0 and extrapolating multiplies any slope
            # error by the whole time since diagnosis; anchoring multiplies it only by the
            # distance to the visit being predicted, which is what makes a forecast
            # degrade gracefully instead of drifting.
            if n_ok:
                last = idx[np.flatnonzero(ok)[-1]]
                t0, v0 = x_all[last], z_all[last]
                anchor_t[idx] = t0
                anchor_v[idx] = v0
                t_far = np.nan_to_num(x, nan=t0)
                anchored_col[idx] = v0 + b_p * (t_far - t0) + c_p * (t_far**2 - t0**2)

        out[CURVE_A] = a_col
        out[CURVE_B] = b_col
        if self.quadratic:
            # Only when it is on. Emitting an all-zero column otherwise would not change
            # any fitted value, but it *would* shift the column subsampling inside the
            # booster — `colsample_bytree` draws from the column list — and the control
            # would then differ from the incumbent by a few thousandths for no reason.
            out[CURVE_C] = c_col
        if self.centre_time:
            # Emitted only in the mode that produces them, for the reason `curve_c` is: an
            # extra column changes `colsample_bytree`'s draw and would move the incumbent by
            # a few thousandths for no reason at all.
            out[CURVE_LEVEL_C] = level_col
            out[CURVE_TBAR] = tbar_col
        out[CURVE_PRED] = pred_col
        out[CURVE_N] = n_col
        out[CURVE_ANCHOR_T] = anchor_t
        out[CURVE_ANCHOR_V] = anchor_v
        out[CURVE_PRED_ANCHORED] = anchored_col
        return self.preserve_index(out, df)

    # -- introspection --------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        return {
            "mu_a": round(self.mu_a_, 3),
            "sd_a": round(self.sd_a_, 3),
            "mu_b": round(self.mu_b_, 3),
            "sd_b": round(self.sd_b_, 3),
            "sigma_resid": round(self.sigma_resid_, 3),
            "n_curves": self.n_curves_,
            "centre_time": self.centre_time,
            # The centred level prior, when there is one. `mu_level`/`sd_level` are the raw
            # pair under `centre_time=False` (nothing sets them otherwise), so a run's record
            # says which prior was actually applied.
            "mu_level": round(self.mu_level_, 3),
            "sd_level": round(self.sd_level_, 3),
            "shrink_k": self.shrink_k,
            "quadratic": self.quadratic,
            "n_quadratic": self.n_quadratic_,
            "mu_c": round(self.mu_c_, 4),
            "sd_c": round(self.sd_c_, 4),
            "level_slope": self.level_slope,
            "rho": round(self.rho_, 4),
            "rho_raw": round(self.rho_raw_, 4),
            "cov_samp_mean": round(self.cov_samp_mean_, 3),
        }
