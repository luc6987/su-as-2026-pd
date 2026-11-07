"""A per-patient curve that respects the structure of the observations.

:class:`~pdchallenge.features.longitudinal.PatientCurveFeatures` fits a straight line to
``pseudo_target`` by least squares. Measured against the true target that is the wrong
model in three ways at once (``reports/diagnosis-2026-09-18/``):

* the deficit is **multiplicative** — ``obs ≈ T · m_v``, so the residual grows with the
  score and an additive line is biased by level (its error has slope −0.26 on ``T``);
* ``m_v`` is **one-sided** — mean 0.82, SD 0.21, left tail down to 0.3, right edge ≈1.1 —
  so the curve should sit near the *upper* edge of the observations, not through their
  middle;
* the target is an exact smooth curve whose slope rises and then decays, so on the log
  scale a quadratic is a better local shape than a line.

So this transformer fits, per patient, a quadratic in **log(obs + k)** with an
**asymmetric (quantile) loss** at ``tau`` — residuals above the curve are charged ``tau``,
those below ``1 − tau`` — and with a **ridge prior** on the slope and curvature toward
population values learned from the training target. The prior is what makes the one-sided
fit usable: with 3–8 points and no prior, a quantile or envelope line is determined by two
of them and is *worse* than least squares (measured, ``a4``); with the prior it is the
single most useful feature the booster sees.

The intercept is never shrunk. A patient's own observations pin their level; the prior
would only pull it toward the population and cost the between-patient term — the same
lesson the additive chain learned about ``shrink_intercept``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT, TIME_SINCE_DIAGNOSIS
from ..data.splits import group_kfold
from .base import Transformer
from .ratio import OBS, OBS_WEIGHT

ENV_PRED = "env_pred"  # the fitted curve at this visit, back on the score scale
ENV_LEVEL_LOG = "env_level_log"  # log-scale level at the patient's mean visit time
ENV_SLOPE_LOG = "env_slope_log"  # log-scale slope there (per year)
ENV_CURV_LOG = "env_curv_log"  # log-scale curvature (per year²); 0 on the linear fallback
ENV_TBAR = "env_tbar"  # the patient's mean visit time — the curve is centred on it
ENV_N_OBS = "env_n_obs"  # how many observations went into this patient's curve
OBS_MINUS_ENV = "obs_minus_env"
OFF_MINUS_ENV = "off_minus_env"

ENV_PARAM_COLS = [ENV_LEVEL_LOG, ENV_SLOPE_LOG, ENV_CURV_LOG, ENV_TBAR, ENV_N_OBS]

# Floors on the prior widths (log scale), so a degenerate prior can never pin every
# patient to the population value. The measured widths are ~0.06 (slope) and ~0.026
# (curvature) on the real cohort.
_MIN_SD_B = 0.01
_MIN_SD_C = 0.005
# When the training frame has too few long trajectories to estimate a curvature prior —
# the forecast validation truncates its fitting patients to k = 2 or 3 visits — fall back
# to "no curvature, loosely held". Reported by `get_params` so a run cannot hide it.
_FALLBACK_CURV_PRIOR = (0.0, 0.03)
_MIN_PRIOR_PATIENTS = 10
# A slope read off a trajectory shorter than this is mostly the target's 0.1 rounding; the
# prior prefers longer ones and only reaches for the short ones when there is nothing else
# (again the truncated forecast frames).
_MIN_SPAN_YEARS = 1.0
_SHORT_SPAN_YEARS = 0.25
_MAX_ITER = 40
_RESID_FLOOR = 0.02  # IRLS weight cap: |r| below this is treated as this
_TOL = 1e-6

# The grid `prior_scale="auto"` searches, and why it has to be searched at all: the scale
# converts the quantile loss's IRLS weights (τ/|r|) into a Gaussian prior precision, which
# is a units question with no textbook answer — so it can only be measured. Measuring it on
# the same CV that judges the chain is a selection, and it was one: 0.25 came off a sweep on
# seed 0 (`docs/consolidation-2026-09-24.md` §3.1). `"auto"` re-measures it *inside* each
# training fold, on a nested patient-grouped split of that fold's own rows, so the outer
# CV's validation rows take no part in the choice. The grid brackets the measured optimum
# (0.25) on both sides and is deliberately coarse — it is six fits per inner fold.
_AUTO_GRID: tuple[float, ...] = (0.1, 0.18, 0.25, 0.35, 0.5, 1.0)
# `prior_scale="auto"` picks by regressing the target on the fold's `env_pred`, so it needs
# enough held-out rows for a two-parameter line to mean anything.
_MIN_AUTO_ROWS = 3


@dataclass(frozen=True)
class CurvePriors:
    """Population prior on a patient's log-scale slope and curvature, and where it came from.

    ``slope_source`` records which tier fired — ``"quadratic"`` (the preferred one, from
    patients with enough visits for a quadratic), ``"linear"``, or ``"short_span"`` — and
    ``curv_source`` is ``"fitted"`` or ``"fallback"``. Both are reported by ``get_params``
    so a run cannot quietly use the fallback.
    """

    mu_b: float
    sd_b: float
    mu_c: float
    sd_c: float
    slope_source: str
    curv_source: str
    n_slope: int
    n_curv: int


def fit_curve_priors(
    t_all: np.ndarray,
    z_all: np.ndarray,
    group_indices: Iterable[np.ndarray],
    quadratic_min_obs: int,
) -> CurvePriors:
    """Read the slope and curvature priors off the **true** trajectories.

    ``t_all``/``z_all`` are the visit times and ``log(target + k)`` for every row, and
    ``group_indices`` is the positional per-patient index arrays (``groupby(...).indices``).
    Shared by the envelope and likelihood steps: both need a population prior on the shape,
    and both must fit it on the training fold only.
    """
    quad_slopes, quad_curvs, lin_slopes, short_slopes = [], [], [], []
    for idx in group_indices:
        t, z = t_all[idx], z_all[idx]
        ok = ~np.isnan(t) & ~np.isnan(z)
        if ok.sum() < 2:
            continue
        span = float(np.ptp(t[ok]))
        if span < _SHORT_SPAN_YEARS:
            continue
        u = t[ok] - t[ok].mean()
        slope = float(np.polyfit(u, z[ok], 1)[0])
        if span < _MIN_SPAN_YEARS:
            short_slopes.append(slope)
            continue
        lin_slopes.append(slope)
        if ok.sum() >= quadratic_min_obs:
            c, b, _ = np.polyfit(u, z[ok], 2)
            quad_slopes.append(float(b))
            quad_curvs.append(float(c))

    if len(quad_slopes) >= _MIN_PRIOR_PATIENTS:
        slopes, slope_source = quad_slopes, "quadratic"
        mu_c, sd_c = float(np.mean(quad_curvs)), float(np.std(quad_curvs))
        curv_source = "fitted"
    else:
        mu_c, sd_c = _FALLBACK_CURV_PRIOR
        curv_source = "fallback"
        if len(lin_slopes) >= _MIN_PRIOR_PATIENTS:
            slopes, slope_source = lin_slopes, "linear"
        elif len(lin_slopes) + len(short_slopes) >= _MIN_PRIOR_PATIENTS:
            slopes, slope_source = lin_slopes + short_slopes, "short_span"
        else:
            raise ValueError(
                f"only {len(lin_slopes) + len(short_slopes)} usable patient trajectories "
                "— cannot set a slope prior"
            )
    return CurvePriors(
        mu_b=float(np.mean(slopes)),
        sd_b=max(float(np.std(slopes)), _MIN_SD_B),
        mu_c=mu_c,
        sd_c=max(sd_c, _MIN_SD_C),
        slope_source=slope_source,
        curv_source=curv_source,
        n_slope=len(slopes),
        n_curv=len(quad_curvs) if curv_source == "fitted" else 0,
    )


def fit_envelope(
    t: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    tau: float,
    lam_b: float,
    lam_c: float,
    mu_b: float,
    mu_c: float,
    quadratic: bool,
) -> np.ndarray:
    """Penalised quantile regression of ``z`` on ``t``, in the centred basis.

    Returns ``(level, slope[, curvature])`` at ``u = t − mean(t)``. The loss is the
    quantile check function at ``tau`` weighted by ``w``, plus ridge penalties pulling the
    slope toward ``mu_b`` with strength ``lam_b`` and (when ``quadratic``) the curvature
    toward ``mu_c`` with ``lam_c``. Solved by iteratively reweighted least squares — the
    check loss is |r| with an asymmetric multiplier, and |r| = r²/|r|, so each step is a
    weighted ridge solve with weight ``tau_or_(1−tau) / |r|``. The starting point is the
    weighted least-squares solution.
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    u = t - t.mean()
    n = len(t)
    cols = [np.ones(n), u] + ([u**2] if quadratic else [])
    A = np.stack(cols, axis=1)
    p = A.shape[1]
    P = np.zeros((p, p))
    m0 = np.zeros(p)
    P[1, 1] = lam_b
    m0[1] = mu_b
    if quadratic:
        P[2, 2] = lam_c
        m0[2] = mu_c

    def _solve(weights: np.ndarray) -> np.ndarray:
        lhs = A.T @ (A * weights[:, None]) + P
        rhs = A.T @ (weights * z) + P @ m0
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    beta = _solve(w)
    for _ in range(_MAX_ITER):
        r = z - A @ beta
        wq = w * np.where(r >= 0, tau, 1.0 - tau) / np.maximum(np.abs(r), _RESID_FLOOR)
        new = _solve(wq)
        done = np.max(np.abs(new - beta)) < _TOL
        beta = new
        if done:
            break
    return beta


def evaluate_envelope(
    level: np.ndarray | float,
    slope: np.ndarray | float,
    curv: np.ndarray | float,
    tbar: np.ndarray | float,
    t: np.ndarray,
    k_offset: float,
) -> np.ndarray:
    """The fitted curve at time ``t``, back on the score scale.

    The forecast validation calls this with parameters carried from a patient's known
    visits and the *future* visit's ``t`` — a curve must be re-evaluated there, never
    copied as a constant. A missing ``t`` evaluates at the patient's centre.
    """
    tbar = np.asarray(tbar, dtype=float)
    u = np.where(np.isnan(t), 0.0, np.asarray(t, dtype=float) - tbar)
    log_pred = np.asarray(level, dtype=float) + np.asarray(slope, dtype=float) * u + np.asarray(curv, dtype=float) * u**2
    return np.exp(log_pred) - k_offset


@dataclass(frozen=True)
class _CurveParams:
    """One fitted curve per row: the parameters, and the times they were fitted at."""

    level: np.ndarray
    slope: np.ndarray
    curv: np.ndarray
    tbar: np.ndarray
    n_obs: np.ndarray
    t: np.ndarray


def _ols_residual_mse(y_true: np.ndarray, x_pred: np.ndarray) -> float:
    """MSE of ``y_true`` about the least-squares line ``a + b·x_pred``.

    This is what ``prior_scale`` is chosen on: how much of the target the fitted curve
    explains, with the two free parameters the curve is *allowed* to be off by (its level
    and its overall scale, both of which the booster refits anyway) taken out. Ranking
    candidates on raw MSE against ``y`` instead would just prefer whichever scale happens to
    land the curve near the target's mean. ``x_pred`` is a single column, so ``1 − r²`` times
    the target's variance — but computed as a regression because that is what the line is.
    """
    ok = np.isfinite(y_true) & np.isfinite(x_pred)
    if int(ok.sum()) < _MIN_AUTO_ROWS:
        raise ValueError(
            f"only {int(ok.sum())} rows to score a prior scale on — need {_MIN_AUTO_ROWS}"
        )
    y_ok, x_ok = y_true[ok], x_pred[ok]
    design = np.stack([np.ones(len(x_ok)), x_ok], axis=1)
    coef = np.linalg.lstsq(design, y_ok, rcond=None)[0]
    resid = y_ok - design @ coef
    return float(np.mean(resid**2))


class EnvelopeCurveFeatures(Transformer):
    """One log-scale, one-sided, prior-shrunk quadratic per patient.

    ``fit`` needs the target: the population priors on the log-scale slope and curvature
    are read off the *true* trajectories. ``transform`` fits every patient's curve to
    their own ``obs`` and needs no target, so it applies unchanged to the test set.

    ``prior_scale`` is either a number or ``"auto"``. As a number it multiplies the prior
    precision directly and is a fixed choice made *outside* this class — which is how the
    shipped 0.25 was picked, on the same CV that then judged it. As ``"auto"`` the scale is
    chosen inside every ``fit``, by a nested patient-grouped CV over the training frame's own
    patients (:meth:`_select_prior_scale`), so the rows that judge the chain never take part
    in the choice. The cost is ``len(auto_grid)`` extra passes of the per-patient fit per
    inner fold; the benefit is that the number is no longer a selection.
    """

    name = "envelope_curve"

    # Measured (tests/test_properties.py): `fit` reads y for the slope/curvature priors;
    # `transform` fits every patient's curve over the whole frame it is handed, so a row's
    # value depends on the patient's later visits (the forecast re-evaluates the carried
    # curve at the future visit's time instead).
    reads_target = True
    causal = False

    def __init__(
        self,
        tau: float = 0.8,
        k_offset: float = 3.0,
        quadratic_min_obs: int = 4,
        prior_scale: float | str = 1.0,
        value_col: str = OBS,
        weight_col: str = OBS_WEIGHT,
        time_col: str = TIME_SINCE_DIAGNOSIS,
        flat_level_fallback: bool = True,
        auto_grid: tuple[float, ...] = _AUTO_GRID,
        auto_folds: int = 3,
        auto_seed: int = 0,
    ) -> None:
        # tau = 0.5 is a symmetric (median) fit; 0.8 charges a point above the curve four
        # times what it charges one below, which is where the one-sidedness goes in.
        self.tau = tau
        # The log offset: log(obs + k). k > 0 keeps a zero score finite and softens the
        # low end, where the deficit is additive noise rather than a fraction of T.
        self.k_offset = k_offset
        self.quadratic_min_obs = quadratic_min_obs
        # Multiplier on the prior precision: >1 shrinks harder toward the population. `"auto"`
        # measures it per training fold instead (see the class docstring); `prior_scale_` is
        # what `transform` actually uses, and for a float it is that float unchanged.
        if prior_scale != "auto":
            # `prior_scale` is what `fit` branches on, so a typo like "Auto" must not slip
            # through as neither a number nor the sentinel.
            try:
                valid = float(prior_scale) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(
                    f"prior_scale must be a positive number or 'auto', got {prior_scale!r}"
                )
        self.prior_scale = prior_scale
        self.auto_grid = auto_grid
        self.auto_folds = auto_folds
        self.auto_seed = auto_seed
        self.value_col = value_col
        self.weight_col = weight_col
        self.time_col = time_col
        # A patient with observations but no time axis still has a level, so the default
        # fits one flat rather than leaving `env_pred` NaN — see `transform`. `False` is the
        # A/B control (`configs/envelope_no_flat.yaml`): it restores the NaN, which is what
        # the model would have to fall back on if those rows had to be scored without a
        # curve. Measured neutral — a mean difference of +0.0224 MSE over seeds 0/1/2 whose
        # sign flips between them (CLAUDE.md §6) — so this is kept for the NaN-freedom.
        self.flat_level_fallback = flat_level_fallback

    # -- fit ------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.require(df, [PATIENT, self.time_col])
        if y is None:
            raise ValueError("EnvelopeCurveFeatures needs the training target to set its priors")
        target = np.asarray(y, dtype=float)
        zT = np.log(np.clip(target, 0.0, None) + self.k_offset)
        t_all = df[self.time_col].to_numpy(dtype=float)

        # The scale is settled before the priors, because `"auto"` measures it against priors
        # fitted on sub-folds rather than on the whole training frame. The final prior fit
        # below is unchanged either way — the selection only picks the number.
        if self.prior_scale == "auto":
            self.prior_scale_ = self._select_prior_scale(df, t_all, zT, target)
        else:
            self.prior_scale_ = float(self.prior_scale)
            self.auto_mse_ = None

        # Priors from the true trajectories — shared with the likelihood step
        # (`features/likelihood.py`), which needs the same population prior on the shape.
        # When there are too few long trajectories (a truncated training frame), the tier
        # that fired is reported by `get_params` so a run cannot quietly use the fallback.
        priors = fit_curve_priors(
            t_all, zT, df.groupby(PATIENT, sort=False).indices.values(), self.quadratic_min_obs
        )
        self.mu_b_, self.sd_b_ = priors.mu_b, priors.sd_b
        self.mu_c_, self.sd_c_ = priors.mu_c, priors.sd_c
        self.slope_prior_, self.curv_prior_ = priors.slope_source, priors.curv_source
        self.n_slope_prior_, self.n_curv_prior_ = priors.n_slope, priors.n_curv
        return self

    # -- auto prior scale -----------------------------------------------------------
    def _select_prior_scale(self, df: pd.DataFrame, t_all: np.ndarray, z_true_log: np.ndarray,
                            target: np.ndarray) -> float:
        """Choose ``prior_scale`` by a nested, patient-grouped CV inside the training frame.

        Each inner fold re-estimates the priors on its own training patients (so the
        candidate is judged against the same kind of prior the outer fold will use), then
        fits every inner-validation patient's curve at each candidate scale and scores the
        resulting ``env_pred`` by :func:`_ols_residual_mse` against the target. Candidates
        are ranked by their mean over folds; ties go to the smaller scale, which is the
        more conservative choice — a smaller scale is a weaker prior.

        This is *not* free: it is ``len(auto_grid)`` extra passes of the per-patient fit per
        inner fold, all of it inside `fit`. What it buys is that the number is measured on
        the training rows of the fold being fitted rather than on the rows that score the
        chain, so a CV number stops carrying the selection.
        """
        grid = tuple(float(c) for c in self.auto_grid)
        if not grid:
            raise ValueError("auto_grid is empty — nothing to choose from")
        groups = df[PATIENT].to_numpy()
        fold_mse: dict[float, list[float]] = {c: [] for c in grid}
        for tr, va in group_kfold(groups, self.auto_folds, seed=self.auto_seed):
            inner = df.iloc[tr]
            priors = fit_curve_priors(
                t_all[tr], z_true_log[tr],
                inner.groupby(PATIENT, sort=False).indices.values(),
                self.quadratic_min_obs,
            )
            inner_val = df.iloc[va]
            for cand in grid:
                params = self._curve_params(
                    inner_val, priors.mu_b, priors.sd_b, priors.mu_c, priors.sd_c, cand
                )
                env_pred = evaluate_envelope(
                    params.level, params.slope, params.curv, params.tbar, params.t, self.k_offset
                )
                fold_mse[cand].append(_ols_residual_mse(target[va], env_pred))
        self.auto_mse_ = {str(c): float(np.mean(v)) for c, v in fold_mse.items()}
        return min(grid, key=lambda c: (float(np.mean(fold_mse[c])), c))

    # -- transform ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.require(df, [PATIENT, self.time_col, self.value_col, self.weight_col])
        out = df.copy()
        p = self._curve_params(df, self.mu_b_, self.sd_b_, self.mu_c_, self.sd_c_, self.prior_scale_)
        out[ENV_LEVEL_LOG] = p.level
        out[ENV_SLOPE_LOG] = p.slope
        out[ENV_CURV_LOG] = p.curv
        out[ENV_TBAR] = p.tbar
        out[ENV_N_OBS] = p.n_obs
        out[ENV_PRED] = evaluate_envelope(p.level, p.slope, p.curv, p.tbar, p.t, self.k_offset)
        out[OBS_MINUS_ENV] = out[self.value_col] - out[ENV_PRED]
        out[OFF_MINUS_ENV] = out["off"] - out[ENV_PRED] if "off" in out else np.nan
        return self.preserve_index(out, df)

    def _curve_params(self, df: pd.DataFrame, mu_b: float, sd_b: float, mu_c: float,
                      sd_c: float, prior_scale: float) -> _CurveParams:
        """Every patient's fitted curve, under an explicit prior and scale.

        The priors and the scale are arguments rather than ``self`` attributes because
        ``prior_scale="auto"`` has to fit the same frame under several scales and several
        priors, none of which is the one `transform` will finally use. Routing both callers
        through one function is what keeps the auto path and the fixed path the *same*
        arithmetic — a separate scoring implementation could disagree with `transform`
        about the thing it is choosing.
        """
        n = len(df)
        level = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        curv = np.full(n, np.nan)
        tbar = np.full(n, np.nan)
        n_obs = np.zeros(n, dtype=int)

        t_all = df[self.time_col].to_numpy(dtype=float)
        z_all = np.log(np.clip(df[self.value_col].to_numpy(dtype=float), 0.0, None) + self.k_offset)
        w_all = df[self.weight_col].to_numpy(dtype=float)
        lam_b = prior_scale / sd_b**2
        lam_c = prior_scale / sd_c**2

        # `groupby().indices` is positional — index arrays, never `.loc`.
        for idx in df.groupby(PATIENT, sort=False).indices.values():
            t, z, w = t_all[idx], z_all[idx], w_all[idx]
            has_obs = ~np.isnan(z) & ~np.isnan(w)
            ok = has_obs & ~np.isnan(t)
            if not ok.any() and has_obs.any():
                # No time axis at all (age at diagnosis unknown — 5% of patients) but
                # observations exist: fit the level alone, at the same quantile, as if every
                # visit sat at one time. The curve is then flat and centred at 0, which is
                # what `evaluate_envelope` returns for a missing `t`.
                #
                # Without this the patient has no curve and `env_pred` — the heaviest
                # feature — is NaN on those rows. `flat_level_fallback=False` is the A/B
                # control and restores the NaN; it is measured neutral (CLAUDE.md §6), so
                # the fallback is kept for the NaN-freedom rather than for the score.
                if not self.flat_level_fallback:
                    continue
                ok = has_obs
                t = np.zeros_like(t)
            n_ok = int(ok.sum())
            n_obs[idx] = n_ok
            if n_ok == 0:
                continue
            centre = float(t[ok].mean())
            tbar[idx] = centre
            if n_ok == 1:
                level[idx] = float(z[ok][0])
                slope[idx] = mu_b
                curv[idx] = 0.0
                continue
            flat = np.ptp(t[ok]) < 1e-6
            beta = fit_envelope(
                t[ok], z[ok], w[ok], self.tau, lam_b, lam_c, mu_b, mu_c,
                quadratic=n_ok >= self.quadratic_min_obs and not flat,
            )
            if flat:
                # Several observations at one time: the quantile level is fitted, and the
                # slope prior alone decides the slope (the data cannot).
                level[idx] = beta[0]
                slope[idx] = mu_b
                curv[idx] = 0.0
                continue
            level[idx] = beta[0]
            slope[idx] = beta[1]
            curv[idx] = beta[2] if len(beta) == 3 else 0.0

        return _CurveParams(level, slope, curv, tbar, n_obs, t_all)

    # -- introspection --------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        return {
            "mu_b": round(self.mu_b_, 4),
            "sd_b": round(self.sd_b_, 4),
            "mu_c": round(self.mu_c_, 4),
            "sd_c": round(self.sd_c_, 4),
            "slope_prior": self.slope_prior_,
            "curv_prior": self.curv_prior_,
            "n_slope_prior": self.n_slope_prior_,
            "n_curv_prior": self.n_curv_prior_,
            "tau": self.tau,
            "k_offset": self.k_offset,
            "quadratic_min_obs": self.quadratic_min_obs,
            "prior_scale": self.prior_scale,
            "prior_scale_mode": "auto" if self.prior_scale == "auto" else "fixed",
            # What `transform` uses — the same number as `prior_scale` in fixed mode, and the
            # value the inner CV picked in auto mode. `prior_scale_auto_mse` is that CV's
            # per-candidate mean residual MSE, so a run can be asked *why* it picked it.
            "prior_scale_selected": self.prior_scale_,
            "prior_scale_auto_mse": self.auto_mse_,
            "auto_grid": list(self.auto_grid),
            "auto_folds": self.auto_folds,
            "auto_seed": self.auto_seed,
            "flat_level_fallback": self.flat_level_fallback,
        }
