"""Turning biased observations into an unbiased pseudo-target.

The observations are not the target. From the EDA and the literature review
(``reports/literature/parkinson_factors.md``):

* observed ``off`` sits about 6 points *below* the true OFF score — the residual of a
  long-duration response (LDR) that outlasts drug elimination by days to weeks;
* observed ``on`` sits about 22 points below, and the shortfall follows a saturating
  pharmacokinetic curve in ``time_since_intake_on`` (effect-site conduction half-life
  ≈0.4 h, Hill ≈2.5).

Both are corrections of the form ``target = observation + bias``, where the bias depends
on which score was observed. This transformer learns those two bias functions from the
training target and applies them to any rows:

    E[target − off | ledd missing?]     — two constants, because the off shortfall is
                                          much larger once the patient is medicated
                                          (−4.3 untreated vs −8.2 treated in the EDA)
    E[target − on  | time_since_intake] — a curve, because the ON shortfall collapses
                                          as the dose wears in

The result, ``pseudo_target``, is a single de-biased observation per visit. When a visit
carries both scores, the two corrected estimates are averaged; that is the only place
the two views are combined, and averaging is safe because at a single visit both scores
describe the same underlying state.

``duration_term`` adds a third correction the two above cannot express: the shortfall is
**not constant within a patient**. Measured on this training data, the ON shortfall grows
about 1.2 points per year of disease duration (and the OFF shortfall about 0.6) once the
patient's own mean is removed. The pharmacology predicts it — the concentration-effect
threshold rises as the disease advances, EC50 389 -> 711 ng/mL across stages
``[pkpd-levodopa#10]``, and rises further over four years of follow-up ``[#11]`` — so a
per-patient constant is the wrong functional form for the slow term.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT, TIME_SINCE_DIAGNOSIS
from .base import Transformer
from .ldr import LDR_HAT, PatientLdrModel

PSEUDO_TARGET = "pseudo_target"
DEBIAS_SOURCE = "debias_source"
BIAS_OFF_HAT = "bias_off_hat"
BIAS_ON_HAT = "bias_on_hat"


def _bin_curve(
    x: np.ndarray, values: np.ndarray, n_bins: int, min_bin: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Quantile-bin ``values`` over ``x``; return (bin medians, bin means) or None.

    The same non-parametric idiom as the ON-bias curve below: the shape is whatever the
    data says, and there is no commitment to a functional form.
    """
    ok = np.isfinite(x) & np.isfinite(values)
    if ok.sum() < min_bin * 2:
        return None
    edges = np.unique(np.quantile(x[ok], np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return None
    centers, means = [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        sel = ok & (x >= lo) & ((x <= hi) if last else (x < hi))
        if sel.sum() >= min_bin:
            centers.append(float(np.median(x[sel])))
            means.append(float(values[sel].mean()))
    if len(centers) < 2:
        return None
    return np.asarray(centers), np.asarray(means)


def _interp_curve(curve: tuple[np.ndarray, np.ndarray] | None, x: np.ndarray) -> np.ndarray:
    """Interpolate a binned curve at ``x``, flat outside the fitted range; 0 where x is NaN."""
    x = np.asarray(x, dtype=float)
    if curve is None:
        return np.zeros(len(x))
    centers, values = curve
    out = np.interp(np.nan_to_num(x, nan=centers[0]), centers, values)
    out[~np.isfinite(x)] = 0.0
    return out


class DebiasTransformer(Transformer):
    """Learn the observation→target bias functions and emit ``pseudo_target``.

    Requires the target at fit time; ``transform`` needs none, so it applies unchanged to
    the test set.
    """

    name = "debias"

    # Measured (tests/test_properties.py): `fit` reads y for the bias curves. `transform` is
    # row-wise — and so causal — under the incumbent chain's `use_patient_ldr=False`; with
    # the per-patient LDR model on (the class default) it reads whole patient frames and the
    # causal flag would flip. The declaration records the chain it is measured in.
    reads_target = True
    causal = True

    def __init__(
        self,
        n_bins_on: int = 8,
        min_bin: int = 50,
        use_patient_ldr: bool = True,
        ldr_alpha: float = 30.0,
        ldr_kind: str = "ridge",
        ldr_n_splits: int = 5,
        ldr_seed: int = 0,
        duration_term: bool = False,
    ) -> None:
        self.n_bins_on = n_bins_on
        self.min_bin = min_bin
        # The level-2 model's cross-fitting is the one source of randomness inside the
        # feature chain. It is reachable from YAML (feature steps take flat kwargs) rather
        # than pinned here, so that a feature-side seed is a setting like any other instead
        # of something a reader has to find in the source (docs/audit-2026-09-16.md P2-4).
        self.ldr_n_splits = ldr_n_splits
        self.ldr_seed = ldr_seed
        # Off by default, and a true A/B control: with it False the fitted curves below are
        # never built and every number is bit-identical to the constant-slow-term model.
        # Measured as a dead end (4.3017 against 4.2781 on the additive chain, CLAUDE.md §6).
        self.duration_term = duration_term
        self.n_bins_dur = 8
        # With ``use_patient_ldr`` the slow term becomes a per-patient estimate from the
        # level-2 model in features/ldr.py, and only the fast term stays global. Set it to
        # False to fall back to the flat, LEDD-stratified constant — useful for measuring
        # what the per-patient term actually buys.
        self.use_patient_ldr = use_patient_ldr
        self.ldr_alpha = ldr_alpha
        # "ridge" or "gbm" — see PatientLdrModel: the tree wins on its own metric and loses
        # end-to-end, so the ridge stays the default. `ldr_alpha` only affects the ridge.
        self.ldr_kind = ldr_kind

    # -- fit ------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if y is None:
            raise ValueError(
                "DebiasTransformer needs the training target to estimate the bias curves"
            )
        self.require(df, ["on", "off", "time_since_intake_on", "ledd"])
        y = pd.Series(np.asarray(y, dtype=float), index=df.index)

        self._fit_off_bias(df, y)
        self._fit_on_bias(df, y)

        # Split the ON bias into slow + fast: E[target−on | dt] = LDR + f(dt), so the
        # fast term is what is left after the (population-average) slow term is removed.
        self.fast_term_ = None
        if self.on_bias_curve_ is not None:
            centers, values = self.on_bias_curve_
            self.fast_term_ = (centers, values - self.off_bias_default_)

        self.ldr_model_ = None
        if self.use_patient_ldr:
            try:
                self.ldr_model_ = PatientLdrModel(
                    kind=self.ldr_kind,
                    alpha=self.ldr_alpha,
                    n_splits=self.ldr_n_splits,
                    seed=self.ldr_seed,
                ).fit(df, y)
            except Exception:  # noqa: BLE001 - a level-2 model is a bonus, not a requirement
                self.ldr_model_ = None

        # Must come after the level-2 model: it is fitted on what that model still gets
        # wrong, so the two terms stay separable instead of both absorbing the level.
        self.dur_off_curve_ = None
        self.dur_on_curve_ = None
        if self.duration_term:
            self._fit_duration_curves(df, y)
        return self

    def _fit_duration_curves(self, df: pd.DataFrame, y: pd.Series) -> None:
        """Within-patient drift of the shortfall along ``time_since_diagnosis``.

        Fitted on the **within-patient demeaned** residual of everything already fitted,
        so by construction it carries only the drift and adds nothing to the per-patient
        level. That separation is what makes it safe to add on top of the LDR term.
        """
        self.require(df, [PATIENT, TIME_SINCE_DIAGNOSIS])
        tsd = df[TIME_SINCE_DIAGNOSIS].to_numpy(dtype=float)
        patient = df[PATIENT].to_numpy()
        yy = np.asarray(y, dtype=float)
        dt = df["time_since_intake_on"].to_numpy(dtype=float)

        branches = {
            "off": (df["off"].to_numpy(dtype=float), self._slow_term(df, for_on=False)),
            "on": (
                df["on"].to_numpy(dtype=float),
                self._slow_term(df, for_on=True) + self._fast_term(dt),
            ),
        }
        for which, (obs, bias) in branches.items():
            m = np.isfinite(obs) & np.isfinite(yy) & np.isfinite(tsd)
            if m.sum() < self.min_bin * 2:
                continue
            resid = yy[m] - obs[m] - bias[m]
            resid = resid - pd.Series(resid).groupby(patient[m]).transform("mean").to_numpy()
            curve = _bin_curve(tsd[m], resid, self.n_bins_dur, self.min_bin)
            if which == "off":
                self.dur_off_curve_ = curve
            else:
                self.dur_on_curve_ = curve

    def _fit_off_bias(self, df: pd.DataFrame, y: pd.Series) -> None:
        m = df["off"].notna() & y.notna()
        if not m.any():
            raise ValueError("no rows with both `off` and target — cannot fit the OFF bias")
        bias = y[m] - df.loc[m, "off"]

        # Missing LEDD marks the early, still-untreated visits, where the residual drug
        # effect is smaller. Stratifying on it is worth ~4 points of accuracy.
        ledd_missing = df.loc[m, "ledd"].isna()
        self.off_bias_by_ledd_ = {
            bool(miss): float(bias[ledd_missing == miss].mean())
            if (ledd_missing == miss).sum() >= self.min_bin
            else float(bias.mean())
            for miss in (True, False)
        }
        self.off_bias_default_ = float(bias.mean())
        self.off_bias_n_ = int(m.sum())

    def _fit_on_bias(self, df: pd.DataFrame, y: pd.Series) -> None:
        m = df["on"].notna() & y.notna()
        if not m.any():
            raise ValueError("no rows with both `on` and target — cannot fit the ON bias")
        bias = (y[m] - df.loc[m, "on"]).to_numpy()
        dt = df.loc[m, "time_since_intake_on"].to_numpy(dtype=float)
        self.on_bias_default_ = float(np.nanmean(bias))

        have_dt = ~np.isnan(dt)
        self.on_bias_no_dt_ = (
            float(bias[~have_dt].mean()) if (~have_dt).sum() >= self.min_bin else self.on_bias_default_
        )

        # Quantile bins over time-since-intake, then interpolate between bin medians.
        # Binning rather than fitting a sigmoid keeps this non-parametric: the shape is
        # whatever the data says, and we are not committed to a functional form.
        self.on_bias_curve_: tuple[np.ndarray, np.ndarray] | None = None
        if have_dt.sum() >= self.min_bin * 2:
            edges = np.unique(np.quantile(dt[have_dt], np.linspace(0.0, 1.0, self.n_bins_on + 1)))
            if len(edges) >= 3:
                centers, values = [], []
                for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                    last = i == len(edges) - 2
                    sel = have_dt & (dt >= lo) & ((dt <= hi) if last else (dt < hi))
                    if sel.sum() >= self.min_bin:
                        centers.append(float(np.median(dt[sel])))
                        values.append(float(bias[sel].mean()))
                if len(centers) >= 2:
                    self.on_bias_curve_ = (np.asarray(centers), np.asarray(values))

    # -- transform ------------------------------------------------------------------
    def _on_bias(self, dt: np.ndarray) -> np.ndarray:
        """Interpolate the fitted ON-bias curve; flat at both ends."""
        if self.on_bias_curve_ is None:
            out = np.full(len(dt), self.on_bias_default_)
        else:
            centers, values = self.on_bias_curve_
            out = np.interp(np.nan_to_num(dt, nan=centers[0]), centers, values)
        # rows with no recorded intake time get the no-dt constant
        if np.any(np.isnan(dt)):
            out = np.asarray(out, dtype=float)
            out[np.isnan(dt)] = self.on_bias_no_dt_
        return np.asarray(out, dtype=float)

    def _fast_term(self, dt: np.ndarray) -> np.ndarray:
        """The fast PK shortfall alone, with the (average) slow term removed."""
        if self.fast_term_ is None:
            return np.full(len(dt), self.on_bias_default_ - self.off_bias_default_)
        centers, values = self.fast_term_
        out = np.interp(np.nan_to_num(dt, nan=centers[0]), centers, values)
        if np.any(np.isnan(dt)):
            out = np.asarray(out, dtype=float)
            out[np.isnan(dt)] = self.on_bias_no_dt_ - self.off_bias_default_
        return np.asarray(out, dtype=float)

    def _slow_term(self, df: pd.DataFrame, for_on: bool = False) -> np.ndarray:
        """Per-patient long-duration offset, with a constant fallback.

        The fallback differs by branch on purpose. For OFF rows the LEDD-stratified
        constant is genuinely better (the shortfall is −4.3 untreated vs −8.2 treated),
        and that is what the original single-part version used. For ON rows, the fitted
        ON-bias curve was estimated over the whole mixed population, so its own implicit
        slow term is the population average — using the stratified constant there would
        shift the ON correction against how the curve was fitted and measurably hurt.
        Keeping both branches faithful makes ``use_patient_ldr=False`` a true control.
        """
        ledd_missing = df["ledd"].isna().to_numpy()
        stratified = np.where(
            ledd_missing, self.off_bias_by_ledd_[True], self.off_bias_by_ledd_[False]
        )
        if self.ldr_model_ is None:
            return np.full(len(df), self.off_bias_default_) if for_on else stratified
        hat = self.ldr_model_.transform(df)
        # the level-2 model cannot say anything about a patient with no usable features
        return np.where(np.isfinite(hat), hat, stratified)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.require(df, ["on", "off", "time_since_intake_on", "ledd"])
        if self.duration_term:
            self.require(df, [TIME_SINCE_DIAGNOSIS])
        out = df.copy()

        off = out["off"].to_numpy(dtype=float)
        on = out["on"].to_numpy(dtype=float)
        dt = out["time_since_intake_on"].to_numpy(dtype=float)

        # Two-part correction: a per-patient slow term (the long-duration response) plus a
        # global fast term (the pharmacokinetic shortfall that fades within ~1.5 h). The
        # optional third part is the within-patient drift of both along disease duration.
        slow_off = self._slow_term(out, for_on=False)
        slow_on = self._slow_term(out, for_on=True)
        # Left as a scalar zero when the term is off, so the arithmetic below is unchanged
        # and the column is not even required on that path.
        dur = dur_on = 0.0
        if self.duration_term:
            tsd = out[TIME_SINCE_DIAGNOSIS].to_numpy(dtype=float)
            dur = _interp_curve(self.dur_off_curve_, tsd)
            dur_on = _interp_curve(self.dur_on_curve_, tsd)
        bias_off = slow_off + dur
        bias_on = slow_on + self._fast_term(dt) + dur_on

        has_off = ~np.isnan(off)
        has_on = ~np.isnan(on)
        if not (has_off | has_on).all():
            raise ValueError("every visit must carry at least one of `on` / `off`")

        est_off = np.where(has_off, off + bias_off, np.nan)
        est_on = np.where(has_on, on + bias_on, np.nan)

        pseudo = np.full(len(out), np.nan)
        source = np.full(len(out), "none", dtype=object)

        both = has_off & has_on
        pseudo[both] = 0.5 * (est_off[both] + est_on[both])
        source[both] = "both"

        only_off = has_off & ~has_on
        pseudo[only_off] = est_off[only_off]
        source[only_off] = "off"

        only_on = has_on & ~has_off
        pseudo[only_on] = est_on[only_on]
        source[only_on] = "on"

        out[PSEUDO_TARGET] = pseudo
        out[DEBIAS_SOURCE] = source
        out[BIAS_OFF_HAT] = bias_off
        out[BIAS_ON_HAT] = bias_on
        # The same array as `bias_off_hat` above, under the name the LDR module gives it.
        # Both reach the design matrix, which double-weights the slow term under
        # `colsample_bytree`; dropping one was measured *worse* (4.280684 → 4.282093), so
        # the redundancy is deliberate — see `features/matrix.py` and the test that pins
        # the pair. docs/audit-2026-09-16.md §2 P1-7.
        out[LDR_HAT] = slow_off
        return self.preserve_index(out, df)

    # -- introspection --------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        curve = self.on_bias_curve_
        fast = self.fast_term_
        def _curve(c: tuple[np.ndarray, np.ndarray] | None) -> list[tuple[float, float]] | None:
            return None if c is None else [
                (round(float(a), 2), round(float(b), 2)) for a, b in zip(*c)
            ]

        return {
            "duration_term": self.duration_term,
            "off_duration_curve": _curve(self.dur_off_curve_),
            "on_duration_curve": _curve(self.dur_on_curve_),
            "off_bias_ledd_missing": round(self.off_bias_by_ledd_[True], 3),
            "off_bias_ledd_present": round(self.off_bias_by_ledd_[False], 3),
            "on_bias_default": round(self.on_bias_default_, 3),
            "on_bias_curve": None
            if curve is None
            else [(round(float(c), 2), round(float(v), 2)) for c, v in zip(*curve, strict=True)],
            "fast_term_curve": None
            if fast is None
            else [(round(float(c), 2), round(float(v), 2)) for c, v in zip(*fast, strict=True)],
            "ldr_kind": self.ldr_kind,
            "ldr_n_splits": self.ldr_n_splits,
            "ldr_seed": self.ldr_seed,
            "patient_ldr": None if self.ldr_model_ is None else self.ldr_model_.get_params(),
        }
