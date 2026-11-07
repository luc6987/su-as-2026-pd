"""Forecast validation: predict a patient's *later* visits from their *earlier* ones.

Why this exists. The main cross-validation is honest across patients — no patient is ever
on both sides of a split — but it is **transductive within** a patient: the per-patient
curve is fitted on that patient's own visits, including the visit being predicted. Since a
patient's trajectory is one near-linear curve, a visit's own observation is a large part of
its own prediction. Nothing about that is unavailable at test time (the test visits carry
their own `on`/`off` too), so the number is legitimate — but it does not answer the
question a reader will actually ask: *how well does this forecast a visit we have not
seen?*

This module answers that. It splits patients, and for the evaluation patients it lets the
model see only their first ``n_known`` visits, then predicts the rest. Every patient-level
feature — the fitted curve, the drug-response offset, the visit counts — is carried forward
from the known visits only. Row-level features (`on`, `off`, time since intake) still come
from the visit being predicted, because those are genuinely observed at that visit.

The gap between this and the main CV is the price of not having seen the trajectory, and it
is the honest number to quote next to the leaderboard score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.loading import add_derived
from ..data.schema import PATIENT, TIME_SINCE_DIAGNOSIS, VISIT_IDX
from ..features.aggregates import AGGREGATE_COLS, OBS_MAX_P, OBS_REL_MAX, relative_to_max
from ..features.envelope import (
    ENV_CURV_LOG,
    ENV_LEVEL_LOG,
    ENV_PARAM_COLS,
    ENV_PRED,
    ENV_SLOPE_LOG,
    ENV_TBAR,
    OBS_MINUS_ENV,
    OFF_MINUS_ENV,
    evaluate_envelope,
)
from ..features.likelihood import (
    LIK_CURV_LOG,
    LIK_LEVEL_LOG,
    LIK_PARAM_COLS,
    LIK_PRED,
    LIK_SLOPE_LOG,
    LIK_TBAR,
    OBS_MINUS_LIK,
    evaluate_likelihood,
)
from ..features.longitudinal import CURVE_LEVEL_C, CURVE_TBAR
from ..features.pipeline import FeaturePipeline, build_pipeline
from ..features.pk_response import PK_AMP, PK_DELAY, PatientPKResponse, apply_pk
from ..features.ratio import OBS
from ..models.registry import build as build_model
from .metrics import summarize

# Patient-level quantities that must be derived from the *known* visits only. Anything
# computed per patient and fed to the model belongs here; miss one and the forecast leaks
# the trajectory it is supposed to be predicting.
PATIENT_LEVEL_COLS = [
    "curve_a",
    "curve_b",
    "curve_c",
    "curve_n_obs",
    "curve_pred",
    "curve_anchor_time",
    "curve_anchor_value",
    # the centred parameterization (`patient_curve` with `centre_time: true`): the level the
    # curve is actually parameterized by, and the mean visit time it is measured at
    CURVE_LEVEL_C,
    CURVE_TBAR,
    "ldr_hat",
    "bias_off_hat",
    "bias_on_hat",
    # patient aggregates from the longitudinal step, if a feature adds them
    "patient_off_mean",
    "patient_on_mean",
    # the structure-aware chain: the envelope curve's parameters, the per-patient
    # summaries of `obs`, and the per-patient PK amplitude. Its row-level columns
    # (`rho_hat`, `on_eq`, `obs`, and their `_pk` counterparts) come from the visit's own
    # scores and are *not* carried — but anything derived from a carried patient quantity
    # is rebuilt below.
    *ENV_PARAM_COLS,
    *AGGREGATE_COLS,
    *PatientPKResponse.patient_level_cols,
    # the likelihood step's posterior-mean parameters, carried the same way as the
    # envelope's: the curve they describe is rebuilt at the future visit's time below.
    *LIK_PARAM_COLS,
    # The `neighbour` step's columns are deliberately **not** here. They are per-visit, not
    # per-patient: a future visit's `prev{j}` is *its* predecessor, so carrying a constant
    # from the known visits would describe every future visit by the same row. They are
    # rebuilt instead, one visit at a time — see `_rebuild_neighbour`.
]


def _rebuild_neighbour(
    pipeline: FeaturePipeline, past_feat: pd.DataFrame, fut_feat: pd.DataFrame
) -> pd.DataFrame:
    """Recompute the future rows' neighbour columns from the *known* visits only.

    ``pipeline.transform(future)`` describes a future visit with the patient's other *future*
    visits: it sorts the frame it is handed by time and shifts within each patient, so the
    visit after the one being predicted becomes its ``next1``. That is exactly the
    transductive framing the leaderboard scores — and exactly what must not happen here. A
    visit we have not seen cannot be told about the visits it is being forecast alongside.

    So each future visit is described by ``known visits + this row`` instead: the same step,
    on a frame where the row has been appended after the patient's known visits, with only the
    appended row's values kept. What that comes to: ``prev{j}`` is the j-th last known visit,
    every ``next{*}`` is NaN (nothing after it is known), the time gaps point at known visits,
    and a window mean covers the known visits inside the window. Future visits never see each
    other, in either direction, and a patient's last known visit has no successors to hide.

    The known rows keep what the pipeline gave them: they are real observations with real
    neighbours among the other known visits, and their residual columns are the carried curve
    evaluated at a visit it was fitted on — which is what the model was trained against.

    Cost: one small ``transform`` per future row on ``n_known + 1`` rows — measured at 13 ms
    per row on a real fold frame, so ~80 s for the dev cohort's 5,801 future rows at k=5 and
    ~120 s for the 8,841 at k=3, on top of the chain fit. Almost all of it is pandas' fixed
    per-call overhead for the 46 columns, not the row count. Re-deriving the shifts and the
    window rule here instead would be faster and would also be a second implementation of
    them, free to drift from the step's own column order — which is the thing
    `tests/test_evaluation_forward.py` exists to prevent.
    """
    return _rebuild_rowwise(pipeline, past_feat, fut_feat, "neighbour")


def _rebuild_rowwise(
    pipeline: FeaturePipeline, past_feat: pd.DataFrame, fut_feat: pd.DataFrame, step_name: str
) -> pd.DataFrame:
    """Recompute one per-visit step's columns on the future rows from ``known visits + row``.

    The mechanism `_rebuild_neighbour` documents, for any step that describes a visit by the
    patient's *other* rows and exposes the names it writes as ``step.columns``. `floor` needs it
    too: its running maxima only look backwards, but on the future frame "backwards" includes
    the future visits before this one, which a forecast has not seen.
    """
    step = pipeline.step(step_name)
    cols = step.columns
    out = fut_feat.copy()
    known = {pid: g for pid, g in past_feat.groupby(PATIENT, sort=False)}
    values = []
    for i in range(len(out)):
        row = out.iloc[[i]]
        history = known.get(row[PATIENT].iloc[0])
        frame = row if history is None else pd.concat([history, row], axis=0)
        # `transform` restores the caller's order, so the appended row is last in the result.
        values.append(step.transform(frame)[cols].to_numpy(dtype=float)[-1])
    out[cols] = np.asarray(values)
    return out


def _step_column(pipeline: FeaturePipeline, step_name: str, attr: str, default: str) -> str:
    """The column a step was configured to read, so a rebuild follows the config.

    A chain that runs `on_ratio` → `pk_response` feeds the envelope and the aggregates
    `obs_pk` rather than `obs`; rebuilding their derived columns against `obs` would
    quietly mix the two scales.
    """
    try:
        return str(getattr(pipeline.step(step_name), attr))
    except KeyError:
        return default


# --- the forecast regression gate --------------------------------------------------------
# The forecast numbers are the honest ones the report quotes next to the leaderboard score,
# and nothing used to protect them: they live in prose, so a change that made the forecast
# materially worse would move a sentence rather than fail a test
# (docs/audit-2026-09-16.md §3 item 8). Written on `3a6ef23`; ported here with the metric
# convention of 09-18 (MSE) and re-measured on the merged code.
#
# Protocol: the default chain (derived → debias → patient_curve) with the gbm,
# ``truncate_train=False``, seed 0 — the "realistic" framing of `forward_validate`. Values
# are **MSE** (what `ForwardResult.mse` returns), re-measured on the merged code 2026-09-24:
# 369.9520 / 92.3858 / 58.7911, i.e. RMSE 19.2341 / 9.6118 / 7.6675. Those RMSEs are the
# branch's recorded 19.23 / 9.61 / 7.67 to the digit, so the port changed nothing about the
# measurement — only its unit.
#
# The tolerance is still the branch's 0.35 *points*: ``d(MSE) = 2·sqrt(MSE)·d``, so a fixed
# error in points is a different amount of MSE at each k, and `_tolerance` converts it per
# baseline rather than picking one MSE number to cover all three. It is a floor on quality,
# not a target to tune against.
FORWARD_REGRESSION_BASELINE = {2: 369.95, 3: 92.39, 5: 58.79}
FORWARD_REGRESSION_TOLERANCE_RMSE = 0.35


def _tolerance(baseline: float) -> float:
    """The 0.35-point tolerance expressed in MSE at this baseline's error level."""
    return 2.0 * float(np.sqrt(baseline)) * FORWARD_REGRESSION_TOLERANCE_RMSE


def check_regression(n_known: int, mse: float, *, tolerance: float | None = None) -> float:
    """Raise if ``mse`` is materially worse than the recorded forecast at this ``k``.

    Returns the remaining slack, so a caller can report how much headroom is left rather
    than only whether the gate passed.
    """
    if n_known not in FORWARD_REGRESSION_BASELINE:
        raise KeyError(
            f"no recorded forecast for k={n_known}; recorded: "
            f"{sorted(FORWARD_REGRESSION_BASELINE)}"
        )
    recorded = FORWARD_REGRESSION_BASELINE[n_known]
    tol = _tolerance(recorded) if tolerance is None else tolerance
    slack = recorded + tol - mse
    if slack < 0:
        raise AssertionError(
            f"forecast MSE at k={n_known} is {mse:.4f}, worse than the recorded "
            f"{recorded:.4f} by more than the {tol:.2f} tolerance — either the "
            f"forecast harness regressed or the recorded value is stale and needs re-measuring"
        )
    return slack


@dataclass
class ForwardResult:
    """Scores on the held-out future visits only."""

    summary: dict[str, float]
    predictions: pd.DataFrame  # one row per predicted future visit
    horizon: pd.DataFrame  # error vs how far ahead the visit is
    n_known: int
    fold_params: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mse(self) -> float:
        return self.summary["mse"]


def _split_patients(patients: np.ndarray, holdout_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(patients))
    n_eval = max(1, int(round(len(uniq) * holdout_frac)))
    return set(uniq[n_eval:]), set(uniq[:n_eval])


def forward_validate(
    feature_specs: list[Any],
    model_spec: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | pd.Series,
    n_known: int = 3,
    holdout_frac: float = 0.3,
    seed: int = 0,
    truncate_train: bool = True,
    verbose: bool = False,
) -> ForwardResult:
    """Fit, then forecast each evaluation patient's later visits from their first ``n_known``.

    ``truncate_train`` (default) also truncates the *fitting* patients to their first
    ``n_known`` visits. Without it the model trains on patients with full 4–12-visit
    trajectories and is then asked about patients it only sees 2 visits of, so
    ``curve_n_obs`` lands outside the range it ever saw and the score measures that
    surprise as much as the forecasting. Truncating both sides makes the comparison
    apples-to-apples: *given k visits per patient, how well do we do on the rest?*
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y, dtype=float))
    groups = np.asarray(groups)

    # We need a time axis to order each patient's visits. `add_derived` uses no target, so
    # computing it up front leaks nothing; the feature pipeline applies it again later and
    # is idempotent.
    if TIME_SINCE_DIAGNOSIS not in X.columns:
        X = add_derived(X)

    fit_patients, eval_patients = _split_patients(groups, holdout_frac, seed)
    fit_mask = np.isin(groups, list(fit_patients))
    eval_mask = np.isin(groups, list(eval_patients))

    fit_rows = X[fit_mask]
    if truncate_train:
        keep = (
            fit_rows.sort_values([PATIENT, TIME_SINCE_DIAGNOSIS], kind="stable")
            .groupby(PATIENT, sort=False)
            .head(n_known)
            .index
        )
        fit_rows = fit_rows.loc[keep]

    pipeline = build_pipeline(feature_specs)
    X_fit = pipeline.fit_transform(fit_rows, y[fit_rows.index])
    model = build_model(model_spec)
    model.fit(X_fit, y[fit_rows.index], groups=groups[fit_rows.index])
    if verbose:
        print(
            f"  fitted on {len(fit_rows):,} visits / {fit_patients.__len__():,} patients"
            f"{' (truncated to ' + str(n_known) + ' visits each)' if truncate_train else ''}"
        )

    # Within each evaluation patient, split their visits by time.
    ev = X[eval_mask].copy()
    ev_y = y[eval_mask].to_numpy()
    ev["__y"] = ev_y

    past_parts, future_parts = [], []
    for _, g in ev.groupby(PATIENT, sort=False):
        g = g.sort_values([TIME_SINCE_DIAGNOSIS, "age"], kind="stable")
        if len(g) <= n_known:
            continue
        past_parts.append(g.iloc[:n_known])
        future_parts.append(g.iloc[n_known:])
    if not future_parts:
        raise ValueError(f"no evaluation patient had more than {n_known} visits")

    past = pd.concat(past_parts)
    future = pd.concat(future_parts)

    past_feat = pipeline.transform(past.drop(columns="__y"))
    fut_feat = pipeline.transform(future.drop(columns="__y"))

    # Carry the patient-level block forward from the known visits. Only columns the
    # pipeline actually produced are carried, and `bias_on_hat` is excluded because it is
    # rebuilt below from the carried slow term plus the visit's own fast term.
    carry_cols = [
        c for c in PATIENT_LEVEL_COLS if c in past_feat.columns and c in fut_feat.columns
    ]
    carry_cols = [c for c in carry_cols if c != "bias_on_hat"]
    for required in ("curve_a", "curve_b", "curve_pred"):
        if required not in carry_cols:
            raise KeyError(f"the patient-curve step did not produce {required!r}")
    carried = past_feat.groupby(PATIENT)[carry_cols].first().add_suffix("__past")
    fut_feat = fut_feat.join(carried, on=PATIENT)

    # The base columns are replaced outright; the two that mix a patient term with a
    # row-level one are shifted by their row-level difference, so the fast (per-visit)
    # component is preserved while the slow (per-patient) one comes from the past.
    fut_feat["bias_on_hat"] = fut_feat["bias_off_hat__past"] + (
        fut_feat["bias_on_hat"] - fut_feat["bias_off_hat"]
    )
    for col in carry_cols:
        fut_feat[col] = fut_feat[f"{col}__past"]
    # The patient's drug response is a random effect estimated from the visits that carry
    # both scores; a future visit has to be converted with the amplitude its own patient
    # earned on the *known* visits, exactly as `curve_pred_anchored` is. Everything
    # downstream of `obs_pk` (the aggregates, the envelope, the model matrix) then follows.
    if PK_AMP in fut_feat.columns and PK_AMP in carry_cols:
        pk_step = pipeline.step("pk_response")
        # The conversion-noise downweight travels with the amplitude: the forecast re-runs
        # `apply_pk` by hand, so the per-class SDs have to be passed here too (task 01).
        sds = None
        if getattr(pk_step, "conversion_weight", False):
            sds = (getattr(pk_step, "sd_conv_known_", float("nan")),
                   getattr(pk_step, "sd_conv_na_", float("nan")))
        shift = fut_feat[PK_AMP].to_numpy(dtype=float)
        if getattr(pk_step, "effect_loading", False) or getattr(pk_step, "timing", False):
            # With the loading or the delay on, the carried amplitude is not the multiplier
            # `apply_pk` wants: a future visit converts with its own λ_v and s_v (its own
            # intake time). The delay is carried like the amplitude; the neighbour dt-slope
            # term stays uncarried, as before. `visit_shift` respects the flags, so a chain
            # with only one of the two on still rebuilds exactly what `transform` computed.
            delay = (fut_feat[PK_DELAY].to_numpy(dtype=float) if PK_DELAY in fut_feat.columns
                     else np.zeros(len(fut_feat)))
            shift = pk_step.visit_shift(fut_feat, shift, delay)
        fut_feat = apply_pk(
            fut_feat, shift,
            pk_step.rho_floor, pk_step.rho_ceiling, pk_step.on_weight, sds,
        )
    # The step's own time axis, not a hard-coded one: a centred curve is a function of
    # `t − t̄`, and the step may be fitted on `age` (`centre_time: true` + `time_col: age` is
    # what gives the 5% of patients with no `age_at_diagnosis` a curve at all), so evaluating
    # it on `time_since_diagnosis` would move every forecast by the difference of the two
    # origins. The envelope below needed the same repair.
    curve_time_col = _step_column(pipeline, "patient_curve", "time_col", TIME_SINCE_DIAGNOSIS)
    t_future = fut_feat[curve_time_col].fillna(0.0)
    # Curvature is 0 for any patient the curve step treated as a straight line, so this
    # collapses to the linear form rather than needing a separate branch.
    c_future = fut_feat["curve_c"].fillna(0.0) if "curve_c" in fut_feat else 0.0
    if CURVE_LEVEL_C in fut_feat and CURVE_TBAR in fut_feat:
        # Centred parameterization: the carried level is the level at the *known* visits'
        # mean time, so it only describes the future visit once offset by `t − t̄`. A
        # level-only patient (no time on the fitted axis) carries a NaN `t̄` and is evaluated
        # at their centre, which is flat — the same convention `evaluate_envelope` uses.
        tbar = fut_feat[CURVE_TBAR].to_numpy(dtype=float)
        u_future = np.where(np.isnan(tbar), 0.0, t_future.to_numpy(dtype=float) - tbar)
        fut_feat["curve_pred"] = (
            fut_feat[CURVE_LEVEL_C] + fut_feat["curve_b"] * u_future + c_future * u_future**2
        )
    else:
        fut_feat["curve_pred"] = (
            fut_feat["curve_a"] + fut_feat["curve_b"] * t_future + c_future * t_future**2
        )
    # The anchored extrapolation has to be recomputed at the *future* visit's time, using
    # the anchor carried from the known visits — not copied as a constant.
    if {"curve_anchor_time", "curve_anchor_value"} <= set(fut_feat.columns):
        t0 = fut_feat["curve_anchor_time"]
        fut_feat["curve_pred_anchored"] = (
            fut_feat["curve_anchor_value"]
            + fut_feat["curve_b"] * (t_future - t0)
            + c_future * (t_future**2 - t0**2)
        )
    # The envelope curve is re-evaluated at the future visit's time from the carried
    # parameters, and the columns derived from it and from the carried patient maximum
    # are rebuilt from the future visit's own observation.
    if ENV_PRED in fut_feat:
        env_step = pipeline.step("envelope_curve")
        k_offset = env_step.k_offset
        # The step's own time axis, not a hard-coded one: the curve is centred on the mean of
        # whatever `time_col` it was fitted on, so evaluating it on another axis would shift
        # every forecast by the difference of the two origins.
        fut_feat[ENV_PRED] = evaluate_envelope(
            fut_feat[ENV_LEVEL_LOG].to_numpy(dtype=float),
            fut_feat[ENV_SLOPE_LOG].to_numpy(dtype=float),
            fut_feat[ENV_CURV_LOG].to_numpy(dtype=float),
            fut_feat[ENV_TBAR].to_numpy(dtype=float),
            fut_feat[env_step.time_col].to_numpy(dtype=float),
            k_offset,
        )
        obs_col = _step_column(pipeline, "envelope_curve", "value_col", OBS)
        fut_feat[OBS_MINUS_ENV] = fut_feat[obs_col] - fut_feat[ENV_PRED]
        fut_feat[OFF_MINUS_ENV] = fut_feat["off"] - fut_feat[ENV_PRED]
    # Same for the likelihood step's posterior-mean curve: the parameters come from the
    # known visits, the evaluation time is the future visit's. The step itself defines
    # `obs_minus_lik` against the population `obs`, so the rebuild does too.
    if LIK_PRED in fut_feat:
        lik_step = pipeline.step("likelihood")
        fut_feat[LIK_PRED] = evaluate_likelihood(
            fut_feat[LIK_LEVEL_LOG].to_numpy(dtype=float),
            fut_feat[LIK_SLOPE_LOG].to_numpy(dtype=float),
            fut_feat[LIK_CURV_LOG].to_numpy(dtype=float),
            fut_feat[LIK_TBAR].to_numpy(dtype=float),
            fut_feat[lik_step.time_col].to_numpy(dtype=float),
            lik_step.k_offset,
        )
        fut_feat[OBS_MINUS_LIK] = fut_feat[OBS] - fut_feat[LIK_PRED]
    if OBS_REL_MAX in fut_feat:
        agg_col = _step_column(pipeline, "patient_aggregates", "obs_col", OBS)
        fut_feat[OBS_REL_MAX] = relative_to_max(
            fut_feat[agg_col].to_numpy(dtype=float), fut_feat[OBS_MAX_P].to_numpy(dtype=float)
        )
    fut_feat = fut_feat.drop(columns=[c for c in fut_feat.columns if c.endswith("__past")])

    # Last of the rebuilds, because the neighbour columns are built *from* the residual columns
    # the blocks above have just put back on the future rows. A chain without the step is
    # untouched, bit for bit.
    if any(step.name == "neighbour" for step in pipeline.steps):
        fut_feat = _rebuild_neighbour(pipeline, past_feat, fut_feat)
    # The running floors (`floor`) compare the bound with `lik_pred` / `env_pred`, which the
    # blocks above have just re-evaluated at the future visit's time; a future visit's floor
    # is its known visits' OFF plus its own, never another future visit's.
    if any(step.name == "floor" for step in pipeline.steps):
        fut_feat = _rebuild_rowwise(pipeline, past_feat, fut_feat, "floor")
    # The calibrated bounds (`edge`) are running maxima too, of per-visit quantiles that read the
    # `pk_response` conversion and `lik_pred` rebuilt above — the same rule, run after them. (A
    # `pk_response` with `dt_slope` converts a future visit with the carried patient amplitude
    # only: its per-visit intake-time term is not carried, which is the conservative direction.)
    if any(step.name == "edge" for step in pipeline.steps):
        fut_feat = _rebuild_rowwise(pipeline, past_feat, fut_feat, "edge")

    # A model that post-processes a patient's predictions *jointly* — `smoothed` blends each
    # patient's predictions with a curve fitted to all of them — must not do that here. Every
    # row of `fut_feat` is a visit we have not seen, so blending them together would describe
    # each one by the others' predictions, and those come from the other visits' own
    # observations. That is precisely the transductive shortcut this module exists to remove.
    # The forecast therefore asks for the raw prediction, and a smoothing wrapper is simply
    # its inner model under this framing (`SmoothedModel.predict_raw`). The hook is one level
    # deep on purpose: the wrapper is meant to be the outermost model, and a smoothing *member
    # inside an ensemble* would be reached through the member's own `predict` — which is why
    # `SmoothedModel`'s docstring says to wrap, not to be wrapped around.
    predict = getattr(model, "predict_raw", None) or model.predict
    pred = predict(fut_feat)

    out = pd.DataFrame(
        {
            "patient_id": future[PATIENT].to_numpy(),
            "y_true": future["__y"].to_numpy(),
            "y_pred": pred,
            "n_known": n_known,
            "n_future": future.groupby(PATIENT)[PATIENT].transform("size").to_numpy(),
            "visit_idx": future["visit_idx"].to_numpy()
            if "visit_idx" in future
            else np.arange(len(future)),
            "time_since_diagnosis": future[TIME_SINCE_DIAGNOSIS].to_numpy(),
        }
    )
    out["residual"] = out["y_pred"] - out["y_true"]
    out["steps_ahead"] = out["visit_idx"] - n_known + 1

    horizon = (
        out.groupby("steps_ahead")
        .agg(n=("residual", "size"), mse=("residual", lambda s: float((s**2).mean())), bias=("residual", "mean"))
        .reset_index()
    )
    summary = summarize(out["y_true"], out["y_pred"])
    if verbose:
        print(f"  forecasting {summary['n']:,} visits for {len(eval_patients):,} patients")

    return ForwardResult(
        summary=summary,
        predictions=out,
        horizon=horizon,
        n_known=n_known,
        fold_params=[{"features": pipeline.get_params(), "model": model.get_params()}],
    )
