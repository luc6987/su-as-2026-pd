"""Command line entry point.

    python -m pdchallenge.cli train --config configs/default.yaml
    python -m pdchallenge.cli train --model gbm --tag gbm-only
    python -m pdchallenge.cli compare --config configs/default.yaml
    python -m pdchallenge.cli segments
    python -m pdchallenge.cli holdout --config configs/consolidated.yaml --note "final check"

``train`` runs the whole chain and writes one self-contained run directory:
grouped cross-validation for an honest score, a refit on all training data, test
predictions, a submission file, diagnostics and SHAP attribution.

``holdout`` is the one command that reads the frozen evaluation holdout. Every other command
takes ``--dev-only`` (or ``dev_only: true`` in the config) and scores on the rest of the
patients, so a number used to *choose* something is never computed on the rows kept back to
*confirm* it. See ``docs/evaluation-protocol.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

# Selecting a backend is process-global, so the entry point does it before importing
# anything that pulls in pyplot. Library modules never do this at import time.
import matplotlib  # noqa: E402
import numpy as np
import pandas as pd

from . import paths
from .config import RunConfig, code_fingerprint
from .data.loading import imputed_age_mask, load_data
from .data.splits import dev_mask, load_holdout_patients
from .evaluation.comparison import compare, compare_on_common_rows
from .evaluation.cv import cross_validate, fit_full
from .evaluation.metrics import decompose_mse, paired_patient_bootstrap, segment_metrics, summarize
from .evaluation.tta import predict_with_tta
from .models.registry import with_seed

matplotlib.use("Agg")

from .viz import diagnostic, interpret  # noqa: E402


def _fmt(x: float) -> str:
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _seeded(cfg: RunConfig) -> Any:
    """``cfg.model`` with ``cfg.model_seed`` folded in.

    ``model_seed`` reaches a model only through its spec, so this is the one step that
    connects the two. The config itself is left alone, so the snapshot and the fingerprint
    still describe what the user wrote rather than the expansion of it.
    """
    return with_seed(cfg.model, cfg.model_seed)


# argparse expands `help=` with `%`-formatting, so a literal percent sign has to be doubled:
# a bare `% h` in this string made `train --help` (and compare/forward/segments) crash with a
# TypeError from argparse's formatter, which meant the `--config` help never printed either.
DEV_ONLY_HELP = (
    "score on the dev patients only — the frozen 15%% holdout is removed, so a number used "
    "to choose something is not computed on the rows kept back to confirm it "
    "(docs/evaluation-protocol.md)"
)

CONFIG_HELP = (
    'run config YAML; "current" resolves through configs/CURRENT to the incumbent chain '
    "(see STATUS.md)"
)


def resolve_config_path(value: Path | str | None) -> Path | None:
    """The config file a ``--config`` argument names.

    ``current`` is an alias for whatever ``configs/CURRENT`` names — one file to edit when the
    incumbent changes, instead of every command and document. Resolution happens at the CLI
    boundary so that everything downstream that records or prints the config (the holdout
    ledger's ``config_path``, the run directory's snapshot, the messages) sees the real file
    rather than the alias. ``None`` stays ``None``: no ``--config`` keeps the built-in defaults.
    """
    if value is None:
        return None
    if str(value) == "current":
        if not paths.CURRENT_CONFIG.is_file():
            raise FileNotFoundError(f"--config current: {paths.CURRENT_CONFIG} does not exist")
        name = paths.CURRENT_CONFIG.read_text(encoding="utf-8").strip()
        resolved = paths.CONFIGS / name
        if not resolved.is_file():
            raise FileNotFoundError(
                f"--config current: {paths.CURRENT_CONFIG} names {name!r}, which does not exist"
            )
        return resolved
    return Path(value)


def dev_subset(
    cfg: RunConfig,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Drop the frozen holdout patients when ``cfg.dev_only`` is set, else hand the frame back.

    Used by every scoring path — ``train``, ``compare``, ``segments`` and ``forward`` — so
    that one flag is enough to make any experiment honest. It never touches the final refit:
    deployment has no reason to discard 15% of the training patients, and ``train`` says so
    in its output when the flag is on.
    """
    if not cfg.dev_only:
        return X, y, groups
    holdout = load_holdout_patients()
    keep = dev_mask(groups, holdout)
    kept_groups = np.asarray(groups)[keep]
    print(
        f"  dev_only: {int(keep.sum()):,} of {len(keep):,} visits · "
        f"{len(np.unique(kept_groups)):,} of {len(np.unique(groups)):,} patients "
        f"(the frozen holdout is {len(holdout):,} patients — docs/evaluation-protocol.md)"
    )
    return (
        X.iloc[np.flatnonzero(keep)].reset_index(drop=True),
        pd.Series(np.asarray(y, dtype=float)[keep]),
        kept_groups,
    )


def write_submission(path: Path, index: np.ndarray, target: np.ndarray | pd.Series) -> None:
    """Write ``Index,target`` and then *re-read it* to prove it is what we intend.

    The organiser's example submission is two columns, ``Index,target``, one row per test
    visit, Index matching ``X_test`` row-for-row. Everything here is checked rather than
    assumed: a submission that is silently misaligned scores far worse than the model is
    worth, and the failure looks like "the model is bad" rather than "the file is wrong".

    Pass ``target`` as a ``Series`` indexed by ``Index`` and the pairing is made on the
    labels rather than on position — the one place where a prediction can be attached to
    the wrong row without anything else noticing. A bare array is still accepted (it is
    what the tests use) and is taken to be already in ``index`` order.
    """
    idx = np.asarray(index)
    if isinstance(target, pd.Series):
        missing = pd.Index(idx).difference(target.index)
        if len(missing):
            raise ValueError(
                f"{len(missing)} Index values have no prediction, e.g. {list(missing[:3])}"
            )
        target = target.reindex(idx).to_numpy(dtype=float)
    tgt = np.asarray(target, dtype=float)
    if len(idx) != len(tgt):
        raise ValueError(f"index/target length mismatch: {len(idx)} vs {len(tgt)}")
    if not np.isfinite(tgt).all():
        raise ValueError(f"{int((~np.isfinite(tgt)).sum())} non-finite predictions")
    if len(np.unique(idx)) != len(idx):
        raise ValueError("duplicate Index values in the submission")

    pd.DataFrame({"Index": idx, "target": np.round(tgt, 4)}).to_csv(path, index=False)

    # Read it back and re-assert against the source of truth.
    back = pd.read_csv(path)
    if list(back.columns) != ["Index", "target"]:
        raise AssertionError(f"unexpected columns written: {list(back.columns)}")
    if len(back) != len(idx):
        raise AssertionError(f"row count changed on write: {len(back)} vs {len(idx)}")
    if not np.array_equal(back["Index"].to_numpy(), idx):
        raise AssertionError("Index column did not survive the round trip")
    if back["target"].isna().any():
        raise AssertionError("NaN targets in the written submission")
    print(
        f"  submission verified: {len(back):,} rows · Index matches X_test row-for-row · "
        f"target {back['target'].min():.2f}-{back['target'].max():.2f} · no NaN"
    )


# ---------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------
def run_train(cfg: RunConfig, run_dir: Path | None = None) -> Path:
    run_dir = run_dir or paths.run_dir(cfg.tag)
    figs = paths.figures_dir(run_dir)
    print(f"→ {cfg.describe()}\n  run dir: {run_dir}")

    ds = load_data(derived=False)
    X, y, groups = ds.X_train, ds.y_train, ds.train_patients
    print(f"  train {X.shape[0]:,} visits / {len(np.unique(groups)):,} patients · test {ds.X_test.shape[0]:,}")

    # The organiser filled X_test's missing `age_at_diagnosis` with a constant and the
    # `derived` step reverts it. Say so on every run: it is invisible to the CV below (the
    # same rows are already NaN in X_train), so it is exactly the kind of intervention that
    # would otherwise be forgotten and misread as "the test set is simply more complete".
    repaired = imputed_age_mask(ds.X_test)
    if repaired.any():
        train_missing = ds.X_train["age_at_diagnosis"].isna().mean()
        print(f"  note: X_test ships `age_at_diagnosis` imputed for {int(repaired.sum()):,} rows "
              f"({ds.X_test.loc[repaired, 'patient_id'].nunique()} patients) — reverted to NaN, "
              f"matching the {train_missing:.2%} missing in X_train")

    # The score below is computed on the dev patients when `dev_only` is set. The refit that
    # produces the submission is not — it uses every training row, holdout included, because
    # the holdout exists to keep *selection* honest, not to make deployment worse.
    Xd, yd, gd = dev_subset(cfg, X, y, groups)

    # 1. honest score: the whole chain is rebuilt inside every fold
    print("\n[cv] grouped by patient")
    res = cross_validate(
        cfg.features, _seeded(cfg), Xd, yd, gd,
        n_splits=cfg.n_splits, seed=cfg.cv_seed, verbose=True, tta_scales=cfg.tta_scales,
    )
    print(f"  OOF MSE {res.summary['mse']:.4f} · MAE {res.summary['mae']:.4f} · bias {res.summary['bias']:+.3f} · R² {res.summary['r2']:.4f}")
    print(f"  error split: between-patient {res.summary['between_patient']:.3f} (level) · "
          f"within-patient {res.summary['within_patient']:.3f} (shape)")

    # 2. comparison against the naive references
    cmp_all = compare(yd, Xd, {"model": res.oof})
    cmp_common = compare_on_common_rows(yd, Xd, {"model": res.oof}, reference="raw_off")
    cmp_all.to_csv(run_dir / "comparison_all_rows.csv")
    cmp_common.to_csv(run_dir / "comparison_off_rows.csv")

    # 3. refit on everything and predict the test set
    print("\n[fit] full training set → test predictions")
    if cfg.dev_only:
        print(f"  note: the refit uses all {X.shape[0]:,} training visits / "
              f"{len(np.unique(groups)):,} patients — the holdout is held back from the score, "
              f"not from the submission")
    pipeline, model = fit_full(cfg.features, _seeded(cfg), X, y, groups=groups)
    X_test = pipeline.transform(ds.X_test)
    # Label the predictions from the frame that was actually transformed, not from the
    # one that went in: `write_submission` then pairs each prediction with its Index by
    # label, so if the two frames ever drift apart the write fails loudly instead of
    # silently scoring every row against another row's truth.
    test_pred = pd.Series(
        predict_with_tta(pipeline, model, ds.X_test, cfg.tta_scales, frame=X_test),
        index=X_test["Index"].to_numpy(),
    )

    if cfg.write_submission:
        write_submission(run_dir / paths.SUBMISSION_CSV, X_test["Index"].to_numpy(), test_pred)
        print(f"  → submit this file: {run_dir / paths.SUBMISSION_CSV}")

    # 4. diagnostics on the out-of-fold predictions
    #
    # The context columns (visit_idx, has_on/off, curve_b, pseudo_target) only exist after
    # the feature chain has run, so the reporting frame is the transformed one. Note the
    # *scores* below still come from res.oof — cross-validated, never from this frame — so
    # the full-data transform here cannot flatter the reported MSE.
    print("\n[report] figures")
    # Same rows the score was computed on, so every figure and segment describes the number
    # above it rather than a larger population that number never saw.
    X_report = pipeline.transform(Xd)
    pred_frame = diagnostic.make_prediction_frame(
        X_report, yd.to_numpy(), res.oof,
        extra=["time_since_diagnosis", "visit_idx", "n_visits", "has_on", "has_off",
               "cohort", "gene", "curve_b", "pseudo_target", "debias_source",
               "env_pred", "obs", "obs_is_off"],
    )
    # The actual-score band is the segment that matters most: an MSE-optimal model pulls
    # extreme values toward the centre, and this is where that shows up. On the current
    # baseline the top band (actual 80-110) carries a bias of roughly -9 points, so the
    # model is visibly conservative at the top of the range. Keep it visible.
    # The next two splits name populations the headline number hid for several rounds: the
    # 353 training patients with no `age_at_diagnosis` have no time axis, so `envelope_curve`
    # can only give them a flat level (21.9 MSE against 8.4 for everyone else), and a short
    # trajectory is scored from far less evidence than a long one. Both come from columns
    # already in the reporting frame, so reporting them costs nothing.
    X_report = X_report.assign(
        actual_band=pd.cut(yd.to_numpy(), bins=[-1, 20, 40, 60, 80, 200],
                           labels=["0-20", "20-40", "40-60", "60-80", "80+"]).astype(str),
        time_axis=np.where(X_report["time_since_diagnosis"].isna(), "missing", "present"),
        n_visits_band=pd.cut(X_report["n_visits"], bins=[0, 4, 11, 99],
                             labels=["4", "5-11", "12"]).astype(str),
    )
    seg = segment_metrics(
        X_report, yd, res.oof,
        by=["actual_band", "visit_idx", "has_on", "has_off", "cohort", "gene",
            "time_axis", "n_visits_band"],
    )
    seg.to_csv(run_dir / "segments.csv", index=False)
    # The guard fires on a band 2x the overall error *in points*, which is 4x in MSE —
    # squaring the multiplier keeps the note meaning what it meant before the unit switch.
    worst = seg[seg["segment"] == "actual_band"].sort_values("mse", ascending=False)
    if len(worst) and worst.iloc[0]["mse"] > 4 * res.summary["mse"]:
        print(f"  note: hardest band is actual_band={worst.iloc[0]['level']} "
              f"(MSE {worst.iloc[0]['mse']:.2f}, bias {worst.iloc[0]['bias']:+.2f})")
    # A segment carrying at least 1% of the rows at more than 2x the overall MSE is a
    # population the average is hiding, not a thin tail — those are always worth a line.
    notable = seg[(seg["n"] >= 0.01 * len(X_report)) & (seg["mse"] > 2 * res.summary["mse"])]
    for _, row in notable.iterrows():
        print(f"  note: {row['segment']}={row['level']} scores "
              f"MSE {row['mse']:.2f} over {row['n']:,} rows "
              f"({row['n'] / len(X_report):.1%} of the data, bias {row['bias']:+.2f})")
    diagnostic.run_all(pred_frame, figs, seg=seg)

    if cfg.explain:
        try:
            table = interpret.write_report(model, X_report, figs / "shap")
            if table is not None:
                print("  top SHAP features:")
                for name, row in table.head(6).iterrows():
                    print(f"    {name:22s} {row['mean_abs_shap']:6.3f} pts  ({row['share']*100:4.1f}%)")
        except Exception as exc:  # noqa: BLE001 - attribution is a bonus, not the job
            print(f"  SHAP skipped: {type(exc).__name__}: {exc}")

    # 5. persist everything needed to reproduce or inspect this run
    cfg.to_yaml(run_dir / paths.CONFIG_SNAPSHOT)
    res.fold_metrics.to_csv(run_dir / "cv_folds.csv", index=False)
    pred_frame.to_csv(run_dir / paths.PREDICTIONS_CSV, index=False)
    summary_row = {
        "tag": cfg.tag,
        # Recorded next to the score, because it decides what the score is over: the two
        # populations differ by 15% of the patients and their MSEs are not comparable.
        "dev_only": cfg.dev_only,
        "fingerprint": cfg.fingerprint(),
        "config_fingerprint": cfg.config_fingerprint(),
        "code_fingerprint": code_fingerprint(),
        **res.summary,
    }
    pd.DataFrame([summary_row]).set_index("tag").to_csv(run_dir / paths.METRICS_CSV)

    _write_summary(run_dir, cfg, res, cmp_all, cmp_common, model, pipeline)
    print(f"\ndone → {run_dir}")
    return run_dir


def _population_line(cfg: RunConfig) -> str:
    """Which patients the headline number covers — the first thing a reader has to know."""
    if cfg.dev_only:
        return (
            "**dev patients only** — the frozen 15% holdout is removed, so this score is "
            "comparable to other dev scores and *not* to a full-training-set CV "
            "(`docs/evaluation-protocol.md`)"
        )
    return (
        "all training patients, holdout included — a number from before the holdout "
        "existed, or an explicit full-set run; see `docs/evaluation-protocol.md`"
    )


def _write_summary(run_dir, cfg, res, cmp_all, cmp_common, model, pipeline) -> None:
    md = [
        f"# Run: {cfg.tag}",
        "",
        f"- **fingerprint**: `{cfg.fingerprint()}` "
        f"(config `{cfg.config_fingerprint()}` · code `{code_fingerprint()}`)",
        f"- **model**: `{json.dumps(cfg.model, default=str)}`",
        f"- **features**: `{cfg.features}`",
        f"- **cv**: {cfg.n_splits} folds grouped by patient, seed {cfg.cv_seed}",
        f"- **evaluation population**: {_population_line(cfg)}",
        "",
        "## Headline",
        "",
        f"Out-of-fold **MSE {res.summary['mse']:.4f}**  ·  "
        f"MAE {res.summary['mae']:.4f}  ·  bias {res.summary['bias']:+.4f}  ·  "
        f"R² {res.summary['r2']:.4f}  (n = {res.summary['n']:,})",
        "",
        "_Error split_: between-patient "
        f"**{res.summary['between_patient']:.3f}** (getting each patient's level right) vs "
        f"within-patient **{res.summary['within_patient']:.3f}** (getting the trajectory shape "
        "right) — the two add up to the headline MSE exactly, which is a property of the "
        "squared metric. The between term is bounded by the one-sided per-visit deficit averaged "
        "over a patient's few visits. What is patient-structured sits on the *ON* side — the "
        "drug-response amplitude `pk_response` divides out when the chain includes it — see "
        "`docs/diagnosis-2026-09-18.md` and `docs/pk-response-2026-09-23.md`.",
        "",
        "## Against the naive references",
        "",
        "All rows:",
        "",
        cmp_all[["n", "coverage", "mse", "mae", "bias"]].to_markdown(),
        "",
        "Restricted to visits that carry an `off` score — the only fair comparison with `raw_off`, "
        "which is otherwise scored on its easiest 58%:",
        "",
        cmp_common[["n", "coverage", "mse", "mae", "bias"]].to_markdown(),
        "",
        "## Fitted parameters",
        "",
        "```json",
        json.dumps(pipeline.get_params(), indent=2, default=str),
        "```",
        "",
        "```json",
        json.dumps(model.get_params(), indent=2, default=str),
        "```",
        "",
        "## Figures",
        "",
        "See `figures/` — diagnostics are indexed in `figures/diagnostics.md`.",
    ]
    (run_dir / paths.SUMMARY_MD).write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------------
# compare — score several model specs under one CV, in one process
# ---------------------------------------------------------------------------------
def run_forward(cfg: RunConfig, ks: list[int]) -> pd.DataFrame:
    """Forecast each evaluation patient's later visits from their first k.

    This is the honest companion to the main CV: the CV lets a patient's curve see every
    visit including the one being predicted, which flatters the score in a way a reader
    will not expect. The gap between the two is the price of not having seen the
    trajectory.
    """
    from .evaluation.forward import forward_validate

    ds = load_data(derived=False)
    X, y, groups = dev_subset(cfg, ds.X_train, ds.y_train, ds.train_patients)

    # Two framings, because they answer different questions and each is misleading alone.
    #
    # realistic — fit on complete trajectories (as in deployment), forecast a new patient
    #             from their first k visits. It is pessimistic at small k because the model
    #             then sees `curve_n_obs` values it never trained on.
    # matched   — truncate the fitting patients to k visits too, so both sides look alike.
    #             Removes that artefact but asks a question nobody faces: a model that has
    #             only ever seen short trajectories.
    rows = []
    for k in ks:
        rec: dict[str, float] = {"k_known_visits": k}
        for label, truncate in (("realistic", False), ("matched", True)):
            res = forward_validate(
                cfg.features, _seeded(cfg), X, y, groups,
                n_known=k, seed=cfg.cv_seed, truncate_train=truncate,
            )
            rec[f"{label}_mse"] = res.summary["mse"]
            rec[f"{label}_bias"] = res.summary["bias"]
            rec["n_forecast"] = res.summary["n"]
            if label == "realistic" and k == ks[-1]:
                res.horizon.to_csv(
                    paths.report_dir("baseline") / "forward_horizon.csv", index=False
                )
        rows.append(rec)
        print(
            f"  k={k}: realistic MSE {rec['realistic_mse']:8.4f} (bias {rec['realistic_bias']:+6.2f})"
            f"   matched MSE {rec['matched_mse']:8.4f} (bias {rec['matched_bias']:+6.2f})"
            f"   n={rec['n_forecast']:,}"
        )

    table = pd.DataFrame(rows).set_index("k_known_visits")
    out = paths.report_dir("baseline") / "forward_validation.csv"
    table.to_csv(out)
    print(f"\nwrote {out}")
    return table


# ---------------------------------------------------------------------------------
# segments — does cutting the cohort into regimes beat one model for all of it?
# ---------------------------------------------------------------------------------
# The three cuts the method survey named, plus one interaction of them. Bin edges are read
# off the training distribution rather than guessed: `visit_idx` is dense for 0-3 and thins
# out after, and `n_visits` is bimodal at 4 and at 12.
SEGMENT_SCHEMES: dict[str, dict[str, Any]] = {
    "obs_type": {"by": ["has_on", "has_off"]},
    "visit_idx": {"by": "visit_idx", "edges": {"visit_idx": [2.0, 6.0]}},
    "n_visits": {"by": "n_visits", "edges": {"n_visits": [6.0, 11.0]}},
    "obs_x_time": {
        "by": ["has_on", "has_off", "visit_idx"],
        "edges": {"visit_idx": [2.0, 6.0]},
    },
}


def run_segments(
    cfg: RunConfig,
    schemes: dict[str, dict[str, Any]],
    seeds: Sequence[int] = (0,),
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Score each way of cutting the cohort against the unsegmented model.

    Same features, same base model, same folds within a seed — so the only thing that
    differs down the table is where the cohort was cut. That is the point of the command:
    segmentation costs a model per regime and is only worth it if it beats sharing one, and
    the answer is not obvious a priori because a boosted tree can already split on every
    column used as a key here.

    Two guards against reading noise as a result, because a difference of a few tenths of an
    MSE point (hundredths of a point on the score scale) is exactly the size that looks
    convincing on one fold assignment:

    * ``seeds`` repeats the whole comparison under different CV seeds. A gain that changes
      sign between them was never there.
    * every scheme gets a **paired patient-level bootstrap** against the unsegmented model
      (see :func:`~pdchallenge.evaluation.metrics.paired_patient_bootstrap`). The interval
      is what tells you whether the point estimate means anything.
    """
    ds = load_data(derived=False)
    X, y, groups = dev_subset(cfg, ds.X_train, ds.y_train, ds.train_patients)

    specs: dict[str, Any] = {"unsegmented": _seeded(cfg)}
    for name, kwargs in schemes.items():
        specs[f"segmented:{name}"] = {"name": "segmented", "base": _seeded(cfg), **kwargs}

    rows: list[dict[str, Any]] = []
    for name, spec in specs.items():
        per_seed: dict[int, float] = {}
        first: dict[str, Any] = {}
        oof_first: np.ndarray | None = None
        fitted: set[str] = set()
        fell_back: set[str] = set()

        for seed in seeds:
            res = cross_validate(
                cfg.features, spec, X, y, groups, n_splits=cfg.n_splits, seed=seed
            )
            per_seed[seed] = res.summary["mse"]
            if oof_first is None:
                oof_first = res.oof
                first = res.summary
                # Which regimes actually got their own model and which fell through to the
                # global one: a scheme that quietly falls back everywhere is a no-op
                # wearing a result's clothes.
                fitted = {lvl for p in res.fold_params for lvl in p["model"].get("fitted", {})}
                fell_back = {lvl for p in res.fold_params for lvl in p["model"].get("fallback", {})}

        rows.append(
            {
                "scheme": name,
                "mse": float(np.mean(list(per_seed.values()))),
                "bias": first["bias"],
                "between_patient": first["between_patient"],
                "within_patient": first["within_patient"],
                "n_segments_fitted": len(fitted),
                "n_segments_fallback": len(fell_back),
                **{f"mse_seed{s}": v for s, v in per_seed.items()},
                "_oof": oof_first,
            }
        )
        show = " ".join(f"{v:.4f}" for v in per_seed.values()) if len(per_seed) > 1 else ""
        print(
            f"  {name:20s} MSE {_fmt(rows[-1]['mse'])}"
            + (f"  seeds {show}" if show else "")
            + f"  ·  {len(fitted)} segments fitted, {len(fell_back)} fell back"
        )

    reference_oof = rows[0]["_oof"]
    for row in rows:
        boot = paired_patient_bootstrap(y, reference_oof, row["_oof"], groups, n_boot=n_boot)
        row.update(
            {
                "delta_vs_unsegmented": boot["delta_mse"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "p_better": boot["p_b_better"],
            }
        )

    table = pd.DataFrame(rows).drop(columns=["_oof"]).set_index("scheme")
    table = table.sort_values("mse")
    out = paths.report_dir("baseline") / "segmented_comparison.csv"
    table.to_csv(out)

    print("\n  scheme                  mse     delta        95% CI          P(better)  n/fallback")
    for name, r in table.iterrows():
        print(
            f"  {name:20s} {r['mse']:8.4f}  {r['delta_vs_unsegmented']:+.4f}  "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {r['p_better']:.3f}      "
            f"{int(r['n_segments_fitted'])}/{int(r['n_segments_fallback'])}"
        )

    best = str(table.index[0])
    winners = int((table["delta_vs_unsegmented"] < 0).sum())
    print(
        f"\nbest: {best}  MSE {table.loc[best, 'mse']:.4f}  "
        f"({table.loc[best, 'delta_vs_unsegmented']:+.4f} vs unsegmented, "
        f"95% CI [{table.loc[best, 'ci_low']:+.4f}, {table.loc[best, 'ci_high']:+.4f}])"
    )
    # An interval that straddles zero is the whole verdict, so say it rather than leaving
    # the reader to compare two numbers by eye.
    if table.loc[best, "ci_low"] < 0 < table.loc[best, "ci_high"]:
        print("  the interval straddles 0 — this difference is not established")
    print(f"  {winners} of {len(table) - 1} schemes beat the unsegmented model")
    print(f"wrote {out}")
    return table


def run_compare(cfg: RunConfig, specs: dict) -> pd.DataFrame:
    ds = load_data(derived=False)
    X, y, groups = dev_subset(cfg, ds.X_train, ds.y_train, ds.train_patients)

    oofs, rows = {}, []
    for name, spec in specs.items():
        res = cross_validate(cfg.features, spec, X, y, groups, n_splits=cfg.n_splits, seed=cfg.cv_seed)
        oofs[name] = res.oof
        rows.append({"model": name, **res.summary})
        print(f"  {name:14s} MSE {_fmt(res.summary['mse'])}  bias {res.summary['bias']:+.3f}")

    combined = compare(y, X, oofs)
    print("\n" + combined[["n", "coverage", "mse", "mae", "bias", "r2"]].to_string(float_format=_fmt))
    out = paths.report_dir("baseline") / "model_comparison.csv"
    combined.to_csv(out)
    print(f"\nwrote {out}")
    return combined


# ---------------------------------------------------------------------------------
# diagnose — the submission smoke test (docs/audit-2026-09-16.md §1.5, §3 item 2)
# ---------------------------------------------------------------------------------
def run_diagnose(run_dir: Path | None = None) -> pd.DataFrame:
    """Write calibration submissions whose leaderboard score is known in advance.

    The 2026-09-16 investigation could not tell "the model is bad" from "the file or the
    scoring is misaligned", and it took a day to separate them. It should have taken one
    upload: a *constant* prediction is immune to row misalignment and its score is fixed by
    the scoring target's own mean and variance, so it measures the harness and nothing else.
    These three files are that check, regenerable on demand.

    ``d1`` is the decisive one — a constant has no rows to misalign, so its score reports
    the scoring target's distribution with no model in the way.
    """
    ds = load_data(derived=False)
    out = paths.report_dir("diagnostics")
    index = ds.X_test["Index"].to_numpy()
    mean = float(ds.y_train.mean())
    sd = float(ds.y_train.std())

    # RMSE(c)² = SD(truth)² + (c − mean(truth))² — which is what makes these informative.
    rows = [
        {"file": "d1_constant_mean.csv", "constant": mean,
         "expected_rmse_if_target_is_true_off": float(np.sqrt(sd**2 + 0.0))},
        {"file": "d2_constant_zero.csv", "constant": 0.0,
         "expected_rmse_if_target_is_true_off": float(np.sqrt(sd**2 + mean**2))},
    ]
    write_submission(out / "d1_constant_mean.csv", index, np.full(len(index), mean))
    write_submission(out / "d2_constant_zero.csv", index, np.zeros(len(index)))

    # d3 keeps the real predictions and their Index column but reorders the rows. If the
    # board joins on Index it scores the same as the original; if it joins on row position
    # it collapses, which is exactly the ambiguity worth ruling out.
    source = run_dir or _latest_run_dir()
    if source is not None and (source / paths.SUBMISSION_CSV).is_file():
        real = pd.read_csv(source / paths.SUBMISSION_CSV)
        patients = ds.X_test["patient_id"].to_numpy()
        order = np.argsort(patients, kind="stable")
        reordered = real.iloc[order]
        reordered.to_csv(out / "d3_real_sorted_by_patient.csv", index=False)
        print(f"  d3 taken from {source.name} and reordered by patient_id "
              f"({len(reordered):,} rows; the Index column is untouched)")
    else:
        print("  d3 skipped: no run directory with a submission.csv to reorder")

    table = pd.DataFrame(rows)
    print("\nEach file is 23,672 rows with Index matching X_test, so any of them can be uploaded:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nHow to read the result (the board reports MSE — see docs/audit-2026-09-16.md §1.0):\n"
        f"  d1 ≈ {np.sqrt(sd**2):.2f} RMSE  → the scoring target is true OFF and the harness is sound\n"
        f"  d1 much larger        → the scoring target is not the variable `y_train` measures\n"
        f"  d3 ≈ the original score → the board joins on the Index column\n"
        f"  d3 far worse          → the board joins on row position, which would explain a lot"
    )
    table.to_csv(out / "README.csv", index=False)
    print(f"\nwrote {out}")
    return table


def _latest_run_dir() -> Path | None:
    """The most recently written run directory, or ``None`` if there are none."""
    if not paths.BASELINE_REPORTS.is_dir():
        return None
    runs = [d for d in paths.BASELINE_REPORTS.iterdir() if (d / paths.SUBMISSION_CSV).is_file()]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


# ---------------------------------------------------------------------------------
# ledger — what was uploaded, and what the board said (docs/audit-2026-09-16.md §3.1)
# ---------------------------------------------------------------------------------
def run_ledger(args: argparse.Namespace) -> int:
    """Show the submission ledger, or record one submission into it."""
    from . import submissions

    if args.run is None:
        ledger = submissions.read()
        if ledger.empty:
            print("the ledger is empty — record a submission with `ledger --run <run_dir>`")
            return 0
        # The board columns are the point of the table, so lead with them.
        print(ledger.to_string(index=False))
        return 0

    ledger = submissions.record(
        args.run, public=args.public, private=args.private, note=args.note or ""
    )
    row = submissions.find(ledger, args.run)
    print(f"recorded {row['run_dir']}")
    print(f"  file         {row['file']}  ({row['n_rows']:,} rows, "
          f"pred mean {row['pred_mean']}, sd {row['pred_sd']})")
    print(f"  fingerprints {row['fingerprint']}  "
          f"(config {row['config_fingerprint']} · code {row['code_fingerprint']})")
    print(f"  board        public {row['board_public']} · private {row['board_private']}")
    if pd.isna(row["board_public"]) and pd.isna(row["board_private"]):
        print("  (no board score yet — add one with `--public <value>` after uploading)")
    return 0


# ---------------------------------------------------------------------------------
# holdout — the one number that selection is not allowed to look at
# ---------------------------------------------------------------------------------
# Kept as an explicit list rather than inferred from whatever the frame happens to hold: the
# ledger is the record of a decision, and a column that moved between two versions of the
# code would make two rows incomparable in a way nobody would notice.
HOLDOUT_LEDGER_COLUMNS = [
    "timestamp",
    "config_path",
    "tag",
    "config_fingerprint",
    "code_fingerprint",
    "fingerprint",
    "look_index",
    "mse",
    "between_mse",
    "within_mse",
    "bias",
    "mae",
    "n_rows",
    "n_patients",
    "preds_path",
    "note",
]

HOLDOUT_WARNING = (
    "⚠ holdout 只用于最终确认，不能用来挑参数 — "
    "final confirmation only, never a selector (docs/evaluation-protocol.md)"
)


def read_holdout_ledger(path: Path | None = None) -> pd.DataFrame:
    """The holdout ledger, or an empty frame with the right columns if there is none yet."""
    path = Path(path) if path is not None else paths.HOLDOUT_LEDGER_CSV
    if not path.is_file():
        return pd.DataFrame(columns=HOLDOUT_LEDGER_COLUMNS)
    return pd.read_csv(path)


def _relative_to_root(path: Path) -> str:
    """A repo-relative path when possible, so the ledger reads the same on another machine."""
    try:
        return str(Path(path).relative_to(paths.ROOT))
    except ValueError:
        return str(path)


def run_holdout(
    cfg: RunConfig,
    note: str = "",
    allow_relook: bool = False,
    config_path: Path | None = None,
    stamp: str | None = None,
) -> pd.DataFrame:
    """Fit on the dev patients, score the frozen holdout, and record the look.

    The refit deliberately ignores ``cfg.dev_only``: this command *is* the holdout
    evaluation, so it always fits on dev rows and always scores the patients those rows
    exclude. The alternative — letting the flag through — would let a config that says
    ``dev_only: true`` be scored on a holdout that had already been trained on.

    A configuration is meant to be looked at **once**. A second look is refused unless
    ``allow_relook`` is passed, because the holdout's whole value is that it stayed unread
    while the choices were being made, and that value is spent by looking. The refusal is
    keyed on the **config** fingerprint, not the full one: a code change that moves the
    number is a reason to look again, but it is not a reason to look twice at the same
    configuration without recording why.
    """
    print(HOLDOUT_WARNING)
    config_key = cfg.config_fingerprint()
    ledger = read_holdout_ledger()
    previous = ledger[ledger["config_fingerprint"] == config_key]
    if len(previous) and not allow_relook:
        print(
            f"\nrefusing to re-run: config fingerprint {config_key} has already been scored "
            f"on the holdout {len(previous)} time(s):"
        )
        print(
            previous[
                ["timestamp", "tag", "look_index", "mse", "between_mse", "within_mse",
                 "bias", "note"]
            ].to_string(index=False)
        )
        print(
            "\n每看一次，holdout 就少一分独立性。If this look is deliberate (a code change, "
            "or the final submission), re-run with --allow-relook and say why in --note."
        )
        raise SystemExit(1)

    ds = load_data(derived=False)
    X, y, groups = ds.X_train, ds.y_train, ds.train_patients
    holdout = load_holdout_patients()
    keep = dev_mask(groups, holdout)
    if not keep.any() or keep.all():
        raise ValueError(
            f"the frozen holdout matches {int((~keep).sum()):,} of {len(keep):,} training "
            f"rows — the list and the training patients do not describe the same cohort"
        )
    dev_idx, hold_idx = np.flatnonzero(keep), np.flatnonzero(~keep)
    groups_hold = groups[hold_idx]
    print(
        f"  fit on {len(dev_idx):,} dev visits / {len(np.unique(groups[dev_idx])):,} patients · "
        f"scored on {len(hold_idx):,} holdout visits / {len(np.unique(groups_hold)):,} patients"
    )

    pipeline, model = fit_full(
        cfg.features, _seeded(cfg), X.iloc[dev_idx], y.iloc[dev_idx], groups=groups[dev_idx]
    )
    X_hold = X.iloc[hold_idx].reset_index(drop=True)
    y_hold = np.asarray(y, dtype=float)[hold_idx]
    X_hold_t = pipeline.transform(X_hold)
    # The chain is allowed to reorder rows internally, so check it did not: `y_hold` is
    # positional, and a silent reorder here would score every row against another row's
    # truth — the failure mode that looks like "the model is bad" rather than "the file is".
    if "Index" not in X_hold_t.columns:
        raise KeyError("the feature chain dropped the Index column; predictions cannot be aligned")
    if not np.array_equal(X_hold_t["Index"].to_numpy(), X_hold["Index"].to_numpy()):
        raise AssertionError("the feature chain reordered the holdout rows — y would be misaligned")
    pred = predict_with_tta(pipeline, model, X_hold, cfg.tta_scales, frame=X_hold_t)

    metrics = summarize(y_hold, pred)
    split = decompose_mse(y_hold, pred, groups_hold)
    preds_path = paths.holdout_preds_path(cfg.tag, stamp)
    pd.DataFrame(
        {
            "Index": X_hold_t["Index"].to_numpy(),
            "patient_id": X_hold_t["patient_id"].to_numpy(),
            "y_true": y_hold,
            "y_pred": pred,
        }
    ).to_csv(preds_path, index=False)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": _relative_to_root(config_path) if config_path else "(defaults)",
        "tag": cfg.tag,
        "config_fingerprint": config_key,
        "code_fingerprint": code_fingerprint(),
        "fingerprint": cfg.fingerprint(),
        "look_index": int(len(previous)) + 1,
        "mse": metrics["mse"],
        "between_mse": split["between_patient"],
        "within_mse": split["within_patient"],
        "bias": metrics["bias"],
        "mae": metrics["mae"],
        "n_rows": metrics["n"],
        "n_patients": int(len(np.unique(groups_hold))),
        "preds_path": _relative_to_root(preds_path),
        "note": note,
    }
    # Appended by rewriting the whole file: it is a handful of rows, and pandas' empty-frame
    # concat path warns about the dtypes it is about to guess. The column list is pinned so
    # the file's shape is the same whether or not a ledger existed before.
    appended = pd.DataFrame([row], columns=HOLDOUT_LEDGER_COLUMNS)
    ledger = appended if ledger.empty else pd.concat([ledger, appended], ignore_index=True)
    Path(paths.HOLDOUT_LEDGER_CSV).parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(paths.HOLDOUT_LEDGER_CSV, index=False)

    print(
        f"\n  holdout MSE {row['mse']:.4f} (RMSE {np.sqrt(row['mse']):.4f}) · "
        f"bias {row['bias']:+.3f} 分 · between {row['between_mse']:.3f} · within {row['within_mse']:.3f}"
    )
    print(f"  look #{row['look_index']} of this config on the frozen holdout")
    print(f"  ledger → {_relative_to_root(Path(paths.HOLDOUT_LEDGER_CSV))}")
    print(f"  preds  → {row['preds_path']}")
    print(
        "\n  This number is a confirmation, not a ranking: it exists to be quoted once, next to "
        "the dev score, and it is now spent for this configuration."
    )
    return pd.DataFrame([row])


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    """The config a subcommand was asked for, with ``--dev-only`` folded in."""
    cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()
    if getattr(args, "dev_only", False):
        cfg.dev_only = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pdchallenge", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="cross-validate, refit, predict the test set, write a report")
    t.add_argument("--config", type=Path, default=None, help=CONFIG_HELP)
    t.add_argument("--tag", default=None)
    t.add_argument("--model", default=None, help="model name, overriding the config")
    t.add_argument("--features", nargs="*", default=None, help="feature names, overriding the config")
    t.add_argument("--no-explain", action="store_true")
    t.add_argument("--out", type=Path, default=None)
    t.add_argument("--dev-only", action="store_true", help=DEV_ONLY_HELP)

    c = sub.add_parser("compare", help="score several models under one CV")
    c.add_argument("--config", type=Path, default=None, help=CONFIG_HELP)
    c.add_argument("--dev-only", action="store_true", help=DEV_ONLY_HELP)

    fw = sub.add_parser("forward", help="forecast later visits from earlier ones (time-aware)")
    fw.add_argument("--config", type=Path, default=None, help=CONFIG_HELP)
    fw.add_argument("--ks", type=int, nargs="*", default=[2, 3, 4, 5, 6])
    fw.add_argument("--dev-only", action="store_true", help=DEV_ONLY_HELP)

    sg = sub.add_parser("segments", help="score cohort-splitting schemes against one shared model")
    sg.add_argument("--config", type=Path, default=None, help=CONFIG_HELP)
    sg.add_argument("--dev-only", action="store_true", help=DEV_ONLY_HELP)
    sg.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="CV seeds to repeat the comparison under; more than one triples the runtime "
        "and is what tells a real gain from a lucky fold assignment",
    )
    sg.add_argument("--n-boot", type=int, default=2000, help="bootstrap draws for the paired CI")

    dg = sub.add_parser("diagnose", help="write calibration submissions with known scores")
    dg.add_argument("--run", type=Path, default=None,
                    help="run directory whose submission d3 should reorder; default: the newest")

    lg = sub.add_parser("ledger", help="what was uploaded and what the leaderboard said")
    lg.add_argument("--run", type=Path, default=None, help="run directory to record; omit to print")
    lg.add_argument("--public", type=float, default=None, help="public board score")
    lg.add_argument("--private", type=float, default=None, help="private board score")
    lg.add_argument("--note", default=None, help="free text, e.g. what changed in this attempt")

    ho = sub.add_parser(
        "holdout",
        help="score a configuration on the frozen holdout — the one look selection cannot take",
        description=(
            "Fit on the dev patients, score the frozen 15% holdout, append the result to "
            "reports/holdout/ledger.csv and save the predictions. This is the only command "
            "that reads holdout targets, and it is meant to be run once per configuration "
            "(docs/evaluation-protocol.md)."
        ),
    )
    ho.add_argument("--config", type=Path, default=None, help=CONFIG_HELP)
    ho.add_argument("--tag", default=None, help="override the tag used for the output filename")
    ho.add_argument("--note", default=None, help="free text: why this look is justified")
    ho.add_argument(
        "--allow-relook",
        action="store_true",
        help="permit a second look at a configuration already in the ledger (say why in --note)",
    )

    args = p.parse_args(argv)

    # `--config current` becomes a real path before any subcommand sees it, so the run
    # directory and the holdout ledger record the file that was run, not the alias.
    if hasattr(args, "config"):
        try:
            args.config = resolve_config_path(args.config)
        except FileNotFoundError as exc:
            p.error(str(exc))

    if args.command == "diagnose":
        run_diagnose(args.run)
        return 0

    if args.command == "ledger":
        return run_ledger(args)

    if args.command == "holdout":
        cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()
        if args.tag:
            cfg.tag = args.tag
        run_holdout(
            cfg,
            note=args.note or "",
            allow_relook=args.allow_relook,
            config_path=args.config,
        )
        return 0

    if args.command == "forward":
        cfg = _config_from_args(args)
        run_forward(cfg, args.ks)
        return 0

    if args.command == "segments":
        cfg = _config_from_args(args)
        run_segments(cfg, SEGMENT_SCHEMES, seeds=args.seeds, n_boot=args.n_boot)
        return 0

    if args.command == "train":
        cfg = _config_from_args(args)
        if args.tag:
            cfg.tag = args.tag
        if args.model:
            cfg.model = args.model
        if args.features is not None:
            cfg.features = args.features
        if args.no_explain:
            cfg.explain = False
        run_train(cfg, run_dir=args.out)
        return 0

    if args.command == "compare":
        cfg = _config_from_args(args)
        # A sweep, not a final fit: the ensemble alone costs members x inner folds x outer
        # folds model fits, so it runs a reduced forest. Compare configurations on the
        # same budget; the real number comes from `train`.
        fast = {"n_estimators": 250, "learning_rate": 0.07}
        gbm = {"name": "gbm", "params": fast}
        default_specs = {
            "hierarchical": "hierarchical",
            "ridge": "ridge",
            "gbm": gbm,
            "ensemble": {
                "name": "ensemble", "method": "ridge_stack",
                "members": ["hierarchical", "ridge", gbm],
            },
            # A family that loses is still worth showing next to the families that win —
            # `compare` reports the number either way. `segments` picks the cut.
            "segmented": {"name": "segmented", "by": ["has_on", "has_off"], "base": gbm},
        }
        run_compare(cfg, default_specs)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
