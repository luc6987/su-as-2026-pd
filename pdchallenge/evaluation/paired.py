"""Paired, patient-level comparison of two models — one seed, or several pooled.

The project's decision rule compares a candidate against an incumbent on the *same rows
under the same fold assignment*, and resamples **patients**, never rows: a patient's visits
lie on one near-linear curve, so they are a single piece of evidence wearing several hats.
The two bootstrap functions here are the measured protocol, lifted out of the two scripts
that produced the 2026-09-24 numbers (`reports/consolidate-2026-09-24/analyse.py`,
`reports/integrate-v2/compare.py`) so that later work cannot drift from them: they are
numerically identical to those implementations, and `tests/test_evaluation_paired.py` pins
that by loading the old files and comparing outputs bit for bit.

`projected_holdout_se` and `pass_probability` were added later (2026-10-04, §8 of
`docs/evaluation-protocol.md`): one projects a Δ's standard error from the confirmation set
to the holdout's 1,046 patients, the other turns that into the probability a holdout look
would pass — together the tool for deciding *whether* a look is worth spending.
`calibrated_pass_probability` (2026-10-11, §9) is the same reading with the projected SE
inflated by the factor the two looks since then have actually shown.

Two things this module is *not*. It is not `metrics.paired_patient_bootstrap`, which answers
the same question but groups with pandas and drops NaN rows — a slightly different estimator
that no published number here was computed with. And it is not a significance test for a
*small* effect: one CV seed's patient bootstrap holds the fold assignment and the fitted
model fixed, so it cannot see the variance that comes from the model itself being refit
under a different booster seed. Only swapping the booster on a second arm reveals that, and
`docs/review-2026-09-26.md` §2.1 measured it at −0.0593 [−0.1124, −0.0093] for a change that
does nothing at all. `pooled_paired_bootstrap` plus the noise floor in
`reports/protocol/noise_calibration.csv` is the answer to that, not a wider single-seed
interval.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

# The two boundaries of §2 of `docs/prompts-2026-09-26/00-RULES.md`: an effect below this
# size cannot be resolved at single-booster precision, so it must be re-measured with the
# five-seed bag; and the pooled interval's upper bound must be strictly below zero.
SMALL_EFFECT: float = 0.10


def _per_patient_sse(
    y: np.ndarray, preds: Sequence[np.ndarray], groups: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(rows per patient, [SSE per patient per arm])`` — the only per-draw lookup tables.

    ``pd.factorize`` over the raw group labels (not a sorted unique) is deliberate: the
    legacy scripts use it, and the patient order decides the columns of every bootstrap
    draw, so any other ordering would change the numbers.
    """
    codes, _ = pd.factorize(groups)
    n_p = np.bincount(codes).astype(float)
    return n_p, [np.bincount(codes, weights=(p - y) ** 2) for p in preds]


def _check_finite(y: np.ndarray, groups: np.ndarray, *preds: np.ndarray) -> None:
    """Refuse NaN instead of letting it eat the denominator.

    A NaN in an out-of-fold vector means some row was never predicted — a fold whose model
    failed, or an OOF vector read from the wrong population. ``pd.bincount`` would then
    carry a NaN into that patient's SSE and every draw containing it, so the delta would be
    NaN while the *rows-per-patient* count stayed right: a silent wrong answer. The legacy
    scripts have no such guard; on finite input this changes nothing.
    """
    for name, a in [("y", y), *[(f"pred[{i}]", p) for i, p in enumerate(preds)]]:
        if a.size and not np.isfinite(np.asarray(a, dtype=float)).all():
            raise ValueError(f"{name} contains NaN/inf — the two arms are not scored on the same rows")
    # Labels, not numbers: `factorize` would silently turn a missing patient id into a
    # patient of its own, which is the opposite of what a missing id means.
    if groups.size and bool(pd.isna(groups).any()):
        raise ValueError("groups contains missing patient ids")


def paired_bootstrap(
    y: np.ndarray,
    pred_ref: np.ndarray,
    pred_new: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """``MSE(pred_new) − MSE(pred_ref)`` with a patient-level paired bootstrap interval.

    Both arms must have predicted the **same rows under the same fold assignment**;
    otherwise the difference is not paired and the interval is meaningless. Negative means
    the candidate is better. ``seed`` seeds the resampler (the protocol uses the CV seed, so
    a rerun of the same comparison reproduces to the last bit).

    The draw is one patient multiset, and the MSE of that draw is the drawn patients'
    summed squared errors over their summed row count — patients with more visits carry
    proportionally more weight, exactly as they do in the score being estimated.
    """
    y = np.asarray(y, dtype=float).ravel()
    ref = np.asarray(pred_ref, dtype=float).ravel()
    new = np.asarray(pred_new, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if not (len(y) == len(ref) == len(new) == len(groups)):
        raise ValueError("y, both prediction vectors and groups must cover the same rows")
    _check_finite(y, groups, ref, new)

    n_p, (sse_r, sse_n) = _per_patient_sse(y, [ref, new], groups)
    rng = np.random.default_rng(seed)
    k = len(n_p)
    boot = np.empty(n_boot)
    # Blocked: one (n_boot, n_patients) index matrix is ~100 MB on this cohort.
    block = max(1, 1_000_000 // k)
    for start in range(0, n_boot, block):
        stop = min(start + block, n_boot)
        draw = rng.integers(0, k, size=(stop - start, k))
        boot[start:stop] = (sse_n[draw].sum(1) - sse_r[draw].sum(1)) / n_p[draw].sum(1)
    return {
        "delta_mse": float((sse_n.sum() - sse_r.sum()) / n_p.sum()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_better": float((boot < 0).mean()),
    }


def pooled_paired_bootstrap(
    y: np.ndarray,
    refs: Sequence[np.ndarray],
    cands: Sequence[np.ndarray],
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The mean paired Δ over several CV seeds, with **one** patient-level interval.

    Why one resample and not three averaged intervals: the seeds score the *same rows*, just
    under different fold assignments, so resampling patients separately per seed and then
    averaging the three intervals would treat the same patient as three independent pieces of
    evidence and report a far too wide band. Here each draw resamples the patient multiset
    **once** and all seeds' deltas are averaged inside that same draw — the pairing across
    seeds is used as pairing, which is the whole point of running several seeds.

    That also makes the interval cover the *fold-assignment* noise a single seed cannot see,
    as long as the seeds are genuinely different fold assignments: with three seeds, a draw's
    quantity is the run-mean Δ of that patient multiset. What it still cannot see is the
    refit variance the *same* fold assignment would give with another booster seed — a second
    CV seed is not a second model seed. That is what §3 of the 00-RULES noise floor is for.

    ``per_seed_delta`` and ``seed_sd`` are reported alongside: the first is the per-seed
    point estimate (mean equals ``delta_mse`` by construction), the second their standard
    deviation with ``ddof=1`` — the protocol asks for a direct estimate of fold-assignment
    noise next to the interval, and NaN when only one seed was given, where it is undefined.
    """
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    refs = [np.asarray(r, dtype=float).ravel() for r in refs]
    cands = [np.asarray(c, dtype=float).ravel() for c in cands]
    if len(refs) != len(cands) or not refs:
        raise ValueError("refs and cands must be non-empty and the same length")
    if len(y) != len(groups) or any(len(a) != len(y) for a in [*refs, *cands]):
        raise ValueError("y, groups and every arm must cover the same rows")
    _check_finite(y, groups, *refs, *cands)

    n_p, sse_r = _per_patient_sse(y, refs, groups)
    _, sse_c = _per_patient_sse(y, cands, groups)
    rng = np.random.default_rng(seed)
    k = len(n_p)
    boot = np.empty(n_boot)
    block = max(1, 1_000_000 // k)
    for start in range(0, n_boot, block):
        stop = min(start + block, n_boot)
        draw = rng.integers(0, k, size=(stop - start, k))
        total = n_p[draw].sum(1)
        acc = np.zeros(stop - start)
        for r, c in zip(sse_r, sse_c):
            acc += (c[draw].sum(1) - r[draw].sum(1)) / total
        boot[start:stop] = acc / len(sse_r)

    per_seed = [(c.sum() - r.sum()) / n_p.sum() for r, c in zip(sse_r, sse_c)]
    return {
        "delta_mse": float(np.mean(per_seed)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_better": float((boot < 0).mean()),
        "n_seeds": len(refs),
        "per_seed_delta": [float(d) for d in per_seed],
        "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan"),
    }


def projected_holdout_se(
    y: np.ndarray,
    pred_ref: np.ndarray,
    pred_new: np.ndarray,
    groups: np.ndarray,
    n_target_patients: int,
) -> float:
    """The paired-Δ standard error these patients imply for a cohort of ``n_target_patients``.

    The protocol's estimator is Δ̂ = Σ_p d_p / Σ_p n_p, with ``d_p`` a patient's SSE
    difference between the two arms and ``n_p`` their visit count. Treating patients as iid
    draws of ``(d_p, n_p)``, the delta method gives
    ``SE(Δ̂) ≈ sd(d_p − Δ̂·n_p) · √N / Σ_p n_p`` — the ratio's own sampling error, with the
    per-patient visit counts carried along. Projecting to another cohort of
    ``n_target_patients`` patients of the same mix divides the sampling variance by the
    count ratio, hence the ``√(N / n_target_patients)`` factor.

    The intended use is §8 of ``docs/evaluation-protocol.md``: measure Δ on the confirmation
    set, convert its SE to the 1,046-patient holdout scale, and let :func:`pass_probability`
    say whether a holdout look is worth spending. Like the other estimators in this module
    this is the sampling error *of the estimate* — the fitted models stay fixed — so it
    excludes refit variance; the bag5 noise floor in ``reports/protocol/noise_calibration.csv``
    is the separate answer to that.
    """
    y = np.asarray(y, dtype=float).ravel()
    ref = np.asarray(pred_ref, dtype=float).ravel()
    new = np.asarray(pred_new, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if not (len(y) == len(ref) == len(new) == len(groups)):
        raise ValueError("y, both prediction vectors and groups must cover the same rows")
    if n_target_patients <= 0:
        raise ValueError(f"n_target_patients must be positive, got {n_target_patients}")
    _check_finite(y, groups, ref, new)

    n_p, (sse_r, sse_n) = _per_patient_sse(y, [ref, new], groups)
    if len(n_p) < 2:
        raise ValueError("need at least two patients to estimate a spread across patients")
    d = sse_n - sse_r
    delta = float(d.sum() / n_p.sum())
    resid = d - delta * n_p
    se = float(np.std(resid, ddof=1) * np.sqrt(len(n_p)) / n_p.sum())
    return se * float(np.sqrt(len(n_p) / n_target_patients))


def pass_probability(delta: float, se: float, z: float = 1.959964) -> float:
    """P(the 95% upper bound lands below zero), given a true effect ``delta`` and SE ``se``.

    The measured paired Δ̂ is treated as ``N(delta, se²)``; a holdout rule of the form "the
    upper bound of the paired interval is below 0" passes when ``Δ̂ + z·se < 0``. That event
    has probability ``Φ(−delta/se − z)``, computed with ``math.erf`` — no new dependency.

    ``delta`` is negative when the candidate is better, as everywhere else in this module:
    ``pass_probability(-0.34, 0.12) ≈ 0.81`` — a real −0.34 effect clears a 0.12-SE holdout
    four times in five. §8 of ``docs/evaluation-protocol.md`` uses 0.5 as the floor: below
    it, a look is more likely to consume the holdout's independence than to pass.
    """
    if se <= 0:
        raise ValueError(f"se must be positive, got {se}")
    return 0.5 * (1.0 + math.erf((-delta / se - z) / math.sqrt(2.0)))


# §9 of `docs/evaluation-protocol.md` (2026-10-11): on the two holdout looks that had a §8
# projection written down first, the realised paired-Δ SE came back 1.96× (v7 − v6) and 1.86×
# (v8 − v7) the projection — `reports/round-2026-10-11/cal/power_ratio.txt`. Until a dev-only
# pseudo-holdout measures the factor, the projection is inflated by this before it is read.
HOLDOUT_SE_CALIBRATION: float = 1.9


def calibrated_pass_probability(
    delta: float, se_projected: float, k: float = HOLDOUT_SE_CALIBRATION, z: float = 1.959964
) -> float:
    """§9's pass probability: :func:`pass_probability` on the §8 projected SE multiplied by ``k``.

    :func:`projected_holdout_se` sees one source of noise — which patients land in the holdout —
    while both looks so far came back about twice as wide as it projected; the cause is not
    established (§9 lists three candidates, none tested). ``k`` carries the measured ratio, so a
    look is only spent when it would still pass at the width the holdout has actually shown.
    ``k = 1`` gives back §8's raw number; the protocol reports both and decides on this one.

    On the v8 look's own inputs, ``calibrated_pass_probability(-0.1243, 0.0567)`` is 0.210 where
    §8 promised 0.592 — and the look failed its rule.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return pass_probability(delta, k * se_projected, z)


def verdict(pooled: dict[str, Any], bagged: bool) -> str:
    """Turn a pooled comparison into one of the three protocol answers.

    ``"成立"`` requires all three legs of §2 of the 00-RULES: every seed's Δ below zero, and
    the pooled 95% upper bound below zero. ``"未确立"`` is everything else.

    ``"需 bag5 重测"`` is the third answer and it short-circuits: when the mean Δ is smaller
    than :data:`SMALL_EFFECT` in magnitude and the two arms were *not* five-seed bags, the
    comparison is not yet a measurement — a single booster's own refit noise is worth about
    that much (0.0593 measured, `reports/protocol/noise_calibration.csv`), so the same
    pipeline with one `random_state` moved would land in the same place. Re-run both arms
    with the bag before reading the result. ``bagged`` is the caller's declaration, and the
    comparison tool refuses to take it on trust: it checks the two model names.
    """
    per_seed = [float(d) for d in pooled["per_seed_delta"]]
    mean_delta = float(np.mean(per_seed))
    if not bagged and abs(mean_delta) < SMALL_EFFECT:
        return "需 bag5 重测"
    if all(d < 0.0 for d in per_seed) and float(pooled["ci_high"]) < 0.0:
        return "成立"
    return "未确立"
