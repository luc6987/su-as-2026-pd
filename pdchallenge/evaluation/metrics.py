"""Error metrics.

**MSE is the metric.** The challenge leaderboard reports mean squared error on the 0-132
MDS-UPDRS III scale, not its square root: a submission of the constant 37.3316 scored
278.83, which only reads correctly as MSE (the training target's variance is 272.39) and
cannot be an RMSE on a bounded scale. So MSE is the headline everywhere here, and a
leaderboard number is directly comparable to a number this module produces.

Three consequences, each of them easy to get wrong:

* **MSE is in squared points.** 18.30 MSE is not "18 points off"; it is sqrt(18.30) = 4.28
  points off. Statements about clinical significance -- the 3.25-4.63 MCID -- are
  point-scale statements and are labelled as such wherever they appear. They are never
  quoted in MSE.
* **Deltas are ~8.6x larger than point-scale deltas** at this error level, since
  ``d(MSE) = 2*sqrt(MSE)*d(sqrt(MSE))``. A gap of 0.14 MSE is 0.016 points, which is why a
  difference must be judged against the MCID in points and not against an MSE threshold.
* **MSE is additive** over error components, so ``decompose_mse``'s between/within split
  sums back to the total exactly. That is what makes the level/shape decomposition legible.

Every report also carries **bias** (are we systematically high or low?) and **coverage**
(how many visits the number was computed on). A metric quoted without coverage is not
comparable: the raw ``off`` score is a very good predictor of the ~58% of visits that have
one, and simply undefined for the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ok = ~np.isnan(a) & ~np.isnan(b)
    return a[ok], b[ok]


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    a, b = _pair(y_true, y_pred)
    if len(a) == 0:
        return float("nan")
    return float(np.mean((a - b) ** 2))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error, in **points** -- a different unit from ``mse``, kept because it
    is the robust read on the typical miss that MSE's tail can hide."""
    a, b = _pair(y_true, y_pred)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed error, ``pred − true``, in **points**. Positive means we over-predict."""
    a, b = _pair(y_true, y_pred)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(b - a))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    a, b = _pair(y_true, y_pred)
    if len(a) == 0:
        return float("nan")
    denom = float(np.sum((a - a.mean()) ** 2))
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / denom)


def summarize(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "n": int(len(_pair(y_true, y_pred)[0])),
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def segment_metrics(
    df: pd.DataFrame,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    by: list[str],
) -> pd.DataFrame:
    """MSE/bias/count for each level of each ``by`` column.

    This is where the interesting failures show up: error is never uniform across
    ``visit_idx`` or across visits that do/don't carry an ON score.
    """
    frame = df[by].copy()
    frame["y_true"] = np.asarray(y_true, dtype=float)
    frame["y_pred"] = np.asarray(y_pred, dtype=float)
    frame["err"] = frame["y_pred"] - frame["y_true"]

    rows = []
    for col in by:
        for level, grp in frame.groupby(col, dropna=False):
            ok = grp["err"].notna()
            e = grp.loc[ok, "err"]
            rows.append(
                {
                    "segment": col,
                    "level": str(level),
                    "n": int(ok.sum()),
                    "mse": float((e**2).mean()) if ok.any() else float("nan"),
                    "bias": float(e.mean()) if ok.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# The MDS-UPDRS Part III total is a sum of 33 items scored 0-4. The bounds are properties of
# the instrument, not settings: they are the same on every run and in every cohort, so they
# are constants rather than config. `clip_to_valid` still takes them as arguments, because a
# test needs to exercise a different range without touching the real one.
SCORE_MIN = 0.0
SCORE_MAX = 132.0


def clip_to_valid(y_pred: ArrayLike, lo: float = SCORE_MIN, hi: float = SCORE_MAX) -> np.ndarray:
    """The MDS-UPDRS III total is bounded; unclipped predictions waste error budget."""
    return np.clip(np.asarray(y_pred, dtype=float), lo, hi)


def paired_patient_bootstrap(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    patient: ArrayLike,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Is ``mse(b) − mse(a)`` distinguishable from zero?

    Answering "model B scored 0.14 MSE better, is that real?" needs a paired comparison, and
    the resampling unit has to be the **patient**, not the row. A patient's visits lie on
    one near-linear curve, so they are a single piece of evidence wearing several hats; a
    row-level bootstrap would treat them as independent and report an interval far
    narrower than the data can support. That mistake is what makes a sub-1% difference look
    significant when it is not.

    ``a`` is the reference model. Returns the point estimate of the difference, its 95%
    interval, and the share of draws in which ``b`` came out ahead — which is the number to
    quote, because the interval is usually the less legible of the two.

    The interval is in MSE units, but the *decision* it supports is unit-free: the square
    root is monotone, so it changes no draw's sign. ``p_b_better`` and whether the interval
    straddles zero are the same whichever scale you read them on; only the magnitudes need
    converting (divide an MSE delta by ``2*sqrt(mse)`` for the point-scale equivalent).
    """
    a = np.asarray(y_true, dtype=float).ravel()
    pa = np.asarray(pred_a, dtype=float).ravel()
    pb = np.asarray(pred_b, dtype=float).ravel()
    groups = np.asarray(patient).ravel()

    ok = ~np.isnan(a) & ~np.isnan(pa) & ~np.isnan(pb)
    a, pa, pb, groups = a[ok], pa[ok], pb[ok], groups[ok]

    # Per-patient sums, so a resample is a lookup rather than a re-scan: the MSE of a draw
    # is the sum of drawn squared errors over the sum of drawn rows.
    positions = pd.Series(np.arange(len(a))).groupby(pd.Series(groups).to_numpy()).indices
    idx = list(positions.values())
    n_p = np.array([len(ix) for ix in idx], dtype=float)
    sse_a = np.array([float(((a[ix] - pa[ix]) ** 2).sum()) for ix in idx])
    sse_b = np.array([float(((a[ix] - pb[ix]) ** 2).sum()) for ix in idx])

    n_patients = len(idx)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    # Drawn in blocks: a full (n_boot, n_patients) index matrix is ~100 MB on this cohort.
    block = max(1, min(n_boot, 1_000_000 // max(n_patients, 1)))
    for start in range(0, n_boot, block):
        stop = min(start + block, n_boot)
        draw = rng.integers(0, n_patients, size=(stop - start, n_patients))
        rows = n_p[draw].sum(axis=1)
        boot[start:stop] = sse_b[draw].sum(axis=1) / rows - sse_a[draw].sum(axis=1) / rows

    point = float(sse_b.sum() / n_p.sum() - sse_a.sum() / n_p.sum())
    return {
        "delta_mse": point,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_b_better": float((boot < 0).mean()),
        "n_boot": int(n_boot),
        "n_patients": int(n_patients),
    }


def decompose_mse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    patient: ArrayLike,
) -> dict[str, float]:
    """Split the error into the part a *level* fix could remove and the part it cannot.

    ``between`` is the MSE of each patient's mean signed error — the error in getting that
    patient's overall level right, which is what the unobservable per-patient drug-response
    offset drives. ``within`` is what is left once you allow a perfect per-patient constant
    correction, i.e. the error in the *shape* of the trajectory.

    Under MSE the split is exact: ``mse == between_patient + within_patient``, because each
    patient's within-patient deviations sum to zero and the cross term vanishes. (Under a
    square root it would not add up, which is one reason the decomposition is read here and
    not on the RMSE scale.) A change that moves ``between`` but not ``within`` is a
    level/de-biasing change; one that moves ``within`` is a trajectory-shape change.
    """
    a, b = _pair(y_true, y_pred)
    groups = np.asarray(patient)[~np.isnan(np.asarray(y_true, dtype=float).ravel())]
    err = b - a
    per_patient = pd.Series(err).groupby(pd.Series(groups).to_numpy()).transform("mean").to_numpy()
    return {
        "mse": float(np.mean(err**2)),
        "within_patient": float(np.mean((err - per_patient) ** 2)),
        "between_patient": float(np.mean(per_patient**2)),
    }
