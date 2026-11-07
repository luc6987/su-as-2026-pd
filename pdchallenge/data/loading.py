"""Reading the challenge CSVs and the light per-visit derivations.

Everything downstream should go through :func:`load_data` so column names, dtypes and
derived features are defined in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from .schema import (
    AGE_FIRST_VISIT,
    FILES,
    HAS_OFF,
    HAS_ON,
    INDEX,
    N_VISITS,
    PATIENT,
    TARGET,
    TIME_SINCE_DIAGNOSIS,
    VISIT_IDX,
    YEARS_IN_STUDY,
)


@dataclass
class Dataset:
    """Train/test frames plus the training target.

    ``X_test`` deliberately has no target: the challenge withholds it. Any pipeline that
    needs a target must therefore be fitted on ``X_train`` alone.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame

    @property
    def train(self) -> pd.DataFrame:
        """``X_train`` with the target attached (convenient for EDA and analysis)."""
        return self.X_train.assign(**{TARGET: self.y_train.values})

    @property
    def train_patients(self) -> np.ndarray:
        return self.X_train[PATIENT].to_numpy()

    @property
    def test_patients(self) -> np.ndarray:
        return self.X_test[PATIENT].to_numpy()


# --- the organiser's imputation of ``age_at_diagnosis`` -----------------------------
# ``X_test`` ships that column with nothing missing, while ``X_train`` is missing it for
# 5.09% of rows (353 patients). The organiser filled the gap: 1093 test rows across 146
# patients carry one full-precision float (56.29514426460239), and every genuine value in
# ``X_train`` sits on a 0.1 grid. See docs/audit-2026-09-16.md §2 P0-2.
#
# A fabricated age is worse than a missing one: it corrupts ``curve_pred`` (whose error
# against the patient's own level goes 5.0 → 14.4 on a simulated fill) and ``curve_a``,
# while ``curve_n_obs`` claims a curve the patient does not really have. Missing is the
# state 5.09% of ``X_train`` is already in, and the state the model is fitted for.
#
# Reverting it is **not** a score gain, and the audit's earlier "+0.41 MSE" does not
# replicate. Over 15 patient-grouped holdout splits the fill costs ≈+0.07 MSE on average
# (sd 1.1, and on 8 of the 15 it helps) — 0.008 points, six times smaller. The reason is
# that a constant fill *shifts* a patient's whole
# time axis rather than scrambling it, and ``curve_pred_anchored`` — the feature the model
# actually leans on — is very nearly invariant to a shift, being a difference of two equally
# shifted times (level error 7.766 filled vs 7.768 on the true axis). So the repair trades a
# corrupted curve for an absent one: on the same simulated patients the model scores 7.53
# repaired against 7.58 filled, and 2.61 where the age at diagnosis is genuinely known.
# What the filler costs those 146 patients is real but it is not recoverable from this data.
#
# That indifference is specific to the transductive CV, which is also the leaderboard's own
# framing. Where the curve genuinely has to extrapolate in ``t`` it stops being indifferent:
# on forward validation the fill costs ≈+7.7 MSE (k=3) and ≈+2.2 MSE (k=5) — 0.39 and 0.14
# points — and this repair takes it back exactly. ``repair_imputed_age=False`` restores the
# old behaviour for an A/B.
_AGE_GRID = 10.0  # the column is recorded to 0.1 years
_GRID_TOL = 1e-6
_MIN_FILL_PATIENTS = 5  # an off-grid value shared by this many patients is a fill constant


def imputed_age_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose ``age_at_diagnosis`` cannot be a real observation.

    Two rules, either of which marks a row:

    * **a repeated off-grid value.** The column is recorded to 0.1 years throughout
      ``X_train``, so a value carrying more precision than that was computed rather than
      measured. It is only called an imputation when several patients share it *exactly* —
      that is what separates a fill constant from a single genuinely finer measurement,
      and it is why this rule does not fire on a frame whose ages are merely unrounded.
    * **a diagnosis after the visit** (``age - age_at_diagnosis < 0``). Impossible, so at
      least one of the two numbers is wrong, and the time axis derived from it with them.

    The limit of the first rule is that a fill *rounded* to the recorded grid would look
    exactly like a measurement, and only a fill that also pushes a visit before its own
    diagnosis would then be caught. Nothing in the data can separate those two cases; this
    is a detector for the fill that is actually there, not a proof of cleanliness.
    """
    aad = df["age_at_diagnosis"]
    off_grid = aad.notna() & ((aad * _AGE_GRID - (aad * _AGE_GRID).round()).abs() > _GRID_TOL)
    if off_grid.any():
        # Count *patients*, not rows: the value is a patient constant, so one filled
        # patient contributing twelve visits would otherwise look like a crowd.
        shared = df.loc[off_grid].groupby(PATIENT)["age_at_diagnosis"].first()
        counts = shared.value_counts()
        off_grid &= aad.isin(counts.index[counts >= _MIN_FILL_PATIENTS])
    return off_grid | ((df["age"] - aad) < 0)


_TIME_AXIS_TOL = 1e-6  # years; the shipped and computed columns agree to 3.6e-15


def check_shipped_time_axis(df: pd.DataFrame, computed: pd.Series) -> None:
    """Refuse a shipped ``time_since_diagnosis`` that disagrees with ``age - age_at_diagnosis``.

    ``X_test`` arrives with the column already computed and the code below lets those
    values win over the recomputed ones — the project's **only** branch that exists on the
    test side alone. Cross-validation cannot reach it (``X_train`` does not ship the
    column), so a re-issued ``X_test`` — a different unit, a different reference date —
    would silently invalidate ``curve_pred``, ``curve_pred_anchored`` and
    ``curve_anchor_time`` with no CV signal whatsoever. On the current data the two agree
    to 3.6e-15, so this is a tripwire rather than a repair, and it costs one vectorised
    comparison per transform.
    """
    shipped = df[TIME_SINCE_DIAGNOSIS]
    diff = (shipped - computed).abs()
    bad = (diff > _TIME_AXIS_TOL) & ~(shipped.isna() & computed.isna())
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows disagree between the shipped and the computed "
            f"{TIME_SINCE_DIAGNOSIS} (max {float(diff.max()):.4g}); the shipped column is "
            f"not age - age_at_diagnosis and must not be trusted"
        )


def add_derived(df: pd.DataFrame, repair_imputed_age: bool = True) -> pd.DataFrame:
    """Add per-row and per-patient derived features. Uses X only — no leakage.

    - ``time_since_diagnosis``: ``age - age_at_diagnosis`` (``X_test`` ships the column
      already; ``X_train`` does not, so it is computed and the existing values win —
      after :func:`check_shipped_time_axis` has confirmed the two agree)
    - ``visit_idx`` / ``n_visits``: chronological position of the visit within the patient
    - ``age_first_visit`` / ``years_in_study``: time since the patient's first visit
    - ``has_on`` / ``has_off``: which motor score(s) this visit carries

    ``repair_imputed_age`` reverts the organiser's fill of ``age_at_diagnosis`` to NaN
    before the time axis is built; see :func:`imputed_age_mask`. It only ever fires on
    ``X_test``, so it changes deployment behaviour and no CV number.
    """
    df = df.copy()
    if repair_imputed_age:
        bad = imputed_age_mask(df)
        if bad.any():
            # Both columns, because the shipped ``time_since_diagnosis`` was computed from
            # the filled age and is as wrong as it is (max |shipped − computed| = 3.6e-15).
            # Dropping the age alone would leave the bad time axis in place.
            df.loc[bad, "age_at_diagnosis"] = np.nan
            if TIME_SINCE_DIAGNOSIS in df:
                df.loc[bad, TIME_SINCE_DIAGNOSIS] = np.nan
    tsd = df["age"] - df["age_at_diagnosis"]
    if TIME_SINCE_DIAGNOSIS in df:
        check_shipped_time_axis(df, tsd)
        df[TIME_SINCE_DIAGNOSIS] = df[TIME_SINCE_DIAGNOSIS].fillna(tsd)
    else:
        df[TIME_SINCE_DIAGNOSIS] = tsd

    df = df.sort_values([PATIENT, "age"], kind="stable")
    g = df.groupby(PATIENT, sort=False)
    df[VISIT_IDX] = g.cumcount()
    df[N_VISITS] = g["age"].transform("size")
    df[AGE_FIRST_VISIT] = g["age"].transform("min")
    df[YEARS_IN_STUDY] = df["age"] - df[AGE_FIRST_VISIT]
    df[HAS_ON] = df["on"].notna().astype(int)
    df[HAS_OFF] = df["off"].notna().astype(int)
    return df.sort_values(INDEX).reset_index(drop=True)


def load_data(data_dir: Path | str | None = None, derived: bool = True) -> Dataset:
    """Load the challenge data.

    ``data_dir`` defaults to :data:`pdchallenge.paths.DATA`.
    """
    data_dir = Path(data_dir) if data_dir is not None else paths.DATA
    X_train = pd.read_csv(data_dir / FILES["X_train"])
    y_train = pd.read_csv(data_dir / FILES["y_train"])
    X_test = pd.read_csv(data_dir / FILES["X_test"])

    assert (X_train[INDEX].values == y_train[INDEX].values).all(), "X/y misaligned"
    assert not set(X_train[PATIENT]) & set(X_test[PATIENT]), (
        "train and test patients overlap — the challenge splits by patient"
    )
    if derived:
        X_train = add_derived(X_train)
        X_test = add_derived(X_test)
    return Dataset(X_train=X_train, y_train=y_train[TARGET], X_test=X_test)


def load_random_submission(data_dir: Path | str | None = None) -> pd.DataFrame:
    """The organiser's example submission, used to confirm the expected format."""
    data_dir = Path(data_dir) if data_dir is not None else paths.DATA
    return pd.read_csv(data_dir / FILES["y_random"])
