"""Grouped cross-validation, and the frozen evaluation holdout.

**The single most important rule in this project**: test patients are disjoint from
training patients, so every split must be made on ``patient_id``. Splitting by visit
would put the same patient on both sides of the split, and because a patient's visits
lie on one near-linear curve that leaks almost the whole answer — cross-validation
scores would look far better than the leaderboard.

The folds here are a seeded permutation of *patients* (not visits), so they are
reproducible and roughly balanced in size.

The holdout is the other half of the same idea, one level up. Folds answer "how well does
this generalise to unseen patients?" — but every configuration was still *chosen* by
looking at those same folds, and re-running the CV under a second seed does not help,
because the seeds share the data and only re-deal the folds. Cutting out a fixed 15% of the
patients once, and making every selection on the rest, is what makes the final number
independent of the choices that led to it. See ``docs/evaluation-protocol.md``.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from .schema import PATIENT

# The evaluation holdout, frozen 2026-09-24. These two numbers plus the set of training
# patients determine `reports/holdout/holdout_patients.csv`, and the file is checked in and
# not to be regenerated: moving the holdout after selection has begun would invalidate every
# dev number that has already been reported against it.
HOLDOUT_FRAC = 0.15
HOLDOUT_SEED = 20260924

# The confirmation set, frozen 2026-10-04, inside the *dev* patients. Two thirds of dev (the
# mining set) is where error anatomy, arm sweeps and structure-finding may look; the remaining
# third (the confirmation set) is only ever *reported* on, so a structure found on the mining
# patients is checked against rows the search never touched. The v5 round is the motivation:
# its dev effect (−0.3393) largely failed to reproduce on the holdout (−0.0741), and the
# likeliest reason is exactly this — discovered and measured on the same patients, with ~10
# arms tried and 2 kept. See `docs/evaluation-protocol.md` §7 and the file
# `reports/protocol/confirmation_patients.csv`.
CONFIRM_FRAC = 1 / 3
CONFIRM_SEED = 20261004


def group_kfold(
    groups: np.ndarray | pd.Series,
    n_splits: int = 5,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_positions, valid_positions)`` as positional indices.

    ``groups`` is the per-row patient id. Fold membership is decided per patient, so no
    patient ever appears in both parts. Patients are shuffled with ``seed`` before being
    dealt into folds.

    Raises ``ValueError`` if there are fewer distinct patients than folds.
    """
    groups = np.asarray(groups)
    if len(groups) == 0:
        raise ValueError("groups is empty")
    patients = np.unique(groups)
    if len(patients) < n_splits:
        raise ValueError(f"need >= {n_splits} distinct patients, got {len(patients)}")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(patients)
    folds = np.array_split(shuffled, n_splits)

    for fold in folds:
        valid_mask = np.isin(groups, fold)
        yield np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)


def assert_disjoint(groups: np.ndarray | pd.Series, train_idx: np.ndarray, valid_idx: np.ndarray) -> None:
    """Guard against the leakage described in the module docstring."""
    groups = np.asarray(groups)
    overlap = set(groups[train_idx]) & set(groups[valid_idx])
    if overlap:
        raise AssertionError(
            f"patient leakage between folds: {len(overlap)} shared patients, e.g. {sorted(overlap)[:3]}"
        )


def holdout_split(
    patients: np.ndarray | pd.Series,
    frac: float = HOLDOUT_FRAC,
    seed: int = HOLDOUT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """``(dev_ids, holdout_ids)`` — a seeded, sorted, frozen cut of the patients.

    A pure function of ``(the patient set, frac, seed)``: no file is read and nothing is
    written, so the caller can compare its output against the checked-in list.

    Both sides are sorted, because the file that freezes this split is a sorted list and a
    comparison that depended on draw order would be a comparison of two different things.
    """
    patients = np.asarray(patients)
    uniq = np.unique(patients)
    if len(uniq) == 0:
        raise ValueError("patients is empty")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(uniq)
    n_holdout = int(round(frac * len(uniq)))
    return np.sort(shuffled[n_holdout:]), np.sort(shuffled[:n_holdout])


def load_holdout_patients(path: Path | str | None = None) -> np.ndarray:
    """The frozen holdout patient list, from ``reports/holdout/holdout_patients.csv``.

    ``path`` is for tests and for re-reading an archived copy; the default is the checked-in
    file. Ids are read as **strings** rather than letting pandas infer a dtype — the column
    is a list of identifiers, and an id that happened to look numeric would otherwise come
    back as a number and never match the frame it is used to index.
    """
    path = Path(path) if path is not None else paths.HOLDOUT_PATIENTS_CSV
    frame = pd.read_csv(path, dtype={PATIENT: str})
    if PATIENT not in frame.columns:
        raise ValueError(f"{path} has no {PATIENT!r} column (found {list(frame.columns)})")
    return frame[PATIENT].to_numpy()


def dev_mask(groups: np.ndarray | pd.Series, holdout_ids: np.ndarray | pd.Series) -> np.ndarray:
    """Boolean row mask: ``True`` where the row belongs to a **dev** (non-holdout) patient."""
    return ~np.isin(np.asarray(groups), np.asarray(holdout_ids))


def confirmation_split(
    dev_patient_ids: np.ndarray | pd.Series,
    frac: float = CONFIRM_FRAC,
    seed: int = CONFIRM_SEED,
) -> np.ndarray:
    """The confirmation patients — a seeded, sorted, frozen cut of the **dev** patients.

    Same construction as :func:`holdout_split`, one level down: shuffle ``np.unique(patients)``
    with ``seed`` and keep the first ``round(frac · n)`` ids. A pure function of the patient
    set, ``frac`` and ``seed`` — no file is read and nothing is written, so the caller can
    compare its output against the checked-in list. The returned side is sorted, because the
    file that freezes it is a sorted list.

    Pass **dev** ids: the confirmation set is defined inside dev, and the evaluation holdout
    is not part of it.
    """
    patients = np.asarray(dev_patient_ids)
    uniq = np.unique(patients)
    if len(uniq) == 0:
        raise ValueError("dev_patient_ids is empty")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(uniq)
    n_confirm = int(round(frac * len(uniq)))
    return np.sort(shuffled[:n_confirm])


def load_confirmation_patients(path: Path | str | None = None) -> np.ndarray:
    """The frozen confirmation list, from ``reports/protocol/confirmation_patients.csv``.

    Ids are read as **strings** for the same reason as :func:`load_holdout_patients`: a
    numeric-looking patient id must not come back as a number and silently match nothing.
    """
    path = Path(path) if path is not None else paths.CONFIRMATION_PATIENTS_CSV
    frame = pd.read_csv(path, dtype={PATIENT: str})
    if PATIENT not in frame.columns:
        raise ValueError(f"{path} has no {PATIENT!r} column (found {list(frame.columns)})")
    return frame[PATIENT].to_numpy()


def mining_mask(groups: np.ndarray | pd.Series, confirm_ids: np.ndarray | pd.Series) -> np.ndarray:
    """Boolean row mask: ``True`` where the row belongs to a **mining** (non-confirmation) patient.

    The complement of the confirmation set *within dev*: on dev rows, this mask and the
    confirmation set partition dev exactly. It does not know about the evaluation holdout —
    a caller starting from all training rows must apply :func:`dev_mask` first, or the
    holdout patients would land in the mining set.
    """
    return ~np.isin(np.asarray(groups), np.asarray(confirm_ids))


def patient_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient, for patient-level analysis and fold bookkeeping."""
    return (
        df.groupby(PATIENT, as_index=False)
        .agg(n_visits=(PATIENT, "size"), age_first=("age", "min"), age_last=("age", "max"))
    )
