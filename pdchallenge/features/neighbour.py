"""Adjacent-visit features: what a patient's *other* visits say about this one.

Every other feature in the chain reaches a visit either through its own observation or through
a summary of the whole patient (`patient_aggregates`, the fitted curves). Neither says *where*
inside the trajectory the visit sits: being 3 points under the patient's maximum means one
thing at the first visit and another at the last. This step hands the booster the patient's own
neighbouring visits directly — the residual columns at the ±k-th visit, the time gaps, and the
mean of the residuals over each window — with the visits sorted by `time_col`.

Nothing is fitted and no target is read: every column is a groupby shift of the patient's own
rows, so the step means the same thing on the test set as inside a training fold, exactly like
`patient_aggregates`. Measured on dev (single booster, three seeds, patient-grouped folds) it
is worth about −0.44 MSE against the incumbent chain — and it is one of the two changes here
that only hold in the *patient-transductive* framing; see below and
`docs/review-2026-09-26.md` §5.1.

Two consumers need different things from it:

* the **main CV** is transductive within a patient — a test visit carries its own scores — so a
  visit may see all of its neighbours, the ones after it included;
* the **forecast** (`evaluation/forward.py`) may not. A future visit has to be described by the
  already-known visits alone, so `forward.py` rebuilds these columns causally, one future visit
  at a time, on `known visits + this row`; the future visits stay invisible to each other. The
  columns are deliberately **not** in `PATIENT_LEVEL_COLS`: carrying a constant forward would
  be exactly the leak that rebuild exists to prevent.

The column order is part of the contract, not a cosmetic choice: `build_matrix` carries the
new columns in the order this step appends them, and `colsample_bytree` samples the design
matrix by position, so a permutation is a different model. `tests/test_features_neighbour.py`
pins the order and the values against the reference implementation the review measured with
(`reports/review-2026-09-26/nbr_feats.py`).
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT
from .base import Transformer
from .envelope import OBS_MINUS_ENV
from .likelihood import OBS_MINUS_LIK
from .pk_response import PK_OBS
from .ratio import OBS_IS_OFF

# The reference implementation's v2 setup, which is what the review measured: a ±4 window over
# the three residual/observation columns, plus the neighbours' observation *kind*.
DEFAULT_COLS = [OBS_MINUS_LIK, OBS_MINUS_ENV, PK_OBS]
DEFAULT_FLAGS = [OBS_IS_OFF]
DEFAULT_MEANS = [1, 2, 3]

# `window=None` means the whole trajectory. No patient in the cohort has more than 12 visits,
# so 11 shifts reach every neighbour any of them could have.
FULL_TRAJECTORY = 11

# Only the `obs_minus_*` residuals get a window mean — `obs_pk` is on the observation scale,
# where a mean over neighbours answers a different question. The prefix is also the test.
MEAN_PREFIX = "obs_minus_"

# The flag columns are named by the reference's spelling (`obs_is_off` -> `isoff_*`), and the
# prefix lands in the design matrix, so it is fixed here rather than derived from the name.
FLAG_PREFIX = {OBS_IS_OFF: "isoff"}

# Who writes each column this step reads, so a missing one can say *why*: a chain without
# `likelihood` has no `obs_minus_lik`, and "missing required columns" alone sends the reader
# looking in the wrong place.
_PRODUCED_BY = {
    OBS_MINUS_LIK: "likelihood",
    OBS_MINUS_ENV: "envelope_curve",
    PK_OBS: "pk_response",
    OBS_IS_OFF: "on_ratio",
}

_POS = "__nbr_pos"


class NeighbourFeatures(Transformer):
    """The patient's adjacent visits, sorted by ``time_col``. Nothing to fit.

    Parameters are the reference implementation's v2 by default (``window=4``, the three
    ``cols``, the ``obs_is_off`` flag, window means at 1/2/3); pass ``cols``/``flags``/
    ``means`` explicitly to get the ±2 variant, which is ``window=2, flags=[], means=[1, 2]``.
    """

    name = "neighbour"

    # Measured (tests/test_properties.py): nothing is fitted and no target is read; the
    # shifts reach each visit's *next* neighbours, so the step is transductive — the main
    # CV's framing, and why the forecast rebuilds these columns causally (module docstring).
    reads_target = False
    causal = False

    def __init__(
        self,
        window: int | None = 4,
        cols: list[str] | None = None,
        flags: list[str] | None = None,
        means: list[int] | None = None,
        time_col: str = "age",
    ) -> None:
        if window is not None and int(window) < 1:
            raise ValueError(f"window must be at least 1 or None, got {window!r}")
        self.window = window
        # Copied, so a caller's list can never be mutated through the instance.
        self.cols = list(DEFAULT_COLS if cols is None else cols)
        self.flags = list(DEFAULT_FLAGS if flags is None else flags)
        self.means = list(DEFAULT_MEANS if means is None else means)
        self.time_col = time_col
        # `None` is the whole trajectory; every loop below uses this count, so the two spellings
        # cannot drift apart.
        self.n_shifts = FULL_TRAJECTORY if window is None else int(window)

    # -- the columns this step writes -------------------------------------------------
    @property
    def columns(self) -> list[str]:
        """The new column names, in the order ``transform`` appends them.

        Same three blocks as :meth:`transform`: per-column shifts, then the time gaps, then the
        flag shifts, then the window means. Built here once so a caller (the forecast path, the
        tests) can ask what this step will produce without running it.
        """
        names: list[str] = []
        for col in self.cols:
            for k in range(1, self.n_shifts + 1):
                names += [f"{col}_prev{k}", f"{col}_next{k}"]
        for k in range(1, self.n_shifts + 1):
            names += [f"dt_prev{k}", f"dt_next{k}"]
        for flag in self.flags:
            prefix = FLAG_PREFIX.get(flag, flag)
            for k in range(1, self.n_shifts + 1):
                names += [f"{prefix}_prev{k}", f"{prefix}_next{k}"]
        for col in self.cols:
            if col.startswith(MEAN_PREFIX):
                names += [f"{col}_nbrmean{w}" for w in self.means]
        return names

    # -- fit / transform --------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Nothing to fit; the columns are groupby shifts of each patient's own rows."""
        self._require(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require(df)
        out = df.copy()

        work = out.copy()
        work[_POS] = np.arange(len(work))
        # Ties in `time_col` are broken by the row's original position (`mergesort` is stable,
        # and `_POS` makes it explicit), so two visits at the same time keep the caller's order
        # rather than the index's — which is what "adjacent" has to mean for a frame that is
        # not already sorted.
        work = work.sort_values([PATIENT, self.time_col, _POS], kind="mergesort")
        # `groupby(...).indices` is not used here: `shift` already works inside each group and
        # keeps the frame's order, which is what the restore below undoes.
        grp = work.groupby(PATIENT, sort=False)
        t = work[self.time_col]

        new: dict[str, pd.Series] = {}
        for col in self.cols:
            for k in range(1, self.n_shifts + 1):
                new[f"{col}_prev{k}"] = grp[col].shift(k)
                new[f"{col}_next{k}"] = grp[col].shift(-k)
        for k in range(1, self.n_shifts + 1):
            new[f"dt_prev{k}"] = t - grp[self.time_col].shift(k)
            new[f"dt_next{k}"] = grp[self.time_col].shift(-k) - t
        for flag in self.flags:
            prefix = FLAG_PREFIX.get(flag, flag)
            for k in range(1, self.n_shifts + 1):
                new[f"{prefix}_prev{k}"] = grp[flag].shift(k)
                new[f"{prefix}_next{k}"] = grp[flag].shift(-k)
        for col in self.cols:
            if not col.startswith(MEAN_PREFIX):
                continue
            for w in self.means:
                # Mean over the 2w neighbours within the window (jobs `-w..-1, 1..w`), skipping
                # the missing ones rather than the whole window: a first visit still has a mean
                # over its successors, and only a patient's single visit gets NaN.
                acc_n, acc_d = 0.0, 0.0
                for k in range(1, w + 1):
                    for shift in (k, -k):
                        s = grp[col].shift(shift)
                        acc_n = acc_n + s.fillna(0.0)
                        acc_d = acc_d + s.notna().astype(float)
                new[f"{col}_nbrmean{w}"] = acc_n / acc_d.replace(0, np.nan)

        # `sort_values` put the rows in time order; this puts them back in the caller's, by
        # position rather than by index label (labels need not be unique, and a duplicate would
        # silently align two different rows onto each other).
        order = np.argsort(work[_POS].to_numpy(), kind="stable")
        if list(new) != self.columns:
            raise AssertionError(
                "neighbour produced columns in a different order than `columns` promises — "
                "the design matrix's column order is part of the contract"
            )
        for name, series in new.items():
            out[name] = series.to_numpy(dtype=float)[order]
        return self.preserve_index(out, df)

    # -- helpers -----------------------------------------------------------------------
    def _require(self, df: pd.DataFrame) -> None:
        needed = [PATIENT, self.time_col, *self.cols, *self.flags]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            named = [
                f"{c} (written by the `{_PRODUCED_BY[c]}` step)" if c in _PRODUCED_BY else c
                for c in missing
            ]
            raise KeyError(
                f"neighbour needs columns the chain has not produced yet: {named}. "
                f"It is the last step of the chain, because it reads the residuals the "
                f"`envelope_curve` and `likelihood` steps write."
            )

    def get_params(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "n_shifts": self.n_shifts,
            "time_col": self.time_col,
            "cols": list(self.cols),
            "flags": list(self.flags),
            "means": list(self.means),
            "n_columns": len(self.columns),
        }
