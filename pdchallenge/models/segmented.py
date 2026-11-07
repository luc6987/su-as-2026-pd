"""One model per regime, for the visits that do not behave like the rest.

Three kinds of visit live in this data and they are not interchangeable. Measured on the
training set:

===============  =======  ============  ============  ==================
segment          n        mean target   mean LEDD     time since dx (y)
===============  =======  ============  ============  ==================
``off`` only     16,659   21.2          100% missing  2.5
``on`` only      23,407   44.3          638           7.2
both             15,537   44.2          638           7.3
===============  =======  ============  ============  ==================

An ``off``-only visit is an **untreated, early** visit: there is no ON score because
there is no drug yet, ``ledd`` is missing for every single one of them, and the mean
target is 23 points below the rest. An ``on``-only visit carries the fast
pharmacokinetic term but no clean baseline. Only a visit with both carries the full
information. Fitting one model across all three asks it to learn a response surface
that changes regime halfway.

**Whether that actually helps is an empirical question, not an assumption.** A boosted
tree can already split on ``has_on`` — and ``has_on``, ``has_off``, ``ledd_missing``,
``visit_idx`` and ``n_visits`` are all already in the design matrix — so segmentation
only pays if the regimes differ enough that dedicated trees beat shared ones. It was
measured rather than assumed: ``python -m pdchallenge.cli segments --seeds 0 1 2`` runs
the comparison, and ``docs/techniques.md`` §4.1 records the answer. In short, the
observation-type cut scored 4.2622 against 4.2781 — a gain that reproduced under three CV
seeds but whose paired patient-level confidence interval straddles zero, so this family is
**available and measured but not the default**. Cutting finer made things clearly worse.

Two properties are deliberately not optional:

* **A global fallback model is always fitted**, and every row is routed to it when its
  segment has no dedicated model — either because the segment was too small to fit at
  training time, or because it was never seen at all. Predictions are therefore total:
  no row is ever dropped, and no unseen category can crash a prediction.
* **Segment keys come from row-level observables only** (``has_on``, ``has_off``,
  ``visit_idx``, ``n_visits``), never from the target. Test visits carry their own
  ``on``/``off``, so the same key is computable there. This is the same rule the
  de-biasing follows: a per-patient quantity that is not identifiable on a test patient
  must not be invented for them.

Key columns must be numeric for a design matrix to be a valid input: ``build_matrix``
one-hots ``gene`` and ``cohort`` into dummy columns and drops the originals, so those two
cannot be used as keys by anything that predicts from a matrix.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..evaluation.metrics import clip_to_valid
from .base import BaseModel
from .registry import build, register, spec_name

# A segment needs both rows and *patients*: visits within a patient are near-collinear,
# so 500 rows drawn from 5 patients is 5 observations wearing a disguise. The patient
# floor is the one that matters; the row floor is a cheap first filter.
DEFAULT_MIN_ROWS = 500
DEFAULT_MIN_PATIENTS = 50

# One label for every kind of absent value. Left to ``astype(str)`` a NaN would be "nan"
# and a None "None", and either could collide with a genuine category.
MISSING_LABEL = "missing"


def _as_label(s: pd.Series) -> pd.Series:
    """Render a key column as stable strings.

    The subtlety is that the *same* column reaches this code as ``int`` (in the pipeline
    frame, where ``has_on`` is 0/1) and as ``float`` (in a design matrix, where
    :func:`~pdchallenge.features.matrix.build_matrix` casts everything to float). A naive
    ``astype(str)`` gives ``"1"`` and ``"1.0"`` for those two, so every row would miss its
    segment and fall back to the global model — a silent, plausible-looking degradation.
    Integral floats are therefore rendered as integers.
    """
    if pd.api.types.is_float_dtype(s):
        finite = s.dropna()
        if len(finite) and bool((finite == finite.round()).all()):
            return s.map(lambda v: MISSING_LABEL if pd.isna(v) else str(int(v)))
    return s.map(lambda v: MISSING_LABEL if pd.isna(v) else str(v))


@register
class SegmentedModel(BaseModel):
    """Route each visit to a model fitted on its own segment.

    ``base`` is any model spec — a string, a dict, or a built model — so segments can
    themselves hold ensembles::

        {"name": "segmented", "by": ["has_on", "has_off"],
         "base": {"name": "gbm", "params": {"n_estimators": 600}}}
    """

    name = "segmented"

    def __init__(
        self,
        base: Any,
        by: str | list[str],
        edges: dict[str, list[float]] | None = None,
        min_rows: int = DEFAULT_MIN_ROWS,
        min_patients: int = DEFAULT_MIN_PATIENTS,
        clip: bool = True,
    ) -> None:
        if isinstance(by, str):
            by = [by]
        if not by:
            raise ValueError("`by` must name at least one column")
        unknown = set(edges or {}) - set(by)
        if unknown:
            raise ValueError(f"`edges` given for columns not in `by`: {sorted(unknown)}")
        self.base = base
        self.by = list(by)
        # ``edges`` holds *interior* breakpoints, not pd.cut's raw ``bins`` — see `levels`.
        self.edges = dict(edges or {})
        self.min_rows = int(min_rows)
        self.min_patients = int(min_patients)
        self.clip = clip

    # -- segment keys ----------------------------------------------------------------
    def levels(self, X: pd.DataFrame) -> pd.Series:
        """The segment label of every row of ``X``.

        Missing values in a key column become their own level rather than being dropped
        or imputed: in this project missingness is usually informative (``ledd`` marks
        the untreated visits, ``gene`` marks the un-genotyped cohorts), so a row with a
        missing key is a row with a real property.

        Keys are read from whatever frame is passed, so a design matrix works as well as
        the pipeline frame — see :func:`_as_label` for why that is worth being careful
        about.
        """
        missing = [c for c in self.by if c not in X.columns]
        if missing:
            raise KeyError(
                f"segment columns absent from the frame: {missing}; "
                f"available: {sorted(map(str, X.columns))[:20]}…"
            )
        parts: list[pd.Series] = []
        for col in self.by:
            s = X[col]
            if col in self.edges:
                # The outer bins are closed off with ±inf so no row can fall outside every
                # bin: pd.cut's raw `bins` would return NaN there, which would then read as
                # a missing category rather than as an out-of-range value.
                s = pd.cut(s, bins=[-np.inf, *self.edges[col], np.inf]).astype(str)
            else:
                s = _as_label(s)
            parts.append(s)
        if len(parts) == 1:
            return parts[0]
        return parts[0].str.cat(parts[1:], sep="|")

    # -- fitting ---------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        yv = np.asarray(y, dtype=float)
        g = np.asarray(groups) if groups is not None else None
        keys = self.levels(X).to_numpy()

        # The fallback is fitted even when every segment qualifies: it is what answers for
        # a level that shows up at predict time and was never fitted here.
        self.fallback_ = build(self.base).fit(X, yv, groups=g)

        self.models_: dict[str, BaseModel] = {}
        self.skipped_: dict[str, str] = {}
        self.counts_: dict[str, int] = {}

        for level in pd.unique(keys):
            level = str(level)
            mask = keys == level
            n_rows = int(mask.sum())
            n_pat = int(len(np.unique(g[mask]))) if g is not None else n_rows
            self.counts_[level] = n_rows

            if n_rows < self.min_rows:
                self.skipped_[level] = f"{n_rows} rows < min_rows {self.min_rows}"
                continue
            if g is not None and n_pat < self.min_patients:
                self.skipped_[level] = f"{n_pat} patients < min_patients {self.min_patients}"
                continue

            self.models_[level] = build(self.base).fit(
                X.iloc[mask], yv[mask], groups=g[mask] if g is not None else None
            )

        self.levels_ = [str(v) for v in pd.unique(keys)]
        return self

    # -- prediction ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = self.levels(X).to_numpy()
        # Every row starts on the fallback and is then overwritten if its level has a
        # dedicated model. Predicting the fallback first costs a model call on rows that
        # will not use it, which is cheap next to the guarantee it buys: no row can be
        # left as NaN by a level that was skipped for size or never seen at all.
        pred = np.asarray(self.fallback_.predict(X), dtype=float)

        for level, model in self.models_.items():
            mask = keys == level
            if mask.any():
                pred[mask] = model.predict(X.iloc[mask])

        return clip_to_valid(pred) if self.clip else pred

    def segment_model(self, level: str) -> BaseModel | None:
        """The model behind a level, or ``None`` if it fell back to the global one."""
        return self.models_.get(str(level))

    def get_params(self) -> dict[str, Any]:
        return {
            "base": spec_name(self.base),
            "by": list(self.by),
            "edges": {k: list(v) for k, v in self.edges.items()},
            "min_rows": self.min_rows,
            "min_patients": self.min_patients,
            # The two numbers a reader needs to judge the fit: which segments got their
            # own model, and which fell back (and therefore repeat the global model).
            "fitted": {lvl: self.counts_[lvl] for lvl in self.models_},
            "fallback": {lvl: self.skipped_[lvl] for lvl in self.skipped_},
        }
