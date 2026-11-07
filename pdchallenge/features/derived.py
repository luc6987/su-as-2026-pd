"""Per-visit derived columns, wrapped as a transformer.

The computation itself lives in :func:`pdchallenge.data.loading.add_derived` — it is a
pure function on frames and ``load_data`` needs it too, so it is defined once there and
adapted here rather than duplicated.
"""
from __future__ import annotations

from typing import Any, Self

import pandas as pd

from ..data.loading import add_derived
from .base import Transformer


class DerivedFeatures(Transformer):
    """``time_since_diagnosis``, ``visit_idx``, ``n_visits``, ``has_on``/``has_off``, …

    Parameters reach it as **flat kwargs** — ``features: [{name: derived,
    repair_imputed_age: false}]`` — not the nested ``params:`` block the *model* registry
    uses (``members: [{name: gbm, params: {...}}]``). Easy to get wrong in the other
    direction, so: whatever the step's ``__init__`` names is what goes in the mapping.
    """

    name = "derived"

    # Measured (tests/test_properties.py): `fit` ignores y, but `n_visits` counts the
    # patient's whole trajectory, so a row's values move when later visits are dropped —
    # the forecast path rebuilds these columns from the known visits for exactly that reason.
    reads_target = False
    causal = False

    def __init__(self, repair_imputed_age: bool = True) -> None:
        # An A/B control, and the only step in this chain whose effect the CV cannot show:
        # the rows it targets are already NaN in X_train, so it can only ever fire on
        # X_test, and `train` prints the same 4.2781 either way. To measure it, *simulate*
        # the fill — overwrite a holdout's missing ages with the organiser's constant and
        # score it both ways (docs/audit-2026-09-16.md §2 P0-2: +0.008 ± 0.033, a wash).
        # Set this False to reproduce the pre-2026-09-16 submission and A/B the leaderboard.
        self.repair_imputed_age = repair_imputed_age

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.require(df, ["age", "age_at_diagnosis", "on", "off", "patient_id"])
        # add_derived sorts by patient/age internally then restores file order, so the
        # result is row-aligned with the input; preserve_index asserts that.
        return self.preserve_index(
            add_derived(df, repair_imputed_age=self.repair_imputed_age), df
        )

    def get_params(self) -> dict[str, Any]:
        # The switch itself, not a count of what it did: by the time the run summary dumps
        # this, the step has been transformed on X_train as well, where the count is 0 and
        # would read as "nothing happened". `run_train` reports the test-side count instead.
        return {"repair_imputed_age": self.repair_imputed_age}
