"""Per-patient summaries of the OFF-scale observations.

The per-visit factor ``m_v`` in ``obs ≈ T · m_v`` is one-sided (right edge ≈1.1, long left
tail), so the *upper* end of a patient's observations says more about their level than the
mean does: measured with the oracle shape, the per-patient max of ``off − T`` has SD 2.4–2.9
for patients with four or more OFF visits against 4.5 for the mean
(``reports/diagnosis-2026-09-18/b3_bayes_level.py``). These columns hand that upper end to
the model directly — the maximum, the upper quartile, the mean, and where the current
visit sits relative to the patient's own maximum.

Everything here is a plain groupby over ``obs``, so there is nothing to fit. All of it is
patient-level, which matters in two places: the design matrix carries it on every visit,
and the forecast validation (``evaluation/forward.py``) must derive it from the *known*
visits only — see ``PATIENT_LEVEL_COLS`` there.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT
from .base import Transformer
from .ratio import OBS

OBS_MAX_P = "obs_max_p"
OBS_Q75_P = "obs_q75_p"
OBS_MEAN_P = "obs_mean_p"
OFF_MAX_P = "off_max_p"
OFF_N_P = "off_n_p"
ON_N_P = "on_n_p"
OBS_REL_MAX = "obs_rel_max"  # this visit's obs relative to the patient's maximum

AGGREGATE_COLS = [OBS_MAX_P, OBS_Q75_P, OBS_MEAN_P, OFF_MAX_P, OFF_N_P, ON_N_P]


class PatientAggregates(Transformer):
    """Patient-level summaries of ``obs``, ``off`` and ``on``; nothing to fit."""

    name = "patient_aggregates"

    # Measured (tests/test_properties.py): `fit` ignores y; every column is a whole-patient
    # groupby, so later visits move an earlier row — the forecast derives these from the
    # *known* visits only (`PATIENT_LEVEL_COLS` in `evaluation/forward.py`).
    reads_target = False
    causal = False

    def __init__(self, obs_col: str = OBS) -> None:
        # `obs_pk` (features/pk_response.py) is the same quantity with the patient's own
        # drug response divided out; a config picks it up by name.
        self.obs_col = obs_col

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.require(df, [PATIENT, self.obs_col, "off", "on"])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.require(df, [PATIENT, self.obs_col, "off", "on"])
        out = df.copy()
        g = out.groupby(PATIENT, sort=False)
        out[OBS_MAX_P] = g[self.obs_col].transform("max")
        out[OBS_Q75_P] = g[self.obs_col].transform(lambda s: s.quantile(0.75))
        out[OBS_MEAN_P] = g[self.obs_col].transform("mean")
        out[OFF_MAX_P] = g["off"].transform("max")
        out[OFF_N_P] = g["off"].transform("count").astype(float)
        out[ON_N_P] = g["on"].transform("count").astype(float)
        out[OBS_REL_MAX] = relative_to_max(
            out[self.obs_col].to_numpy(dtype=float), out[OBS_MAX_P].to_numpy(dtype=float)
        )
        return self.preserve_index(out, df)

    def get_params(self) -> dict[str, Any]:
        return {"obs_col": self.obs_col, "columns": [*AGGREGATE_COLS, OBS_REL_MAX]}


def relative_to_max(obs: np.ndarray, obs_max: np.ndarray) -> np.ndarray:
    """``obs / max(obs_max, 1)`` — shared with the forecast path, which recomputes it
    from a carried maximum rather than the future visit's own."""
    return np.asarray(obs, dtype=float) / np.maximum(np.asarray(obs_max, dtype=float), 1.0)
