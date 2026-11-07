"""Per-visit lower bounds on the latent score from the patient's own earlier OFF scores.

The latent target never decreases inside a patient: over the 48,632 consecutive steps of the
training set not one goes down (484 are flat), and the OFF observation is ``T · m_v`` with
``m_v``'s right edge near 1.0–1.1 (CLAUDE.md §5; the first visit sits at m ≈ 0.98, later ones
reach ~1.05). So every OFF score at or *before* a visit bounds that visit's ``T`` from below:
``T_v >= T_u >= off_u / m_max`` for ``u <= v``. The chain already has the patient-wide maxima
(`obs_max_p`, `off_max_p`), but those include the patient's *later* visits and so bound nothing
at an individual visit; and a tree cannot assemble a running maximum out of eleven neighbour
columns with missing values in them. This step writes the running maximum directly.

Measured through the whole v3 chain on dev (single booster + smoothing, seeds 0/1/2,
`reports/mining-2026-09-29/pinning/floor_probe.py`, variant F2 = these eight columns):
6.9484 / 6.9005 / 6.8899 → 6.4053 / 6.4740 / 6.4339, pooled −0.4752 [−0.5746, −0.3721]; the
running maximum of ``obs_pk`` (ON scores converted to the OFF scale) is the largest single piece
(F1 → F2 on seed 0: −0.356 → −0.543). Adding an isotonic fit of the observations or running
maxima of the chain's own curves on top measured no better (F6), and a future-side "ceiling"
(running minimum over later visits) added nothing (F3); `docs/mining-2026-09-29.md` §4.2.

Every column uses the visit itself and the visits before it, never a later one, so the step
is causal by construction; the forecast path only has to make sure a future visit does not
see the *other* future visits it is predicted alongside (`evaluation/forward.py`). Nothing is
fitted and no target is read. It must run after `likelihood` and `envelope_curve`, whose
curves it is compared against — in practice as the last step of the chain.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT
from .base import Transformer

_POS = "__floor_pos"

#: The columns, in the order the step appends them (the order is part of the design
#: matrix's contract — `colsample_bytree` samples by position).
FLOOR_COLS = [
    "fl_cummax_off",           # running max of OFF, carried forward over visits without one
    "fl_lik_minus_cummax",     # the likelihood curve against that floor
    "fl_floor1",               # running max of OFF / (first_scale at the first visit, later_scale after)
    "fl_lik_minus_floor1",
    "fl_env_minus_floor1",
    "fl_floor1_age_gap",       # how long ago the scaled floor was last raised
    "fl_cummax_obs",           # running max of the OFF-scale observation (ON converted), every visit
    "fl_lik_minus_cummax_obs",
]

# Which step writes a column this one reads, for the error message.
_PRODUCED_BY = {
    "obs_pk": "pk_response",
    "lik_pred": "likelihood",
    "env_pred": "envelope_curve",
}


class FloorFeatures(Transformer):
    """Running lower bounds on ``T`` from the patient's earlier OFF scores. Nothing to fit."""

    name = "floor"

    # Measured (tests/test_properties.py): nothing is fitted and no target is read; every
    # column is a running statistic over this visit and the earlier ones — causal by
    # construction, as the module docstring says.
    reads_target = False
    causal = True

    def __init__(
        self,
        first_scale: float = 1.00,
        later_scale: float = 1.05,
        off_col: str = "off",
        obs_col: str = "obs_pk",
        lik_col: str = "lik_pred",
        env_col: str = "env_pred",
        time_col: str = "age",
        obs_known_dt_only: bool = False,
    ) -> None:
        if first_scale <= 0 or later_scale <= 0:
            raise ValueError(f"scales must be positive, got {first_scale!r} / {later_scale!r}")
        # m's right edge by position: the first visit is the near-exact one (median m 0.98,
        # 46% of its mass within ±4% of 1), later visits reach ~1.05 (CLAUDE.md §5).
        self.first_scale = first_scale
        self.later_scale = later_scale
        self.off_col = off_col
        self.obs_col = obs_col
        self.lik_col = lik_col
        self.env_col = env_col
        self.time_col = time_col
        # Task 01 of the 2026-10-03 pack: only OFF and intake-time-known observations may
        # raise the `obs_pk` running bound — a conversion with no intake time is ~3× noisier
        # (SD 0.144 against 0.080, `reports/v6-dtna/logs/facts.log`), so letting it raise the
        # floor would bake its error in as a bound. The bound still carries *forward* over
        # such a visit (`running_max` ffills), and the other columns are untouched. Off, the
        # step is the shipped one, bit for bit.
        self.obs_known_dt_only = obs_known_dt_only

    @property
    def columns(self) -> list[str]:
        """The new column names, in the order ``transform`` appends them."""
        return list(FLOOR_COLS)

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Nothing to fit; every column is a running statistic of the patient's own rows."""
        self._require(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require(df)
        out = df.copy()

        work = out[[PATIENT, self.time_col, self.off_col, self.obs_col, self.lik_col, self.env_col]].copy()
        if self.obs_known_dt_only:
            work["time_since_intake_on"] = out["time_since_intake_on"]
        work[_POS] = np.arange(len(work))
        work = work.sort_values([PATIENT, self.time_col, _POS], kind="mergesort")
        pid = work[PATIENT]
        first = work.groupby(pid, sort=False).cumcount().to_numpy() == 0

        def running_max(s: pd.Series) -> pd.Series:
            # `cummax` leaves a row without a value at NaN; carry the bound forward over it.
            return s.groupby(pid, sort=False).cummax().groupby(pid, sort=False).ffill()

        t = work[self.time_col].astype(float)
        off = work[self.off_col].astype(float)
        lik = work[self.lik_col].astype(float)
        env = work[self.env_col].astype(float)

        cm = running_max(off)
        scaled = pd.Series(off.to_numpy(dtype=float) / np.where(first, self.first_scale, self.later_scale),
                           index=work.index)
        fl = running_max(scaled)
        raised = scaled.eq(fl) & scaled.notna()
        set_at = pd.Series(np.where(raised, t, np.nan), index=work.index).groupby(pid, sort=False).ffill()
        obs = work[self.obs_col].astype(float)
        if self.obs_known_dt_only:
            # A conversion with no intake time carries ~3× the noise, so it may not *raise*
            # the bound — but `running_max` ffills, so an earlier bound still carries over it.
            obs = obs.where(work[self.off_col].notna() | work["time_since_intake_on"].notna())
        co = running_max(obs)

        new = {
            "fl_cummax_off": cm,
            "fl_lik_minus_cummax": lik - cm,
            "fl_floor1": fl,
            "fl_lik_minus_floor1": lik - fl,
            "fl_env_minus_floor1": env - fl,
            "fl_floor1_age_gap": t - set_at,
            "fl_cummax_obs": co,
            "fl_lik_minus_cummax_obs": lik - co,
        }
        if list(new) != self.columns:
            raise AssertionError("floor produced columns in a different order than `columns` promises")
        order = np.argsort(work[_POS].to_numpy(), kind="stable")
        for name, series in new.items():
            out[name] = series.to_numpy(dtype=float)[order]
        return self.preserve_index(out, df)

    def _require(self, df: pd.DataFrame) -> None:
        needed = [PATIENT, self.time_col, self.off_col, self.obs_col, self.lik_col, self.env_col]
        if self.obs_known_dt_only:
            needed.append("time_since_intake_on")
        missing = [c for c in needed if c not in df.columns]
        if missing:
            named = [
                f"{c} (written by the `{_PRODUCED_BY[c]}` step)" if c in _PRODUCED_BY else c
                for c in missing
            ]
            raise KeyError(
                f"floor needs columns the chain has not produced yet: {named}. It compares the "
                f"running bound with the `likelihood` and `envelope_curve` curves, so it runs "
                f"after them — in practice as the last step of the chain."
            )

    def get_params(self) -> dict[str, Any]:
        params = {
            "first_scale": self.first_scale,
            "later_scale": self.later_scale,
            "time_col": self.time_col,
            "columns": list(FLOOR_COLS),
        }
        if self.obs_known_dt_only:
            params["obs_known_dt_only"] = True
        return params


__all__ = ["FloorFeatures", "FLOOR_COLS"]
