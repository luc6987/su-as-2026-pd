"""Turning an ON score into an OFF-equivalent observation.

Measured against the true target (``reports/diagnosis-2026-09-18/a2_structure.py``,
``a9_ratio_model.py``), the two scores are *multiplicative* views of the same visit:

    off ≈ T · m_v                       on ≈ T · m_v · ρ(ledd, time_since_intake_on)

``m_v`` is a one-sided per-visit factor shared by both scores (their same-visit deviations
correlate at 0.76), and ``ρ`` — the short-duration drug response — depends on the **dose**
as well as on the time since intake: ``E[on − T]`` runs from −11 to −34 across the
ledd × dt grid. So the right way to make an ON-only visit comparable with an OFF visit is
not to add a constant but to divide by the ratio the dose and intake time imply:

    on_eq = on / ρ̂(ledd, dt)

``ρ̂`` is fitted on the visits that carry *both* scores, which needs no target at all.
The resulting ``obs`` column (``off`` where it exists, ``on_eq`` otherwise) is a single
observation of ``T · m_v`` per visit, on one scale, and is what the envelope curve and the
patient aggregates downstream consume. A both-score visit keeps its ``off``: the two scores
share the visit factor, so averaging them would not average anything away.

``on_eq`` is noisier than ``off`` (dividing by ρ ≈ 0.6 inflates the residual: its SD
against the target is ~12 where ``off``'s is ~9), which is what ``obs_weight`` records
for the curve fit.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from .base import Transformer

RHO_HAT = "rho_hat"  # predicted on/off ratio at this visit
ON_EQ = "on_eq"  # the ON score rescaled onto the OFF scale
OBS = "obs"  # one OFF-scale observation per visit: off, else on_eq
OBS_WEIGHT = "obs_weight"  # relative precision of `obs` (1 for off, `on_weight` for on_eq)
OBS_IS_OFF = "obs_is_off"  # 1 when `obs` is a genuine OFF score

RATIO_INPUTS = ["ledd", "time_since_intake_on"]


class OnRatioTransformer(Transformer):
    """Fit ``on/off ~ (ledd, time_since_intake_on)`` and emit an OFF-equivalent ``obs``.

    ``fit`` ignores ``y``: the ratio is learned from visits that carry both scores. It
    therefore fits on any frame with the raw score columns, including the test set —
    but it is fitted on the training rows like every other step, so the CV stays honest.
    """

    name = "on_ratio"

    # Measured (tests/test_properties.py): `fit` ignores y — the ratio comes from the visits
    # that carry both scores — and `transform` is row-wise, so it only looks at the visit
    # itself. (The model is fitted on the training frame, never on future visits.)
    reads_target = False
    causal = True

    def __init__(
        self,
        min_off: float = 5.0,
        rho_floor: float = 0.15,
        rho_ceiling: float = 1.5,
        on_weight: float = 0.5,
        seed: int = 0,
    ) -> None:
        # Below `min_off` the ratio on/off is dominated by the integer rounding of a
        # near-zero denominator, so those visits are left out of the fit.
        self.min_off = min_off
        # The booster's predictions are bounded by the training range anyway; the floor
        # is a guard against dividing by a ratio that would blow `on_eq` up.
        self.rho_floor = rho_floor
        self.rho_ceiling = rho_ceiling
        self.on_weight = on_weight
        self.seed = seed

    # -- fit ------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.require(df, ["on", "off", *RATIO_INPUTS])
        both = df["on"].notna() & df["off"].notna() & (df["off"] >= self.min_off)
        if both.sum() < 20:
            raise ValueError(
                f"only {int(both.sum())} visits carry both scores with off >= {self.min_off}; "
                "cannot fit the on/off ratio"
            )
        X = df.loc[both, RATIO_INPUTS].to_numpy(dtype=float)
        ratio = (df.loc[both, "on"] / df.loc[both, "off"]).to_numpy(dtype=float)

        # Lazy import, as in features/ldr.py: the feature layer stays importable without
        # scikit-learn, and the histogram booster takes NaN (missing dose or intake time)
        # natively, which a table lookup would need explicit bins for.
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model_ = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=self.seed,
        ).fit(X, ratio)
        self.n_fit_ = int(both.sum())
        self.ratio_mean_ = float(ratio.mean())
        pred = self.model_.predict(X)
        denom = float(np.sum((ratio - ratio.mean()) ** 2))
        self.r2_in_sample_ = float(1.0 - np.sum((ratio - pred) ** 2) / denom) if denom > 0 else float("nan")
        return self

    # -- transform ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.require(df, ["on", "off", *RATIO_INPUTS])
        out = df.copy()

        rho = self.model_.predict(out[RATIO_INPUTS].to_numpy(dtype=float))
        rho = np.clip(rho, self.rho_floor, self.rho_ceiling)
        off = out["off"].to_numpy(dtype=float)
        on = out["on"].to_numpy(dtype=float)
        has_off = ~np.isnan(off)

        on_eq = on / rho
        out[RHO_HAT] = rho
        out[ON_EQ] = on_eq
        out[OBS] = np.where(has_off, off, on_eq)
        out[OBS_WEIGHT] = np.where(has_off, 1.0, self.on_weight)
        out[OBS_IS_OFF] = has_off.astype(int)
        return self.preserve_index(out, df)

    # -- introspection --------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        return {
            "n_fit": getattr(self, "n_fit_", 0),
            "ratio_mean": round(getattr(self, "ratio_mean_", float("nan")), 4),
            "r2_in_sample": round(getattr(self, "r2_in_sample_", float("nan")), 4),
            "min_off": self.min_off,
            "rho_floor": self.rho_floor,
            "on_weight": self.on_weight,
        }
