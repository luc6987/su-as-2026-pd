"""A member looking at the chain through the switch positions it was trained under.

A model in this package is handed the chain's *feature frame* — the pipeline's output, not a
design matrix — and this wrapper lets one member read that frame under a different view: the
steps named in ``view`` are rebuilt from their own specs, their columns recomputed and replaced
in place, and only then does the member see the frame. The split, the target and every other
column stay the current chain's.

**Why a view of its own.** The 2026-10-14 round found that a member trained on the *older* chain
is worth more inside the newer blend than the same member retrained on the new chain. D140
(`reports/round-2026-10-14/planner/p2_chain_mix_v9.txt`): pairing v9's boosters with the v8-chain
NN members blends 0.0142 better than the v9-chain retrains, although neither set is individually
better — the booster takes the new chain's gain while the NN member keeps the old view, so their
errors decorrelate. D139 (same file): S_old, a GRU pre-trained and fine-tuned on the v8 chain,
added to v9's blend is worth +0.0239 on mining (three seeds), against +0.0134 on v8; the round's
own read of the assembled v10M is Δ −0.0387 mining / −0.0604 confirm
(`reports/round-2026-10-14/M/03_step1.txt`). So the member that carries that gain needs a way to
say "compute `floor` and `edge` the way I was trained" while everything else is the current
chain's — which is what this wrapper does, as task 10's `consolidated_v10.yaml` uses it::

    model:
      name: chain_view
      view: [floor, edge]        # the v8 view of a v9 chain
      base: {name: visit_gru, seed: 0, reps: 2, layers: 3, epochs: 40, threads: 1,
             pretrain: mvm, pre_epochs: 60}

**Only the three steps this round's switches live in are supported** — `floor`
(``obs_known_dt_only``), `edge` (``timing``) and `patient_aggregates` (C2's ``known_dt_only``).
Anything else is a ``ValueError`` at construction, because a step the wrapper silently skipped
would be a different view than the one the config asked for, and a wrong view trains a member
that is quietly not the one that was measured.

**`edge`'s cross-fitting comes for free.** `edge.fit` reads the training fold's target and, when
it is then handed the frame it was fitted on, returns cross-fitted values — it recognises that
frame by its ``Index`` and patient columns (`edge._frame_key`), which is exactly how
``FeaturePipeline.fit_transform`` calls it. Dropping columns changes neither of those, so the
wrapper's `fit` hands `edge` the frame it was fitted on and gets the out-of-fold training rows
with no extra bookkeeping — the training frame the member sees is the one a plain chain would
have given it, verified bit for bit against the v8 pipeline in
``reports/round-2026-10-14/pkg/09_parity.txt``.

**It changes nothing that does not use it.** No config names `chain_view`; a member that is not
wrapped in it sees the frame untouched, and the wrapper is only ever reached through a spec.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import INDEX
from ..features.aggregates import AGGREGATE_COLS, OBS_REL_MAX
from ..features.base import Transformer
from ..features.edge import EDGE_COLS
from ..features.floor import FLOOR_COLS
from ..features.pipeline import build as build_step
from .base import BaseModel
from .registry import build, register, spec_name

#: The steps this wrapper may recompute, and the columns each one writes — read off the classes'
#: own constants so a step that adds a column cannot drift away from the list kept here.
_VIEW_STEPS: dict[str, list[str]] = {
    "floor": list(FLOOR_COLS),
    "edge": list(EDGE_COLS),
    "patient_aggregates": [*AGGREGATE_COLS, OBS_REL_MAX],
}


def _spec_parts(spec: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """``"floor"`` / ``{"name": "floor", "obs_known_dt_only": true}`` as ``(name, params)``."""
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict) and isinstance(spec.get("name"), str):
        params = dict(spec)
        return params.pop("name"), params
    raise TypeError(f"cannot interpret feature spec {spec!r} as a chain_view step")


@register
class ChainViewModel(BaseModel):
    """A member that sees the chain frame with ``view``'s steps recomputed from their own specs.

    ``base`` is any model spec, like :class:`~pdchallenge.models.smoothed.SmoothedModel`'s, so a
    view can wrap an ensemble as easily as a single model. It deliberately has **no** ``seed``
    parameter of its own: the seed belongs to the member, and
    :func:`~pdchallenge.models.registry.with_seed` walks ``base`` (the registry's ``CHILD_KEYS``)
    to put it there.
    """

    name = "chain_view"

    def __init__(self, base: Any, view: list[str | dict[str, Any]]) -> None:
        view = list(view)
        if not view:
            raise ValueError("view must name at least one feature step to recompute")
        names = [_spec_parts(spec)[0] for spec in view]
        unknown = [n for n in names if n not in _VIEW_STEPS]
        if unknown:
            raise ValueError(
                f"chain_view can only recompute {sorted(_VIEW_STEPS)} — the steps this round's "
                f"chain switches live in; got {unknown}. A step it skipped silently would be a "
                f"different view than the one the config asked for."
            )
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            raise ValueError(f"view names {twice} more than once; the steps would fight over columns")
        self.base = base
        self.view = view

    # -- the view ---------------------------------------------------------------------
    def _steps(self) -> list[Transformer]:
        """Fresh step objects from ``view``, in view order."""
        return [build_step(name, **params) for name, params in map(_spec_parts, self.view)]

    def _recompute(self, X: pd.DataFrame, y: pd.Series | None, *, fit: bool) -> pd.DataFrame:
        """``X`` with every view step's columns recomputed and put back where they were.

        With ``fit``, each step is fitted on this frame (out-of-fold where the step says so) and
        the fitted steps are kept in ``self.steps_``; without it they are only transformed, as in
        `predict`. The parity check calls this directly: the replacement is the whole of what it
        checks, and it needs no wrapped model to exercise it.
        """
        out = X.copy()
        steps = self._steps() if fit else self.steps_
        for step in steps:
            cols = _VIEW_STEPS[step.name]
            missing = [c for c in cols if c not in out.columns]
            if missing:
                raise KeyError(
                    f"chain_view recomputes `{step.name}`, whose columns {missing} are not in the "
                    f"frame it was handed. It wraps a chain member, so it needs the chain's "
                    f"feature frame — these columns are replaced, not added, and a view of a "
                    f"chain that never produced them is not a view of this chain."
                )
            # The step's input is the frame without *its own* output columns: exactly what a
            # chain that ran this step's parameters would have handed it. A previously
            # recomputed step's new values stay in place (the three steps read none of each
            # other's columns).
            Xd = out.drop(columns=cols)
            if fit:
                step.fit(Xd, y)
            produced = step.transform(Xd)
            # Row identity first: the steps restore the caller's index (`preserve_index`), so a
            # reordering here would attach every recomputed bound to another row's visit — a
            # frame that looks healthy and trains the member on someone else's answer.
            if not produced.index.equals(Xd.index):
                raise AssertionError(f"`{step.name}` changed the row order of the frame")
            if INDEX in produced.columns and not np.array_equal(
                produced[INDEX].to_numpy(), Xd[INDEX].to_numpy()
            ):
                raise AssertionError(f"`{step.name}` changed the {INDEX} column")
            # What the step wrote: its output beyond the frame it was handed. This is the class
            # constant checked against the step's actual output, so a column list that has
            # drifted apart from the step is caught here rather than half-replaced.
            wrote = [c for c in produced.columns if c not in Xd.columns]
            if wrote != cols:
                raise AssertionError(
                    f"`{step.name}` wrote {wrote}, expected {cols} — the column list and the "
                    f"step's output have drifted apart"
                )
            for c in cols:
                # Name, position and dtype are the frame's, not the step's: the member must see
                # a frame that differs from the chain's in these columns' values and in nothing
                # else, so a chain that carries float32 keeps its float32.
                out[c] = produced[c].astype(out[c].dtype).to_numpy()
        if list(out.columns) != list(X.columns):
            raise AssertionError("chain_view changed the frame's columns or their order")
        if fit:
            self.steps_ = steps
        return out

    # -- the model protocol ------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        # `groups` is handed through like `smoothed` does: a wrapped ensemble resamples
        # internally and has to keep the outer patient grouping.
        X_view = self._recompute(X, y, fit=True)
        self.base_ = build(self.base).fit(X_view, y, groups=groups)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.base_.predict(self._recompute(X, None, fit=False)), dtype=float)

    # `predict_tta` is deliberately not overridden: the default uses only the unscaled frame,
    # which is what the set models inside the blend do (the chain is not refitted per scale).

    def get_params(self) -> dict[str, Any]:
        return {
            "view": list(self.view),
            "base": spec_name(self.base),
            "base_params": self.base_.get_params(),
        }
