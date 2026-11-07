"""The feature-transformer protocol.

A transformer receives a visit-level frame and returns the same frame with extra
columns. Two rules keep the pipeline honest:

1. **Row order and index are preserved.** Everything here is row-wise aligned to the
   input; anything that would reorder rows must restore the original index.
2. **``transform`` never needs the target.** ``fit`` sees ``y``; ``transform`` must work
   on the test set, where no target exists. A transformer that peeks at the target
   during ``transform`` is leaking.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd

from ..data.schema import INDEX


class Transformer(ABC):
    """Fit on training rows, then apply to any rows with the same schema.

    Two measured properties every step declares (``tests/test_properties.py`` measures both
    per registered step, on the chain in ``configs/CURRENT``, and asserts the declaration):

    * ``reads_target`` — whether the step's *fitted state* depends on the training target.
      ``transform`` never receives ``y``; a step that learns from it must learn in ``fit``.
    * ``causal`` — whether a row's output depends only on that row and the rows *before* it
      in time. Dropping each patient's last visits must leave the retained rows' values
      unchanged for a causal step; the patient-level steps are not (the forecast validation
      rebuilds their columns from the known visits, ``evaluation/forward.py``).

    The defaults are the conservative ones; every registered step overrides both explicitly.
    """

    name: str = "transformer"
    reads_target: bool = True
    causal: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Learn whatever the transformer needs. Must return ``self``."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with the transformer's columns added."""

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def get_params(self) -> dict[str, Any]:
        """Fitted parameters, for logging and reproducibility."""
        return {}

    # -- helpers for subclasses -----------------------------------------------------
    @staticmethod
    def require(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"{type(df).__name__} is missing required columns: {missing}")

    @staticmethod
    def preserve_index(out: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """Row-align ``out`` to ``original`` and restore ``original``'s index.

        Length alone does not prove the rows still line up, and relabelling a *reordered*
        frame by position attaches every row to another row's label — a result that looks
        perfectly healthy and scores far worse than the model is worth. That is the one
        failure mode ``cli.write_submission``'s docstring describes, so when both frames
        carry the ``Index`` column it is used to put the rows back in the caller's order
        rather than trusted to position. Frames without that column (the unit fixtures)
        have no row identity to check against and fall back to the positional contract.
        """
        if len(out) != len(original):
            raise ValueError(f"row count changed: {len(original)} -> {len(out)}")
        out = out.copy()
        if INDEX in out.columns and INDEX in original.columns:
            got = out[INDEX].to_numpy()
            want = original[INDEX].to_numpy()
            if len(np.unique(got)) != len(got):
                raise ValueError(f"duplicate {INDEX} values: rows cannot be identified")
            if not np.array_equal(got, want):
                # `take` needs a position per wanted label; a missing label means the two
                # frames no longer describe the same rows, which is not fixable here.
                position = pd.Series(np.arange(len(got)), index=got).reindex(want)
                if position.isna().any():
                    n_missing = int(position.isna().sum())
                    raise ValueError(
                        f"{n_missing} {INDEX} values in the input are absent from the "
                        f"transformed frame — it dropped or invented rows"
                    )
                out = out.take(position.to_numpy())
        out.index = original.index
        return out
