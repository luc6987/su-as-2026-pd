"""The model protocol.

Every model — including :class:`~pdchallenge.models.ensemble.EnsembleModel`, which is
built out of other models — satisfies this interface. That is the whole point: an
ensemble is a model, so ensembling an ensemble costs nothing at the call site and
stacking depth is a config decision rather than a code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Fit on a design matrix, predict a 1-D array."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        """Fit the model.

        ``groups`` is the patient id per row. Most models ignore it, but anything that
        needs internal resampling (the ensemble) must use it, so that internal folds are
        patient-disjoint in the same way the outer CV is.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on rows sharing the schema passed to :meth:`fit`."""

    def predict_tta(self, frames: dict[float, pd.DataFrame]) -> np.ndarray:
        """Prediction from the chain's frames at several score scales (scale -> frame).

        The default ignores the scaled frames and predicts the unscaled one, so a model takes
        part in test-time augmentation only by overriding this (:mod:`..evaluation.tta`).
        """
        return self.predict(frames[1.0])

    def fit_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(X, y, groups=groups).predict(X)

    def get_params(self) -> dict[str, Any]:
        """Fitted state worth recording, for the run log."""
        return {}
