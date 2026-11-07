"""Chaining transformers, and looking them up by name from config.

A pipeline fits each step in order, feeding each step's output into the next, so later
steps can build on earlier ones (``debias`` needs the raw scores; ``patient_curve``
needs ``debias``'s ``pseudo_target``).

Only the ``fit`` path passes ``y``. ``transform`` never does — see
:mod:`pdchallenge.features.base`.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

import pandas as pd

from .aggregates import PatientAggregates
from .base import Transformer
from .debias import DebiasTransformer
from .derived import DerivedFeatures
from .envelope import EnvelopeCurveFeatures
from .edge import ConditionalEdgeFeatures
from .floor import FloorFeatures
from .likelihood import LikelihoodFeatures
from .longitudinal import PatientCurveFeatures
from .neighbour import NeighbourFeatures
from .pk_response import PatientPKResponse
from .ratio import OnRatioTransformer

# Name -> class. ``@register`` on a class adds it, so a new transformer becomes
# addressable from YAML config without touching any dispatch code.
FEATURES: dict[str, type[Transformer]] = {}


def register(cls: type[Transformer]) -> type[Transformer]:
    key = cls.name
    if key in FEATURES and FEATURES[key] is not cls:
        raise ValueError(f"duplicate feature name {key!r}")
    FEATURES[key] = cls
    return cls


class FeaturePipeline(Transformer):
    """An ordered chain of transformers, itself a transformer."""

    name = "pipeline"

    def __init__(self, steps: Iterable[Transformer]) -> None:
        self.steps: list[Transformer] = list(steps)

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        current = df
        for step in self.steps:
            current = step.fit_transform(current, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        current = df
        for step in self.steps:
            current = step.transform(current)
        return current

    def add(self, step: Transformer) -> Self:
        self.steps.append(step)
        return self

    def get_params(self) -> dict[str, Any]:
        return {step.name: step.get_params() for step in self.steps}

    def step(self, name: str) -> Transformer:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(f"no step named {name!r} in pipeline")


def build(name: str, **kwargs: Any) -> Transformer:
    """Instantiate a registered transformer by name."""
    if name in FEATURES:
        return FEATURES[name](**kwargs)
    raise KeyError(f"unknown feature {name!r}; available: {sorted(FEATURES)}")


def build_pipeline(specs: Iterable[str | dict[str, Any]]) -> FeaturePipeline:
    """Build a pipeline from a config list, e.g. ``["derived", {"name": "debias"}]``."""
    steps: list[Transformer] = []
    for spec in specs:
        if isinstance(spec, str):
            steps.append(build(spec))
        elif isinstance(spec, dict):
            params = dict(spec)
            name = params.pop("name")
            steps.append(build(name, **params))
        else:
            raise TypeError(f"cannot interpret feature spec {spec!r}")
    return FeaturePipeline(steps)


# Register the built-ins. Kept at the bottom so the classes are defined first. The last
# ones are the structure-aware chain (docs/diagnosis-2026-09-18.md): `on_ratio` must run
# before `patient_aggregates` and `envelope_curve`, which both consume its `obs` column,
# and before `pk_response`, which refines that ratio per patient. `likelihood` (merged from
# branch 4444354) fits its own ratio model and reads only the raw scores, so it can sit
# anywhere after `on_ratio`. `neighbour` goes last of all: it reads the residuals
# `envelope_curve` and `likelihood` write, so a chain that puts it earlier cannot run.
for _cls in (
    DerivedFeatures,
    DebiasTransformer,
    PatientCurveFeatures,
    OnRatioTransformer,
    PatientPKResponse,
    PatientAggregates,
    EnvelopeCurveFeatures,
    LikelihoodFeatures,
    NeighbourFeatures,
    # `floor` reads `lik_pred` / `env_pred` like `neighbour` does; registered last, run last.
    FloorFeatures,
    # `edge` reads the `pk_response` conversion and `lik_pred`; it goes after `floor`.
    ConditionalEdgeFeatures,
):
    register(_cls)
