"""Name -> model class, so models are chosen by config rather than by editing code.

Note this module deliberately does **not** import the model implementations:
``ensemble.py`` imports :func:`build` from here, so a cyclic import would result.
Registration happens in :mod:`pdchallenge.models`.
"""
from __future__ import annotations

import inspect
from typing import Any

from .base import BaseModel

MODELS: dict[str, type[BaseModel]] = {}

# Spec keys whose value is itself a model spec — the ensemble's members and the segmented
# model's base. `with_seed` walks these so a nested model is seeded like a top-level one.
CHILD_KEYS = ("members", "base")


def register(cls: type[BaseModel]) -> type[BaseModel]:
    key = cls.name
    if key in MODELS and MODELS[key] is not cls:
        raise ValueError(f"duplicate model name {key!r}")
    MODELS[key] = cls
    return cls


def spec_name(spec: Any) -> str:
    """Human-readable name for a model spec, without instantiating anything.

    Lives here rather than next to any one model because both the ensemble and the
    segmented model use it for logging, and neither should import the other.
    """
    if isinstance(spec, BaseModel):
        return spec.name
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and "name" in spec:
        return str(spec["name"])
    return type(spec).__name__


def accepts_seed(name: str) -> bool:
    """Whether the named model's constructor takes ``seed``.

    Checked rather than assumed: ``ridge`` and ``hierarchical`` are deterministic and have
    no such parameter, so injecting one blindly would be a ``TypeError`` at build time.
    """
    cls = MODELS.get(name)
    return cls is not None and "seed" in inspect.signature(cls.__init__).parameters


def with_seed(spec: Any, seed: int) -> Any:
    """Return ``spec`` with ``seed`` set on every model in it that accepts one.

    ``RunConfig.model_seed`` used to be inert: declared, present in five YAMLs, asserted by
    the config tests, and read by no constructor anywhere. Changing it moved the
    fingerprint and no number, so a "vary the seed and check robustness" experiment
    silently compared a run with itself (docs/audit-2026-09-16.md §2 P2-1). Folding it into
    the spec is what makes the field mean what it says; walking :data:`CHILD_KEYS` is what
    makes it reach a nested member rather than only the outermost model.

    A ``seed`` already written into the spec is left alone — that is how a single member is
    pinned while the rest of the run varies.
    """
    if isinstance(spec, str):
        return {"name": spec, "seed": seed} if accepts_seed(spec) else spec
    if isinstance(spec, dict):
        out = dict(spec)
        for key in CHILD_KEYS:
            if key in out:
                child = out[key]
                out[key] = (
                    [with_seed(s, seed) for s in child]
                    if isinstance(child, list)
                    else with_seed(child, seed)
                )
        name = out.get("name")
        if "seed" not in out and isinstance(name, str) and accepts_seed(name):
            out["seed"] = seed
        return out
    return spec


def build(spec: str | dict[str, Any] | BaseModel) -> BaseModel:
    """Instantiate a model from ``"gbm"`` or ``{"name": "gbm", "params": {...}}``.

    Passing an already-built model returns it unchanged, which keeps nested ensembles
    straightforward.
    """
    if isinstance(spec, BaseModel):
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("name")
    else:
        raise TypeError(f"cannot interpret model spec {spec!r}")

    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**params)
