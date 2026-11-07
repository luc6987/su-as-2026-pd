"""Run configuration: what to build, and a fingerprint of it.

A config is plain data (YAML in, YAML out). Its ``fingerprint`` is a short hash of the
parts that actually change results, so two run directories can be compared at a glance
without diffing their logs.
"""
from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The structure-aware chain (docs/diagnosis-2026-09-18.md). The first three steps are the
# additive chain kept as a secondary view; the last three carry most of the signal.
DEFAULT_FEATURES: list[Any] = [
    "derived",
    "debias",
    "patient_curve",
    "on_ratio",
    "patient_aggregates",
    "envelope_curve",
]


@functools.lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Short hash of the package source — the half of an experiment a config cannot see.

    Settings alone were not enough to identify a run. Seven runs in ``reports/baseline``
    share one config fingerprint while their CV RMSE spans 4.2781-4.4448, because what
    changed between them was *code*, not configuration — the three fixes recorded in
    ``CLAUDE.md``. That is why "which run scored 18?" could not be answered from the run
    directories (docs/audit-2026-09-16.md §2 P1-8), and it is what this closes.

    Content, not ``git rev-parse``: uncommitted edits are precisely the ones worth
    catching, and several of those runs predate the repository.

    Cached for the process — the source does not change under a running job, and
    :meth:`RunConfig.fingerprint` is called several times per run.
    """
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


@dataclass
class RunConfig:
    """Everything needed to reproduce a run."""

    tag: str = "baseline"
    features: list[Any] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    model: Any = "gbm"
    n_splits: int = 5
    cv_seed: int = 0
    model_seed: int = 0
    # Score on the dev patients only (everything but the frozen holdout), so a number used to
    # *choose* something is not computed on the rows kept back to *confirm* it. Unlike the CV
    # seed, this is not another view of the same data: it removes rows. It is part of the
    # fingerprint for that reason, and the final `train` refit still uses every training row
    # — deployment has no reason to discard a patient. See docs/evaluation-protocol.md.
    dev_only: bool = False
    explain: bool = True
    write_submission: bool = True
    # Scale test-time augmentation (`evaluation/tta.py`): the input-score scales every prediction
    # is averaged over, e.g. [0.9, 0.95, 1.0, 1.05, 1.1]. Which members average is up to the model
    # (`gbm` with `tta: true`). Left out of the fingerprint while unset, so every configuration
    # written before the field existed keeps the fingerprint the holdout ledger knows it by.
    tta_scales: list[float] | None = None

    # -- serialisation ---------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: Path | str) -> RunConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Path | str) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    # -- identity --------------------------------------------------------------------
    def config_fingerprint(self) -> str:
        """Stable short hash of the result-affecting *settings* (not the tag)."""
        payload = {k: v for k, v in self.to_dict().items() if k != "tag"}
        if payload.get("tta_scales") is None:
            payload.pop("tta_scales", None)
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    def fingerprint(self) -> str:
        """Stable short hash of the settings **and** the package source.

        Two runs with the same fingerprint are the same experiment run twice; two runs
        differing only in code now differ here too, which is what the settings-only hash
        used to get wrong. :meth:`config_fingerprint` is still available on its own for
        comparing configurations.
        """
        blob = f"{self.config_fingerprint()}|{code_fingerprint()}"
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    def describe(self) -> str:
        model = self.model if isinstance(self.model, str) else self.model.get("name", "?")
        return f"{self.tag} · model={model} · features={'+'.join(map(str, self.features))}"
