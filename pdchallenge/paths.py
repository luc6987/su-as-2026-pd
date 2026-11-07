"""Single source of truth for every filesystem path in the project.

Nothing outside this module should build paths by hand. The rules:

* ``data/`` and ``archive/`` are **read-only inputs** — never write there.
* Every generated artefact goes under ``reports/``.
* A training run writes one self-contained directory, ``reports/baseline/<tag>_<stamp>/``,
  holding a config snapshot, metrics, predictions and figures, so a result can always be
  traced back to the exact configuration that produced it.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
ARCHIVE = ROOT / "archive"
CONFIGS = ROOT / "configs"

# `configs/CURRENT` holds one line — the file name of the incumbent chain in `CONFIGS`.
# `python -m pdchallenge.cli ... --config current` resolves through it (cli.resolve_config_path),
# so the pointer moves in one place when the configuration changes, not in every command and doc.
CURRENT_CONFIG = CONFIGS / "CURRENT"

# Report areas, one per kind of work.
EDA_REPORTS = REPORTS / "eda"
LITERATURE_REPORTS = REPORTS / "literature"
BASELINE_REPORTS = REPORTS / "baseline"

# The submission ledger — one row per upload. Checked in deliberately: it is the record of
# what the leaderboard said, which is the one thing that cannot be regenerated from the
# repository. See pdchallenge/submissions.py.
SUBMISSIONS_CSV = REPORTS / "submissions.csv"

# The evaluation holdout (docs/evaluation-protocol.md). 15% of the training patients are cut
# out once and frozen in a checked-in file; every model-selection decision from 2026-09-24 on
# is made on the other 85% (the *dev* patients), and the holdout is looked at only in the
# final integration. `HOLDOUT_LEDGER_CSV` is the record of every such look — the reason this
# lives next to the submission ledger rather than inside a run directory: it is one of the
# two things here that cannot be regenerated from the repository.
HOLDOUT_DIR = REPORTS / "holdout"
HOLDOUT_PATIENTS_CSV = HOLDOUT_DIR / "holdout_patients.csv"
HOLDOUT_LEDGER_CSV = HOLDOUT_DIR / "ledger.csv"
HOLDOUT_PREDS_DIR = HOLDOUT_DIR / "preds"

# The confirmation set (docs/evaluation-protocol.md §7): one third of the dev patients, frozen
# 2026-10-04 in a checked-in list. Mining work only looks at the other two thirds, so a
# structure found on dev and a Δ reported next to it are not measured on the same rows.
CONFIRMATION_PATIENTS_CSV = REPORTS / "protocol" / "confirmation_patients.csv"

# Artifact filenames inside a run directory. Kept here so writers and readers agree.
CONFIG_SNAPSHOT = "config.yaml"
METRICS_CSV = "metrics.csv"
PREDICTIONS_CSV = "predictions.csv"
SUBMISSION_CSV = "submission.csv"
SUMMARY_MD = "summary.md"
FIGURES_DIR = "figures"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_dir(name: str) -> Path:
    """``reports/<name>``, created if absent."""
    return _ensure(REPORTS / name)


def run_dir(tag: str, stamp: str | None = None) -> Path:
    """``reports/baseline/<tag>_<stamp>/``, created if absent.

    ``stamp`` may be passed explicitly to make a run reproducible in tests; otherwise
    it is the current UTC time.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return _ensure(BASELINE_REPORTS / f"{tag}_{stamp}")


def figures_dir(run: Path) -> Path:
    return _ensure(run / FIGURES_DIR)


def holdout_dir() -> Path:
    """``reports/holdout``, created if absent."""
    return _ensure(HOLDOUT_DIR)


def holdout_preds_path(tag: str, stamp: str | None = None) -> Path:
    """``reports/holdout/preds/<tag>_<stamp>.csv``, parent created if absent.

    One file per look at the holdout, kept so two configurations can be compared *paired* on
    the same rows later — a paired patient-level bootstrap needs both prediction vectors, and
    the ledger alone only stores the summary.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return _ensure(HOLDOUT_PREDS_DIR) / f"{tag}_{stamp}.csv"
