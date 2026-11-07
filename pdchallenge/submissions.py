"""The submission ledger: which file was uploaded, and what the board said about it.

The gap this closes is the one the audit opens with (docs/audit-2026-09-16.md §3 item 1).
When the leaderboard returned 17.17 for a model whose cross-validation said 4.28, the
question "which run was that?" had no answer anywhere in the repository. A run directory
records what was *built*; nothing recorded what was *sent*, and the only thing joining the
two is a score. That the submission turned out to be fine and the units were the problem is
beside the point — it took a day to establish, and this file is what would have made it
five minutes.

One row per submission, keyed by run directory:

* **what was sent** — the file, its row count, and the mean/spread of its predictions, so a
  file that collapsed to a constant is visible in the ledger itself.
* **what produced it** — the configuration fingerprint *and* the code fingerprint, which
  together are what :meth:`RunConfig.fingerprint` now hashes. Seven historical runs shared
  one settings fingerprint across a 0.17 RMSE spread, so the settings alone are not enough.
* **what came back** — the public and private board scores. Typed in after uploading,
  because nothing local can know them.

The ledger lives at ``reports/submissions.csv`` and is append-or-update, never rewritten:
re-recording the same run directory fills in its score rather than adding a second row.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import paths
from .config import RunConfig, code_fingerprint

COLUMNS = [
    "submitted_at",
    "run_dir",
    "tag",
    "fingerprint",
    "config_fingerprint",
    "code_fingerprint",
    "file",
    "n_rows",
    "pred_mean",
    "pred_sd",
    "board_public",
    "board_private",
    "note",
]


def read(path: Path | str | None = None) -> pd.DataFrame:
    """The ledger as a frame, empty (with the right columns) if it does not exist yet."""
    path = Path(path) if path is not None else paths.SUBMISSIONS_CSV
    if not path.is_file():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def _rel(path: Path) -> str:
    """A path recorded relative to the repo when it is inside it, absolute otherwise.

    Relative, so the ledger survives the repository being moved or cloned elsewhere — an
    absolute path would be a record of this machine rather than of the experiment.
    """
    try:
        return path.relative_to(paths.ROOT).as_posix()
    except ValueError:  # outside the repo (tests, scratch directories)
        return str(path)


def _key(run_dir: Path) -> str:
    """How a run directory is named in the ledger."""
    return _rel(run_dir)


def find(ledger: pd.DataFrame, run_dir: Path | str) -> pd.Series | None:
    """The ledger row recorded for ``run_dir``, or ``None`` if it has not been recorded."""
    key = _key(Path(run_dir))
    match = ledger.index[ledger["run_dir"] == key] if len(ledger) else []
    return ledger.loc[match[0]] if len(match) else None


def _run_facts(run_dir: Path) -> dict[str, Any]:
    """Everything the run directory itself can answer about the submission it produced.

    Fingerprints are read back from ``metrics.csv`` when it has them, rather than
    recomputed: the point of a fingerprint is what the code was *then*, and re-hashing the
    current source would relabel an old submission as a product of today's code. Runs
    predating the code fingerprint simply leave those cells empty.
    """
    cfg_path = run_dir / paths.CONFIG_SNAPSHOT
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{run_dir} has no {paths.CONFIG_SNAPSHOT}; is it a run directory?")
    cfg = RunConfig.from_yaml(cfg_path)

    submission = run_dir / paths.SUBMISSION_CSV
    if not submission.is_file():
        raise FileNotFoundError(f"{run_dir} has no {paths.SUBMISSION_CSV}; nothing to record")

    facts: dict[str, Any] = {
        "tag": cfg.tag,
        "fingerprint": cfg.fingerprint(),
        "config_fingerprint": cfg.config_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }

    metrics_path = run_dir / paths.METRICS_CSV
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path).iloc[0]
        for column in ("fingerprint", "config_fingerprint", "code_fingerprint"):
            value = metrics.get(column)
            # NaN for the columns older runs did not carry; keep the recomputed default.
            if value is not None and not pd.isna(value):
                facts[column] = str(value)

    preds = pd.read_csv(submission)["target"]
    facts["file"] = _rel(submission)
    facts["n_rows"] = int(len(preds))
    facts["pred_mean"] = round(float(preds.mean()), 4)
    facts["pred_sd"] = round(float(preds.std()), 4)
    return facts


def record(
    run_dir: Path | str,
    *,
    public: float | None = None,
    private: float | None = None,
    note: str = "",
    path: Path | str | None = None,
    now: str | None = None,
) -> pd.DataFrame:
    """Add or update this run directory's ledger row, and return the whole ledger.

    ``now`` exists so a test can pin the timestamp; in normal use it is the current time,
    which is the upload time as far as the ledger is concerned.
    """
    run_dir = Path(run_dir)
    ledger_path = Path(path) if path is not None else paths.SUBMISSIONS_CSV
    ledger = read(ledger_path)

    facts = _run_facts(run_dir)
    key = _key(run_dir)

    row = {
        "submitted_at": now or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": key,
        **facts,
        "board_public": public,
        "board_private": private,
        "note": note,
    }

    existing = ledger.index[ledger["run_dir"] == key] if len(ledger) else []
    if len(existing):
        # Fill in rather than duplicate: the same submission re-recorded with a score is
        # the same submission. An omitted score keeps whatever was already known.
        previous = ledger.loc[existing[0]]
        for column in ("board_public", "board_private", "note"):
            if row[column] is None or row[column] == "":
                row[column] = previous[column]
        ledger.loc[existing[0]] = {**previous.to_dict(), **row}
    else:
        ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger[COLUMNS].to_csv(ledger_path, index=False)
    return ledger[COLUMNS]


def board_score(raw: float, *, metric: str = "mse") -> float:
    """Convert a leaderboard number to the scale this project reports, or vice versa.

    **The board reports MSE.** The proof is arithmetic and does not depend on trusting the
    organisers: a constant prediction ``c`` on a target bounded in [0, 132] has
    ``|error| <= max(c, 132 - c)``, so its *RMSE* cannot exceed 94.67 — and the constant
    submission scored 278.83. Read as MSE everything lines up, including the training
    target's variance of 272.39 against that same 278.83.

    This function exists so the conversion is done by a named thing rather than by hand at
    the point of comparison, which is exactly where it went wrong before.
    """
    if metric == "mse":
        return raw
    if metric == "rmse":
        import numpy as np

        return float(np.sqrt(raw))
    raise ValueError(f"unknown metric {metric!r}; expected 'mse' or 'rmse'")
