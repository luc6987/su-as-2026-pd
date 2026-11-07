"""Reference points to beat.

Any model we build has to be read against these, otherwise an MSE of 88.4 is just a
number. Three of them come from the EDA:

* **constant mean** — MSE 272.39 on train (it *is* the target's variance). The floor.
* **raw off** — MSE 100.40, but only on the 58% of visits that carry an ``off`` score.
* **raw on** — MSE 585.16, and worse than the constant mean.

The coverage column exists because of exactly that trap: ``raw off`` looks strong and is
undefined for two fifths of the visits. A model must be compared on a common set of rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mse, summarize


def naive_predictions(
    train_target: pd.Series,
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Predictions from each naive rule, as full-length arrays (NaN where undefined)."""
    n = len(df)
    const = np.full(n, float(np.mean(train_target)))

    off = df["off"].to_numpy(dtype=float) if "off" in df else np.full(n, np.nan)
    on = df["on"].to_numpy(dtype=float) if "on" in df else np.full(n, np.nan)

    # "use whatever score exists, preferring off" — the best a zero-effort entry can do
    fallback = np.where(~np.isnan(off), off, on)

    return {
        "constant_mean": const,
        "raw_off": off,
        "raw_on": on,
        "prefer_off_then_on": fallback,
    }


def compare(
    y_true: pd.Series,
    df: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    train_target: pd.Series | None = None,
) -> pd.DataFrame:
    """Score every candidate rule, including the naive references.

    ``candidates`` maps a name to a full-length prediction array. Naive rules are added
    automatically unless overridden by a candidate of the same name.
    """
    y_true = np.asarray(y_true, dtype=float)
    all_preds = dict(naive_predictions(train_target if train_target is not None else pd.Series(y_true), df))
    all_preds.update(candidates)

    rows = []
    for name, pred in all_preds.items():
        s = summarize(y_true, pred)
        s["name"] = name
        s["coverage"] = s["n"] / len(y_true)
        rows.append(s)

    out = pd.DataFrame(rows).set_index("name")[["n", "coverage", "mse", "mae", "bias", "r2"]]
    return out.sort_values("mse", na_position="last")


def compare_on_common_rows(
    y_true: pd.Series,
    df: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    reference: str = "raw_off",
) -> pd.DataFrame:
    """Restrict every candidate to the rows where ``reference`` is defined.

    This is the apples-to-apples comparison. Without it, ``raw_off`` gets scored only on
    its easy 58% while a model is scored on everything.
    """
    ref = naive_predictions(pd.Series(np.asarray(y_true, dtype=float)), df)[reference]
    mask = ~np.isnan(ref)
    sub_df = df.loc[mask]
    sub_y = np.asarray(y_true, dtype=float)[mask]
    sub_cands = {k: np.asarray(v, dtype=float)[mask] for k, v in candidates.items()}
    out = compare(pd.Series(sub_y), sub_df, sub_cands, train_target=pd.Series(sub_y))
    out.insert(0, "common_rows", f"{reference}-defined only")
    return out
