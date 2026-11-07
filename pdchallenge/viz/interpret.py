"""SHAP attribution, plus the model-specific fallbacks.

Two details make this less trivial than calling ``shap.TreeExplainer``:

* Models here consume a **design matrix** built by
  :func:`pdchallenge.features.matrix.build_matrix`, and their ``predict`` re-derives that
  matrix from a raw frame. Explaining must therefore call the *inner* estimator on a
  matrix that has already been built, or the features get encoded twice.
* Not every model is tree-shaped. The dispatcher below picks TreeExplainer where it is
  exact and fast, and falls back to a permutation explainer everywhere else so the
  analysis still runs for the ridge or the curve model.

SHAP values are in points of MDS-UPDRS III, which is the unit clinicians actually use —
that makes the output directly comparable with the literature's effect sizes.
"""
from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

import numpy as np
import pandas as pd

from ..features.matrix import build_matrix


def _inner_predict(model: Any, Xm: pd.DataFrame) -> np.ndarray:
    """Predict from an already-built design matrix, skipping re-encoding."""
    est = getattr(model, "estimator_", None)
    if est is not None and hasattr(est, "predict"):
        return np.asarray(est.predict(Xm), dtype=float)
    pipe = getattr(model, "pipeline_", None)
    if pipe is not None and hasattr(pipe, "predict"):
        return np.asarray(pipe.predict(Xm), dtype=float)
    raise TypeError(
        f"don't know how to explain {type(model).__name__} from a design matrix; "
        "add a branch here or use a model with `estimator_` / `pipeline_`"
    )


def design_matrix(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """The matrix a model actually consumes; honours a model's own column subset."""
    m = build_matrix(X)
    cols = getattr(model, "columns", None)
    if cols:
        m = m[[c for c in cols if c in m.columns]]
    elif getattr(model, "columns_", None):
        m = m[[c for c in model.columns_ if c in m.columns]]
    return m


def is_explainable(model: Any) -> bool:
    return hasattr(model, "estimator_") or hasattr(model, "pipeline_")


def shap_values(
    model: Any,
    X: pd.DataFrame,
    max_rows: int = 3000,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return ``(shap_values, design_matrix)``.

    ``max_rows`` subsamples because a permutation explainer costs a model call per feature
    per row; 3,000 visits is plenty to see the structure.
    """
    import shap

    Xm = design_matrix(model, X)
    if len(Xm) > max_rows:
        idx = np.random.default_rng(seed).choice(len(Xm), max_rows, replace=False)
        Xm = Xm.iloc[np.sort(idx)]

    est = getattr(model, "estimator_", None)
    if est is not None and hasattr(est, "get_booster"):  # xgboost / lightgbm
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(Xm, check_additivity=False)
    else:
        # generic fallback: works for any model with a matrix-level predict
        explainer = shap.PermutationExplainer(
            lambda z: _inner_predict(model, pd.DataFrame(z, columns=Xm.columns, index=Xm.index)),
            Xm.iloc[: min(200, len(Xm))],
        )
        sv = explainer(Xm).values

    return np.asarray(sv, dtype=float), Xm


def plot_summary(sv: np.ndarray, Xm: pd.DataFrame, top: int = 18):
    """Beeswarm of the strongest features, in points of MDS-UPDRS III."""
    import shap

    # `summary_plot` draws onto the current figure and returns None, so the figure has to
    # be taken from pyplot — binding its result looked like it held the figure and did not.
    shap.summary_plot(sv, Xm, max_display=top, show=False, plot_size=(7.2, 5.4))
    import matplotlib.pyplot as plt

    return plt.gcf()


def plot_bar(sv: np.ndarray, Xm: pd.DataFrame, top: int = 18):
    """Mean absolute attribution — the honest importance ranking."""
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(sv, Xm, plot_type="bar", max_display=top, show=False, plot_size=(7.2, 5.0))
    return plt.gcf()


def importance_table(sv: np.ndarray, Xm: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature, with the mean signed contribution.

    The *sign* matters as much as the size: a feature can matter a lot and still cancel
    out on average.
    """
    return (
        pd.DataFrame(
            {
                "mean_abs_shap": np.abs(sv).mean(axis=0),
                "mean_shap": sv.mean(axis=0),
                "std_shap": sv.std(axis=0),
            },
            index=Xm.columns,
        )
        .sort_values("mean_abs_shap", ascending=False)
        .assign(share=lambda d: d["mean_abs_shap"] / d["mean_abs_shap"].sum())
    )


def write_report(model: Any, X: pd.DataFrame, out_dir: _Path, top: int = 18) -> pd.DataFrame | None:
    """Write the SHAP figures + a CSV of the importance table. Never fatal."""
    import matplotlib.pyplot as plt

    from .style import apply_style, save

    # Before either branch writes anything: `train` passes figures/shap, which it never
    # creates, so a non-explainable model (hierarchical, segmented) used to raise
    # FileNotFoundError here and have it swallowed by the caller's broad except — the
    # run reported "SHAP skipped" for a reason that had nothing to do with the model.
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_explainable(model):
        (out_dir / "shap_skipped.txt").write_text(
            f"{type(model).__name__} has no inner estimator to explain from a matrix.\n",
            encoding="utf-8",
        )
        return None

    apply_style()
    sv, Xm = shap_values(model, X)
    table = importance_table(sv, Xm)
    table.to_csv(out_dir / "shap_importance.csv")

    for name, fn in (("shap_summary", plot_summary), ("shap_bar", plot_bar)):
        try:
            fig = fn(sv, Xm, top=top)
            save(fig, out_dir / f"{name}.png")
        except Exception:  # noqa: BLE001 - a plot must not fail the run
            plt.close("all")

    return table
