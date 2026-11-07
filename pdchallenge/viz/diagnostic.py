"""Residual and calibration diagnostics for a fitted model.

Every figure answers a question that could change what we do next:

* where is the error concentrated? (residual vs prediction)
* does it grow with disease duration? (residual vs time since diagnosis)
* is the model systematically high or low in any band? (calibration)
* which kinds of visits are hard? (error by segment)
* do individual trajectories look right, or only the average? (patient trajectories)

All of them take a *prediction frame*: ``patient_id``, ``y_true``, ``y_pred``, ``residual``
plus whatever context columns the segmentation needs.
"""
from __future__ import annotations

from pathlib import Path as _Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .helpers import annotate_n, band, binned, reference_line
from .style import BLUE, INK2, ORANGE, PRED_COLOR, TRUE_COLOR, save

PRED_FRAME_COLS = ["patient_id", "y_true", "y_pred", "residual"]


def make_prediction_frame(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the frame every diagnostic below expects."""
    cols = ["patient_id"] + [c for c in (extra or []) if c in df.columns]
    out = df[cols].copy()
    out["y_true"] = np.asarray(y_true, dtype=float)
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    out["residual"] = out["y_pred"] - out["y_true"]
    return out


def plot_residual_vs_pred(df: pd.DataFrame, bins: int = 20) -> plt.Figure:
    """Residual against prediction level. A flat band at zero is what we want."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    c, p25, p50, p75, _ = binned(df["y_pred"], df["residual"], bins=bins)
    band(ax, c, p50, p25, p75, color=ORANGE, label="median ± IQR of residual")
    reference_line(ax, 0.0, "zero error")
    ax.set_xlabel("predicted true-OFF score")
    ax.set_ylabel("residual  (predicted − actual)")
    ax.set_title("Residual against prediction level")
    ax.legend(loc="best")
    annotate_n(ax, f"n = {len(df):,}")
    return fig


def plot_residual_vs_time(df: pd.DataFrame, time_col: str = "time_since_diagnosis", bins: int = 20) -> plt.Figure:
    """Residual against disease duration — catches drift the average metric hides."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    c, p25, p50, p75, _ = binned(df[time_col], df["residual"], bins=bins)
    band(ax, c, p50, p25, p75, color=ORANGE, label="median ± IQR of residual")
    reference_line(ax, 0.0, "zero error")
    ax.set_xlabel("time since diagnosis (years)")
    ax.set_ylabel("residual  (predicted − actual)")
    ax.set_title("Residual against disease duration")
    ax.legend(loc="best")
    annotate_n(ax, f"n = {len(df):,}")
    return fig


def plot_calibration(df: pd.DataFrame, bins: int = 20) -> plt.Figure:
    """Predicted vs actual, binned. Systematic curvature here means a biased model."""
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    c, p25, p50, p75, _ = binned(df["y_pred"], df["y_true"], bins=bins)
    lims = [
        float(min(df["y_true"].min(), df["y_pred"].min())),
        float(max(df["y_true"].max(), df["y_pred"].max())),
    ]
    ax.plot(lims, lims, color=INK2, linewidth=1, linestyle="--", label="perfect calibration", zorder=0)
    band(ax, c, p50, p25, p75, color=BLUE, label="observed (median ± IQR)")
    ax.set_xlabel("predicted true-OFF score")
    ax.set_ylabel("actual true-OFF score")
    ax.set_title("Calibration")
    ax.legend(loc="upper left")
    return fig


def plot_error_by_segment(seg: pd.DataFrame, top: int = 24) -> plt.Figure:
    """MSE per segment level, from ``evaluation.metrics.segment_metrics``."""
    d = seg.dropna(subset=["mse"]).copy()
    d["label"] = d["segment"] + " = " + d["level"]
    d = d.sort_values("mse", ascending=False).head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(6.8, max(3.0, 0.28 * len(d) + 1.2)))
    ax.barh(d["label"], d["mse"], color=BLUE, height=0.62, zorder=3)
    ax.set_xlabel("MSE")
    ax.set_title("Error by segment")
    ax.grid(axis="y", visible=False)
    for y, (v, n) in enumerate(zip(d["mse"], d["n"], strict=True)):
        ax.text(v, y, f"  {v:.2f}  (n={n:,})", va="center", fontsize=7.5, color=INK2)
    ax.margins(x=0.22)
    return fig


def plot_error_distribution(df: pd.DataFrame) -> plt.Figure:
    """Where the error actually lives — the tail is what dominates an MSE."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    r = df["residual"].to_numpy()
    lim = float(np.percentile(np.abs(r), 99.5))
    bins = np.linspace(-lim, lim, 61)
    ax.hist(r, bins=bins, color=BLUE, alpha=0.9, zorder=3)
    reference_line(ax, 0.0, "zero error")
    # The x-axis carries residuals in points, so the marker drawn on it has to be in points
    # too. This is the RMS of the residuals — the square root of the headline MSE — and it
    # is labelled as a residual statistic so it cannot be read as the metric itself.
    rms = float(np.sqrt(np.mean(r**2)))
    ax.axvline(rms, color=ORANGE, linewidth=1.4, linestyle="-.", label=f"RMS residual = {rms:.2f} pts")
    ax.axvline(-rms, color=ORANGE, linewidth=1.4, linestyle="-.")
    ax.set_xlabel("residual  (predicted − actual)")
    ax.set_ylabel("visits")
    ax.set_title("Residual distribution")
    ax.legend(loc="upper right")
    annotate_n(ax, f"n = {len(df):,} · MAE = {np.abs(r).mean():.2f}")
    return fig


def plot_patient_trajectories(
    df: pd.DataFrame,
    n: int = 12,
    seed: int = 0,
    time_col: str = "time_since_diagnosis",
) -> plt.Figure:
    """A few whole patients: actual vs predicted across their visits.

    Aggregate metrics can look healthy while individual curves are visibly wrong, so this
    is the check that the *shape* is right and not just the level.
    """
    rng = np.random.default_rng(seed)
    counts = df.groupby("patient_id").size()
    eligible = counts[counts >= 6].index.to_numpy()
    if len(eligible) == 0:
        eligible = counts.index.to_numpy()
    picked = rng.choice(eligible, size=min(n, len(eligible)), replace=False)

    ncol = 4
    nrow = int(np.ceil(len(picked) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    # `axes` is a full nrow x ncol grid while `picked` fills only the first cells, so the
    # truncation is deliberate; `strict=False` states that rather than leaving a reader to
    # wonder whether the lengths were meant to match.
    for ax, pid in zip(axes, picked, strict=False):
        g = df[df["patient_id"] == pid].sort_values(time_col)
        ax.plot(g[time_col], g["y_true"], "o-", color=TRUE_COLOR, markersize=4, linewidth=1.6, label="actual")
        ax.plot(g[time_col], g["y_pred"], "o--", color=PRED_COLOR, markersize=4, linewidth=1.6, label="predicted")
        ax.set_title(f"{pid}  (n={len(g)})", fontsize=9)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.5)

    for ax in axes[len(picked) :]:
        ax.set_visible(False)
    axes[0].legend(loc="best", fontsize=7.5)
    fig.suptitle("Per-patient trajectories: actual vs predicted", fontsize=11, y=1.005)
    fig.supxlabel("time since diagnosis (years)", fontsize=9, color=INK2)
    fig.supylabel("true-OFF MDS-UPDRS III", fontsize=9, color=INK2)
    fig.tight_layout()
    return fig


def plot_shrinkage(df: pd.DataFrame, curve_col: str = "curve_b") -> plt.Figure:
    """Slope against number of visits, showing how hard shrinkage pulls.

    Patients with few visits should collapse onto the population slope; only
    long-followed patients should spread out. If this plot does not fan out, the
    shrinkage is too strong; if it does not collapse on the left, it is too weak.
    """
    if curve_col not in df:
        raise KeyError(f"{curve_col!r} not in frame — run the patient_curve feature first")
    per_patient = df.groupby("patient_id").agg(slope=(curve_col, "first"), n=("y_true", "size"))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(
        per_patient["n"] + np.random.default_rng(0).normal(0, 0.08, len(per_patient)),
        per_patient["slope"],
        s=7,
        color=BLUE,
        alpha=0.25,
        linewidths=0,
        zorder=2,
        label="patients",
    )
    c, p25, p50, p75, _ = binned(pd.Series(per_patient["n"].to_numpy(dtype=float)), per_patient["slope"], bins=9)
    band(ax, c, p50, p25, p75, color=ORANGE, label="median ± IQR")
    reference_line(ax, float(p50.mean()), "population slope")
    ax.set_xlabel("number of visits for that patient")
    ax.set_ylabel("shrunk slope (points/year)")
    ax.set_title("Shrinkage: short trajectories collapse onto the population slope")
    ax.legend(loc="best")
    return fig


FIGURES = {
    "01_residual_vs_pred": plot_residual_vs_pred,
    "02_residual_vs_time": plot_residual_vs_time,
    "03_calibration": plot_calibration,
    "04_error_distribution": plot_error_distribution,
    "05_patient_trajectories": plot_patient_trajectories,
    "06_shrinkage": plot_shrinkage,
}


def run_all(pred_frame: pd.DataFrame, out_dir: _Path, seg: pd.DataFrame | None = None) -> _Path:
    """Write every diagnostic figure plus a markdown index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from .style import apply_style

    apply_style()
    md = ["# Model diagnostics", ""]

    for name, fn in FIGURES.items():
        try:
            if name == "06_shrinkage" and "curve_b" not in pred_frame:
                continue
            fig = fn(pred_frame)
            save(fig, out_dir / f"{name}.png")
            md += [f"### {name}", f"![{name}]({name}.png)", ""]
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never kill a run
            md += [f"### {name}", f"_skipped: {type(exc).__name__}: {exc}_", ""]

    if seg is not None and len(seg):
        fig = plot_error_by_segment(seg)
        save(fig, out_dir / "07_error_by_segment.png")
        md += ["### 07_error_by_segment", "![07_error_by_segment](07_error_by_segment.png)", ""]

    (out_dir / "diagnostics.md").write_text("\n".join(md), encoding="utf-8")
    return out_dir
