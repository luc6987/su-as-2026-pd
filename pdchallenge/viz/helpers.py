"""Shared plotting primitives.

These were private helpers in the original EDA module; they are generic — bin a scattered
relationship and draw its central tendency with an uncertainty ribbon — so they are
promoted here and used by every diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import INK2, ORANGE


def binned(
    x: pd.Series,
    y: pd.Series,
    bins: int = 20,
    q: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and IQR of ``y`` within bins of ``x``.

    Quantile bins by default, so every bin holds the same number of points and the tails
    do not turn into single-observation noise.

    Returns ``(centers, p25, p50, p75, n)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < bins * 2:
        raise ValueError(f"too few points ({len(x)}) for {bins} bins")

    edges = (
        np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        if q
        else np.linspace(x.min(), x.max(), bins + 1)
    )
    if len(edges) < 3:
        raise ValueError("x has no spread to bin on")

    centers, p25, p50, p75, counts = [], [], [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = i == len(edges) - 2
        sel = (x >= lo) & ((x <= hi) if last else (x < hi))
        if sel.sum() < 5:
            continue
        centers.append(float(np.median(x[sel])))
        p25.append(float(np.percentile(y[sel], 25)))
        p50.append(float(np.median(y[sel])))
        p75.append(float(np.percentile(y[sel], 75)))
        counts.append(int(sel.sum()))

    return (np.array(centers), np.array(p25), np.array(p50), np.array(p75), np.array(counts))


def band(ax, x, y, lo, hi, color=ORANGE, label: str | None = None, alpha: float = 0.22):
    """Draw a median line with an IQR ribbon around it."""
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, y, color=color, label=label, linewidth=2)
    return ax


def reference_line(ax, y: float = 0.0, label: str | None = None):
    """A recessive horizontal reference (zero error, identity, …)."""
    ax.axhline(y, color=INK2, linewidth=1, linestyle="--", alpha=0.5, label=label, zorder=0)
    return ax


def annotate_n(ax, text: str):
    """Put a small muted annotation in the corner."""
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color=INK2,
    )
    return ax
