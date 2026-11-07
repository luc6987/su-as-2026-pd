"""The house style, as an explicit opt-in.

Inherited from the original EDA module, with one deliberate change: ``import eda`` used to
mutate ``plt.rcParams`` as a side effect, so merely importing it restyled every figure in
the process. Here the palette is data and :func:`apply_style` does the mutating, so a
caller opts in.

The palette is validated rather than eyeballed — it passes the lightness-band, chroma,
CVD-separation, normal-vision and contrast checks on the light surface. Three of the six
hues sit below 3:1 contrast against the surface, which obliges visible labelling: every
figure here carries a legend or direct labels, and every run also writes its numbers as
CSV, so colour is never the only channel carrying identity.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

# Deliberately no module-level `matplotlib.use(...)` or `import matplotlib.pyplot`.
# Selecting a backend is process-global state, and a library module must not choose one
# for its importer — `apply_style(headless=True)` does it, explicitly, when asked.

# Fixed categorical order. Assign in order; never cycle, never reorder by rank — a
# series keeps its colour when a filter changes how many series are on the chart.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

INK = "#0b0b0b"  # primary text / reference lines
INK2 = "#52514e"  # secondary text
GRID = "#e6e5e1"  # gridlines and axes
SURF = "#fcfcfb"  # the chart surface everything is validated against

BLUE, ORANGE, GREEN, YELLOW, PINK, VIOLET = PALETTE

# Semantic assignments, so figures agree with each other across the report.
TRUE_COLOR = INK  # ground truth is ink, not a series colour
PRED_COLOR = BLUE  # model predictions
REF_COLOR = INK2  # naive reference lines

_APPLIED = False


def use_headless_backend() -> None:
    """Select a non-interactive backend. Explicit, and safe to call repeatedly.

    Entry points that only write files to disk should call this before importing
    ``pyplot`` anywhere.
    """
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")


def apply_style(headless: bool = True, force: bool = False) -> None:
    """Apply the house rcParams. Idempotent, and never runs on import.

    ``headless`` also selects the Agg backend, because in practice every caller of this
    function is writing figures to files. Pass ``False`` for an interactive session.
    """
    global _APPLIED
    if _APPLIED and not force:
        return
    if headless:
        use_headless_backend()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURF,
            "axes.facecolor": SURF,
            "savefig.facecolor": SURF,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "figure.dpi": 110,
        }
    )
    _APPLIED = True


def save(fig: Figure, path: Any) -> None:
    """Write a figure and close it, so long report runs do not leak handles."""
    import matplotlib.pyplot as plt

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
