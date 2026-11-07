"""Figures: the house style, plotting primitives, diagnostics and attribution."""

from .helpers import annotate_n, band, binned, reference_line
from .style import PALETTE, apply_style, save

__all__ = [
    "apply_style",
    "save",
    "PALETTE",
    "binned",
    "band",
    "reference_line",
    "annotate_n",
]
