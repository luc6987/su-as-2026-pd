"""Metrics, grouped cross-validation, and comparison against naive baselines."""

from .comparison import compare, compare_on_common_rows, naive_predictions
from .cv import CVResult, cross_validate, fit_full
from .forward import ForwardResult, forward_validate
from .metrics import (
    bias,
    clip_to_valid,
    decompose_mse,
    mae,
    mse,
    paired_patient_bootstrap,
    r2,
    segment_metrics,
    summarize,
)
from .paired import (
    SMALL_EFFECT,
    paired_bootstrap,
    pooled_paired_bootstrap,
    verdict,
)

__all__ = [
    "paired_bootstrap",
    "pooled_paired_bootstrap",
    "verdict",
    "SMALL_EFFECT",
    "mse",
    "mae",
    "bias",
    "r2",
    "summarize",
    "segment_metrics",
    "clip_to_valid",
    "decompose_mse",
    "paired_patient_bootstrap",
    "cross_validate",
    "fit_full",
    "CVResult",
    "forward_validate",
    "ForwardResult",
    "compare",
    "compare_on_common_rows",
    "naive_predictions",
]
