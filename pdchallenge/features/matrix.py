"""Turning a transformed visit frame into a numeric design matrix.

Two encoding decisions come straight from the literature review and are easy to get
wrong by defaulting to the obvious:

* ``gene`` is **not** imputed. It is 32% missing and almost certainly missing-not-at-random
  (G2019S carrier rates run from ~1% in sporadic PD to >40% in some cohorts), so the
  four observed levels get one-hot columns **plus** a separate ``gene_missing`` indicator.
  Filling it with "No Mutation" would systematically mislabel high-carrier populations.
* ``ledd`` missingness is informative — it marks the early, still-untreated visits (80%
  missing at visit 0, ~12% by visit 11) — so it also gets its own indicator.

NaNs are left in place rather than filled here: the linear model imputes+scales inside its
own pipeline, and tree models handle NaN natively. Filling centrally would force one
choice on both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENE_COL = "gene"
GENE_LEVELS = ["GBA+", "LRRK2+", "No Mutation", "OTHER+"]
GENE_MISSING = "gene_missing"
LEDD_MISSING = "ledd_missing"
COHORT_COL = "cohort"

# Columns that must never enter the design matrix: identifiers, the target, and the
# per-patient *observed* offset.
#
# That last one — ``ldr.LDR_OBS``, spelled out here to keep this module free of the import
# — is derived from the training target and exists only on the training side. Nothing
# writes it today, which is exactly why it is worth naming: a column that is present in
# training and absent at test time is a leak, and it would arrive silently, since
# `numeric` below sweeps up every numeric column that is not listed here.
NON_FEATURE = {
    "Index",
    "patient_id",
    "target",
    GENE_COL,
    COHORT_COL,
    "debias_source",
    "ldr_obs",
}

# Note the raw score columns (`on`, `off`) are kept and are *biased* — their de-biased
# counterparts (`pseudo_target`, `curve_*`) carry the signal, and a tree can use both. There
# is deliberately no name list here for them: everything numeric that is not in
# `NON_FEATURE` is passed through, so an allow-list would go stale the first time a feature
# step added a column. (An earlier `RAW_NUMERIC` constant was declared and never read.)
#
# --- one array, two names (measured, left alone) ------------------------------------
# `debias` writes the per-patient long-duration offset twice: as `bias_off_hat` (the OFF
# branch of the correction, paired with `bias_on_hat`) and as `ldr_hat` (the same quantity
# under the name its own module gives it). They are the same array, so the matrix carries
# the slow term in two identical columns and `colsample_bytree=0.8` samples it twice as
# often as any other feature (docs/audit-2026-09-16.md §2 P1-7).
#
# Dropping one was implemented and **measured worse**: 4.280684 → 4.282093 on the main CV
# (between 3.3536 → 3.3520, within 2.6604 → 2.6640), 2026-09-18. Small, but this project
# does not adopt a change that measures worse — the same rule that kept `quadratic` and
# `ldr_kind="gbm"` out. So the duplication stays, deliberately, and the pair is pinned by
# `tests/test_features_matrix.py` so it reads as a measured choice rather than an accident.


def build_matrix(
    df: pd.DataFrame,
    include: list[str] | None = None,
    drop: list[str] | None = None,
) -> pd.DataFrame:
    """Return a numeric design matrix with categorical encodings applied.

    ``include`` restricts to a column list (after encoding); ``drop`` removes columns.
    """
    out = pd.DataFrame(index=df.index)

    # categoricals -> indicators
    gene = df[GENE_COL] if GENE_COL in df else pd.Series(np.nan, index=df.index, dtype=object)
    for level in GENE_LEVELS:
        out[f"gene_{level.replace(' ', '_').replace('+', 'pos')}"] = (gene == level).astype(int)
    out[GENE_MISSING] = gene.isna().astype(int)

    if COHORT_COL in df:
        out["cohort_B"] = (df[COHORT_COL] == "B").astype(int)

    # numeric passthrough, everything not explicitly excluded
    numeric = [
        c
        for c in df.columns
        if c not in NON_FEATURE
        and c not in out.columns
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in numeric:
        out[c] = df[c].astype(float)

    if "ledd" in df:
        out[LEDD_MISSING] = df["ledd"].isna().astype(int)

    if include is not None:
        missing = [c for c in include if c not in out.columns]
        if missing:
            raise KeyError(f"requested columns not in matrix: {missing}")
        out = out[include]
    if drop:
        out = out.drop(columns=[c for c in drop if c in out.columns])

    return out


def feature_names(df: pd.DataFrame) -> list[str]:
    """The default model input list."""
    return list(build_matrix(df).columns)
