"""Column names, file names and missing-data semantics.

Everything that names a column lives here, so the raw CSVs are described in exactly
one place. See ``reports/literature/parkinson_factors.md`` for the evidence behind the
missing-data rules.
"""
from __future__ import annotations

# --- input files -------------------------------------------------------------------
# ``y_random`` is the organiser's random-submission example. It is never used for
# training; it exists so we can confirm the submission format and index alignment.
FILES = {
    "X_train": "X_train_6ZIKlTY.csv",
    "y_train": "y_train_lXj6X5y.csv",
    "X_test": "X_test_oiZ2ukx.csv",
    "y_random": "y_pred_random_obUqjcJ.csv",
}

INDEX = "Index"
PATIENT = "patient_id"
TARGET = "target"  # the "true OFF" score, 0-132

# --- raw columns -------------------------------------------------------------------
ID_COLS = [INDEX, PATIENT, "cohort"]
CAT_COLS = ["cohort", "sexM", "gene"]
NUM_COLS = [
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

# --- derived columns (see data.loading.add_derived) --------------------------------
TIME_SINCE_DIAGNOSIS = "time_since_diagnosis"  # age - age_at_diagnosis, in years
VISIT_IDX = "visit_idx"
N_VISITS = "n_visits"
AGE_FIRST_VISIT = "age_first_visit"
YEARS_IN_STUDY = "years_in_study"
HAS_ON = "has_on"
HAS_OFF = "has_off"

DERIVED_COLS = [
    TIME_SINCE_DIAGNOSIS,
    VISIT_IDX,
    N_VISITS,
    AGE_FIRST_VISIT,
    YEARS_IN_STUDY,
    HAS_ON,
    HAS_OFF,
]

SCORE_COLS = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off", "gene"]

# --- missing-data semantics --------------------------------------------------------
# These are documented because every one of them is easy to get wrong.
#
# * Every visit carries at least one of ``on`` / ``off`` — never neither.
# * ``ledd`` is 37% missing and its missingness is strongly tied to visit order
#   (80% at visit 0 falling to ~12% by visit 11): early, untreated visits. Missingness
#   is therefore informative, so it gets an explicit indicator column rather than being
#   imputed away.
# * ``gene`` is 32% missing and is very likely missing-not-at-random (carrier rates
#   swing from 1% in sporadic PD to >40% in some cohorts). It is encoded as four
#   one-hot levels plus a separate ``gene_missing`` indicator — never imputed to
#   "No Mutation".
# * ``time_since_intake_off`` is 79% missing, so it is a weak feature at best.
INFORMATIVE_MISSING = ["ledd", "gene"]
