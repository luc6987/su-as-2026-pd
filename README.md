# Predicting the "true OFF" MDS-UPDRS Part III score

Code for our entry to [ENS Challenge Data #159](https://challengedata.ens.fr/participants/challenges/159/)
(Paris Brain Institute), written for the SU-AS 2026 course at Sorbonne Université.

For every visit of a Parkinson's patient, the challenge asks for the latent **"true OFF" motor
score** (MDS-UPDRS Part III, 0–132): the score the patient would show after a complete drug
washout. The data never show it directly, only a score on medication (ON), a score after an
incomplete washout (OFF), or both, and each is a noisy, multiplicative view of the target. The
metric is the mean squared error.

| final configuration (v11) | MSE |
|---|---|
| public leaderboard | 4.1202 |
| private leaderboard | 6.0332 |

## Running the notebook

`notebook/pd_challenge.ipynb` is the entry point. It trains the final model on all training
patients and writes the submission file, and explains each step along the way. The copy here is
already executed, so its outputs can be read without running anything.

1. **Get the data.** Download `X_train_6ZIKlTY.csv`, `y_train_lXj6X5y.csv` and `X_test_oiZ2ukx.csv`
   from the challenge page (it needs an account). The data are not redistributed here.
2. **Get a Python 3.12 environment with Jupyter**, ideally a fresh one, because the notebook
   installs its pinned library versions into whichever Python runs it:

   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install jupyterlab
   ```

3. **Run it**, in either of two ways:
   - *Notebook alone.* Put `pd_challenge.ipynb` in a folder, the three CSVs in a `data/`
     subfolder next to it, open it and run all cells. The first cell downloads the `v11` release
     of this repository next to the notebook, and the second installs the pinned versions.
   - *From a clone.* `git clone https://github.com/luc6987/su-as-2026-pd.git`, put the CSVs in
     `su-as-2026-pd/data/`, then open `notebook/pd_challenge.ipynb` and run all cells. The
     notebook finds the clone around it and downloads nothing.

A full run takes about half an hour on an 8-core laptop CPU. `PD_NB_SMOKE=1` runs a one-minute
check on a subsample instead; the notebook's section 0 lists the other switches.

To set the environment up yourself instead, `pip install -r requirements.txt` or
`conda env create -f environment.yml`; the notebook then finds every pinned version and installs
nothing.

## Reproducibility

On the machine that wrote our upload (Apple silicon, macOS, the versions in `requirements.txt`),
the file the notebook writes is the uploaded one byte for byte, and the notebook checks its SHA-256.
On another platform the boosters and the GRU networks can differ in the last digits; the notebook
then compares summary statistics of the predictions with those of the uploaded file instead.

On macOS, if Python stops with `OMP: Error #15` (two copies of the OpenMP runtime, one from
PyTorch and one from XGBoost or LightGBM), restart the kernel with `KMP_DUPLICATE_LIB_OK=TRUE`
set in the environment, or point PyTorch's `torch/lib/libomp.dylib` at the other copy (what we did).

## What is in the repository

| path | what it is |
|---|---|
| `notebook/pd_challenge.ipynb` | the notebook, executed, with its outputs |
| `notebook/pd_challenge.py` | its source, in the jupytext *percent* format; `python tools/notebook/build.py notebook/pd_challenge.py notebook/pd_challenge.ipynb` rebuilds the notebook |
| `pdchallenge/` | the package the notebook drives: data loading (`data/`), the feature chain (`features/`), the models (`models/`), evaluation (`evaluation/`) |
| `configs/consolidated_v11.yaml` | the final configuration |
| `requirements.txt`, `environment.yml` | the pinned environment, for pip or conda |
| `pyproject.toml` | package metadata (`pip install -e .` makes `pdchallenge` importable from anywhere) |
