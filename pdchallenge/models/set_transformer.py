"""A patient-as-a-set transformer over the chain's per-visit features.

Every visit of a patient is one token carrying the chain's numeric columns; a transformer encoder lets
each visit attend to the patient's other visits, and a per-visit head predicts the residual of the
target over the chain's own posterior curve (``lik_pred``). On its own it is clearly worse than the
boosters (dev ≈ 6.0 MSE against v6 bag5 5.32 with two reps), but its errors are only 0.83 correlated
with theirs, and a fixed 0.2 / 0.8 blend with v6's scale-averaged bag5 is −0.19 MSE on dev, three CV
seeds pooled, with the weight chosen on the mining patients only
(`reports/paradigm-2026-10-08/README.md` §5). The point is a different inductive bias, not a better
model: the boosters see the patient through hand-built neighbour columns, this sees the visits as a set.

Design choices carried over unchanged from the probe that measured it (`nn_set.py`, `nnlib.py`):

* the per-neighbour ``*_prev*`` / ``*_next*`` / ``*_nbrmean*`` columns are dropped (attention sees the
  neighbours itself), every other numeric column is mapped through its training-fold empirical CDF to
  a normal score, NaN becomes 0 plus a missing indicator for columns missing on 1–99% of rows;
* 3 encoder layers, width 128, 4 heads, dropout 0.1, residual head scaled by 10 points;
* AdamW (lr 1e-3, weight decay 1e-4), one-cycle schedule, 40 epochs, 64 patients per batch;
* rep ``r`` is seeded ``1000 r + 17`` (+ ``7919 × seed``), so ``seed=0`` reproduces the probe's reps.

torch is imported inside :meth:`fit` / :meth:`predict` only: the package and its test suite do not
need it unless a config actually uses this model (run those configs from the ``pdnn`` environment).

Two options, both off by default (v7's member is unchanged), from the 2026-10-09 round
(`reports/setnn-2026-10-09/README.md`): ``arch: gru`` reads the same tokens as an age-ordered sequence
with a bidirectional GRU (registered on its own as ``visit_gru``) — better alone and a better blend member
than the attention model; ``level_norm`` divides the score columns by the patient's own level
(`levelnorm.py`) and scales the residual head with it.

A third, also off by default: ``pretrain="mvm"`` (2026-10-10 round F(b),
`reports/round-2026-10-10/F/README.md`) pre-trains the GRU encoder on a *masked-visit* task before the
usual fine-tune — for every patient with ≥ 2 valid visits one visit is hidden down to its time and
covariates and the network predicts that visit's ``log1p(off)`` / ``log1p(on)`` from the other visits.
The pool is exactly the rows this ``fit`` receives (the training fold, probe ``--pool train``; no test
row, no holdout and no target — the pre-training targets are the frame's *observation* columns), so the
model protocol does not change and the fine-tune that follows sees the fold's target as ever. The member
that carried this into v9/v10 is `reports/round-2026-10-14/M/`'s ``agru_premvm_pooltrain_pe60`` (D115,
D128, D130): the encoder starts from the observations instead of from scratch, the head is trained from
scratch on the fold's target.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Self

import numpy as np
import pandas as pd

from ..data.schema import PATIENT
from ..evaluation.metrics import clip_to_valid
from ..features.matrix import build_matrix
from .base import BaseModel
from .levelnorm import normalise, patient_scale
from .registry import register

NEIGHBOUR_TAGS = ("_prev", "_next", "_nbrmean")
MAX_VISITS = 12

# The masked-visit pre-training schedule (2026-10-10 round F(b)), written down rather than tuned: a
# 64-patient batch of visits, the fine-tune's optimiser settings. Named here as in the probe
# (`nn3.PRE_BS` / `PRE_LR` / `PRE_WD`) so the two cannot drift apart.
PRE_BS, PRE_LR, PRE_WD = 64, 1e-3, 1e-4
# What a masked visit still shows (`nn3.KEEP_NAMES`): its time axis and everything known without the
# visit's own scores. The chain columns this visit's scores went into are hidden with it; the
# indicators of these columns stay — "there is no LEDD here" is itself information.
KEEP_NAMES = ("age", "age_at_diagnosis", "sexM", "cohort_B", "gene_GBApos", "gene_LRRK2pos",
              "gene_No_Mutation", "gene_OTHERpos", "gene_missing", "ledd", "ledd_missing",
              "time_since_intake_on", "time_since_intake_off", "time_since_diagnosis", "visit_idx",
              "n_visits", "age_first_visit", "years_in_study", "has_on", "has_off")


class _RankNormaliser:
    """Training-fold empirical CDF -> standard normal score per column, NaN -> 0 + indicator."""

    def fit(self, M: pd.DataFrame) -> "_RankNormaliser":
        A = M.to_numpy(dtype=np.float64)
        self.cols = list(M.columns)
        miss = np.isnan(A).mean(axis=0)
        self.ind = np.flatnonzero((miss > 0.01) & (miss < 0.99))
        self.q = []
        for j in range(A.shape[1]):
            v = A[:, j][np.isfinite(A[:, j])]
            self.q.append(np.unique(np.quantile(v, np.linspace(0, 1, 201))) if len(v) else np.array([0.0]))
        return self

    def transform(self, M: pd.DataFrame) -> np.ndarray:
        from scipy.stats import norm

        A = M[self.cols].to_numpy(dtype=np.float64)
        nan = np.isnan(A)
        Z = np.zeros_like(A)
        for j, q in enumerate(self.q):
            if len(q) < 2:
                continue
            u = np.interp(A[:, j], q, np.linspace(0, 1, len(q)))
            Z[:, j] = norm.ppf(np.clip(u, 0.005, 0.995))
        Z[nan] = 0.0
        return np.concatenate([Z, nan[:, self.ind].astype(np.float64)], axis=1).astype(np.float32)


def _pack(frame: pd.DataFrame, Z: np.ndarray, y: np.ndarray | None, base: np.ndarray,
          rs: np.ndarray | None = None,
          extra: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Rows -> (patients × visits) arrays, visits ordered by age; ``row`` maps back to frame rows.

    ``rs`` (optional, per row) is the residual head's scale in points; packed as ``rs`` when given.
    ``extra`` (optional) packs further per-row arrays the same way — the pre-training scores and
    their availability masks — into (P, V) float32 blocks padded with 0, as the probe's `pack` does."""
    pid = frame[PATIENT].to_numpy()
    age = frame["age"].to_numpy(dtype=float)
    order = np.lexsort((age, pd.factorize(pid, sort=False)[0]))
    codes = pd.factorize(pid[order], sort=False)[0]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(order)]
    P, F = len(starts), Z.shape[1]
    x = np.zeros((P, MAX_VISITS, F), np.float32)
    mask = np.ones((P, MAX_VISITS), bool)
    row = np.full((P, MAX_VISITS), -1, np.int64)
    yy = np.zeros((P, MAX_VISITS), np.float32)
    bb = np.zeros((P, MAX_VISITS), np.float32)
    rr = np.zeros((P, MAX_VISITS), np.float32) if rs is not None else None
    ex = {k: np.zeros((P, MAX_VISITS), np.float32) for k in (extra or {})}
    for i, (s, e) in enumerate(zip(starts, ends)):
        idx = order[s:e]
        n = len(idx)
        if n > MAX_VISITS:
            raise ValueError(f"a patient has {n} visits; the set model is sized for {MAX_VISITS}")
        x[i, :n] = Z[idx]
        mask[i, :n] = False
        row[i, :n] = idx
        bb[i, :n] = base[idx]
        if rr is not None:
            rr[i, :n] = rs[idx]
        if y is not None:
            yy[i, :n] = y[idx]
        for k, v in (extra or {}).items():
            ex[k][i, :n] = v[idx]
    out = {"x": x, "mask": mask, "row": row, "y": yy, "base": bb}
    if rr is not None:
        out["rs"] = rr
    out.update(ex)
    return out


def _base_curve(frame: pd.DataFrame) -> np.ndarray:
    b = frame["lik_pred"].to_numpy(dtype=float) if "lik_pred" in frame else np.full(len(frame), np.nan)
    alt = frame["env_pred"].to_numpy(dtype=float) if "env_pred" in frame else np.full(len(frame), np.nan)
    b = np.where(np.isfinite(b), b, alt)
    fill = np.nanmean(b) if np.isfinite(b).any() else 0.0
    return np.where(np.isfinite(b), b, fill)


@contextmanager
def _torch_threads(n: int) -> Iterator[None]:
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def _build_net(n_in: int, d: int, layers: int, heads: int, drop: float, arch: str = "set") -> Any:
    import torch.nn as nn

    if arch == "gru":
        return _build_gru(n_in, d, layers, drop)
    if arch != "set":
        raise ValueError(f"unknown set-model arch {arch!r}; use 'set' or 'gru'")

    class SetNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True,
                                               activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        def forward(self, x, mask):  # noqa: ANN001, ANN202 - torch tensors
            return self.out(self.enc(self.inp(x), src_key_padding_mask=mask)).squeeze(-1)

    return SetNet()


def _build_gru(n_in: int, d: int, layers: int, drop: float) -> Any:
    """Bidirectional GRU over the age-ordered visits (packed, so padding is never read).

    Module order (input MLP, GRU, head) is the probe's (`reports/setnn-2026-10-09/nn2.py`), so the same
    seed gives the same initial weights."""
    import torch.nn as nn

    class GRUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
            self.rnn = nn.GRU(d, d // 2, num_layers=layers, batch_first=True, bidirectional=True,
                              dropout=drop if layers > 1 else 0.0)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        def forward(self, x, mask):  # noqa: ANN001, ANN202 - torch tensors
            h = self.inp(x)
            lengths = (~mask).sum(1).clamp(min=1).cpu()
            pk = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            o, _ = self.rnn(pk)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
            return self.out(h + o).squeeze(-1)

    return GRUNet()


# --- masked-visit pre-training (``pretrain='mvm'``) -------------------------------------
#
# Ported from the probe that defines it, `reports/round-2026-10-10/F/nn3.py` (kept verbatim in
# `reports/round-2026-10-14/M/nn3.py`): same keep vector, same score targets, same masking, same
# PreNet, same two seeds (`rs` for the pre-training, `rs + 1` for the fine-tune — seed 0 is the
# probe's `1000·rep + 17` / `+ 18`), same schedule. Only the pool differs: the probe can add the
# validation fold and the test frame (`--pool devtest`); the package's pool is the rows ``fit``
# receives, which is the probe's ``--pool train`` — the one that changes no model protocol.


def _build_pre(n_in: int, d: int, layers: int, drop: float) -> Any:
    """The masked-visit net: `GRUNet`'s input MLP and GRU, with a two-number head instead of a residual.

    ``inp`` and ``rnn`` are `_build_gru`'s construction written out again with the same shapes and
    the same order, so ``torch.manual_seed(rs)`` gives the same initial weights as the probe's
    `PreNet`; the head reads ``z = h + o`` at the masked visit and predicts its two ``log1p`` scores.
    """
    import torch.nn as nn

    class PreNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
            self.rnn = nn.GRU(d, d // 2, num_layers=layers, batch_first=True, bidirectional=True,
                              dropout=drop if layers > 1 else 0.0)
            self.head = nn.Linear(d, 2)

        def forward(self, x, mask):  # noqa: ANN001, ANN202 - torch tensors
            h = self.inp(x)
            lengths = (~mask).sum(1).clamp(min=1).cpu()
            pk = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            o, _ = self.rnn(pk)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
            return h + o

    return PreNet()


def _keep_vector(cols: list[str], ind: np.ndarray) -> np.ndarray:
    """Which input columns survive on a masked visit, as a boolean vector over the network's inputs.

    The `KEEP_NAMES` columns of the design matrix, then their missing indicators (indicator j belongs
    to column ``cols[ind[j]]`` in `_RankNormaliser.transform`'s layout), then ``False`` for the mask
    indicator itself — the one input the masked visit sets. The probe's `keep_vector` on the rank
    normaliser, which is the only normaliser this model has.
    """
    hit = np.array([c in set(KEEP_NAMES) for c in cols], bool)
    return np.r_[np.concatenate([hit, hit[ind]]), False]


def _score_targets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-visit pre-training targets and their availability masks, from the frame's **observation**
    columns ``off`` / ``on`` — never from the target, which the test patients do not have.

    ``t = log1p(score)`` where the score was recorded and 0 where it was not: the missing entries
    carry weight 0, and a NaN under a 0 weight would still poison the sum (the probe's `score_targets`).
    """
    out: dict[str, np.ndarray] = {}
    for name in ("off", "on"):
        v = frame[name].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        out[f"t_{name}"] = np.log1p(np.where(ok, v, 0.0)).astype(np.float32)
        out[f"m_{name}"] = ok.astype(np.float32)
    return out


def _mask_tokens(x: Any, m: Any, keep: np.ndarray, rng: np.random.Generator) -> tuple[Any, np.ndarray]:
    """Hide one valid visit per patient: keep its time and covariates only, flag it as masked.

    ``x`` (P, V, F) float tensor, ``m`` (P, V) bool (True = padding), ``keep`` (F,) bool — the columns
    a masked visit still shows — and ``rng`` a ``numpy.random.Generator``. Returns a **new** tensor
    where, for every patient with ≥ 2 valid visits, one uniformly drawn slot j carries ``x[:, j] *
    keep`` with the last (indicator) column set to 1, plus ``pos`` (P,) int64: the masked slot, or −1
    for the patients left alone (fewer than 2 valid visits) — those stay out of the pre-training loss.
    ``x`` itself is never written through. The probe's `flib.mask_tokens`, unchanged.

    One uniform draw per patient over ``0 … n_valid−1`` is a uniform draw over the valid positions:
    `_pack` leaves the valid visits as a prefix of every row."""
    import torch

    keep = np.asarray(keep, dtype=bool)
    n_pat, _, f = x.shape
    assert keep.shape[0] == f, f"keep has {keep.shape[0]} entries, the inputs have {f} columns"
    n_valid = (~m).sum(1).cpu().numpy()
    elig = n_valid >= 2
    pos = np.full(n_pat, -1, np.int64)
    if elig.any():
        pos[elig] = np.floor(rng.random(int(elig.sum())) * n_valid[elig]).astype(np.int64)
    xm = x.clone()
    rows = np.flatnonzero(pos >= 0)
    if len(rows):
        rt = torch.from_numpy(rows)
        pt = torch.from_numpy(pos[rows])
        xm[rt, pt] = x[rt, pt] * torch.as_tensor(keep)
        xm[rt, pt, -1] = 1.0
    return xm, pos


@register
class SetTransformerModel(BaseModel):
    """Transformer over each patient's visits; predicts every visit jointly."""

    name = "set_transformer"
    RESIDUAL_SCALE = 10.0

    def __init__(
        self,
        reps: int = 2,
        epochs: int = 40,
        d: int = 128,
        layers: int = 3,
        heads: int = 4,
        drop: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        seed: int = 0,
        threads: int = 4,
        clip: bool = True,
        arch: str = "set",
        level_norm: bool = False,
        neighbour_cols: bool = False,
        pretrain: str = "none",
        pre_epochs: int = 20,
    ) -> None:
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}")
        if arch not in ("set", "gru"):
            raise ValueError(f"unknown set-model arch {arch!r}; use 'set' or 'gru'")
        if pretrain not in ("none", "mvm"):
            raise ValueError(f"unknown pretrain mode {pretrain!r}; use 'none' or 'mvm'")
        if pretrain == "mvm" and arch != "gru":
            # The pre-training hands its encoder to the fine-tune, and the encoder is the GRU's;
            # the attention model has no `rnn` to initialise (the probe refuses the mirror case).
            raise ValueError(f"pretrain='mvm' pre-trains the GRU encoder; it needs arch='gru', got {arch!r}")
        if pre_epochs < 1:
            raise ValueError(f"pre_epochs must be >= 1, got {pre_epochs}")
        self.reps = int(reps)
        self.epochs = int(epochs)
        self.d, self.layers, self.heads, self.drop = int(d), int(layers), int(heads), float(drop)
        self.lr, self.weight_decay, self.batch_size = float(lr), float(weight_decay), int(batch_size)
        self.seed = int(seed)
        self.threads = int(threads)
        self.clip = clip
        # arch 'gru' reads the same visit tokens as an age-ordered sequence; level_norm expresses the
        # score columns as fractions of the patient's level (`levelnorm.py`) and scales the residual head
        # with it (0.3·s_p points instead of 10). Both off by default: v7's member is unchanged.
        self.arch = arch
        self.level_norm = bool(level_norm)
        # neighbour_cols keeps the chain's per-neighbour prev/next/nbrmean columns, which the base design
        # drops because attention sees the neighbours itself; kept, the model is a little different again
        self.neighbour_cols = bool(neighbour_cols)
        # pretrain 'mvm' pre-trains the encoder on the masked-visit task before the fine-tune; off by
        # default, so v7 / v8 / v9's members are unchanged bit for bit.
        self.pretrain = pretrain
        self.pre_epochs = int(pre_epochs)

    # -- design -----------------------------------------------------------------------
    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        M = build_matrix(X)
        if not self.neighbour_cols:
            M = M[[c for c in M.columns if not any(t in c for t in NEIGHBOUR_TAGS)]]
        return normalise(M, patient_scale(X)) if self.level_norm else M

    def _input_columns(self, Z: np.ndarray) -> np.ndarray:
        """``Z`` with the mask-indicator column appended when the pre-training uses one.

        The column is all zeros on the fine-tune's and the prediction's inputs — only the masked-visit
        pool sets it — so the network's width is ``n_in + 1`` in this mode and unchanged otherwise
        (the probe's `add_mask_col`)."""
        if self.pretrain != "mvm":
            return Z
        return np.concatenate([Z, np.zeros((len(Z), 1), np.float32)], axis=1)

    def _residual_scale(self, X: pd.DataFrame) -> np.ndarray | None:
        return (0.3 * patient_scale(X)).astype(np.float32) if self.level_norm else None

    # -- fitting ----------------------------------------------------------------------
    def _pretrain(self, pool: dict[str, np.ndarray], keep: np.ndarray, rs: int) -> Any:
        """Pre-train the encoder on the masked-visit pool; returns the fitted `_build_pre` net.

        One pass over the pool per epoch with the mask re-drawn each time (an independent
        ``default_rng`` stream, so the batch permutations keep the fine-tune's sequence); the loss is
        the mean squared error on the scores that exist, each score counting once. Seeding is ``rs``
        for all of torch, the permutations and the mask — exactly the probe's `train_pre`.
        """
        import torch
        import torch.nn as nn

        torch.manual_seed(rs)
        enc = _build_pre(self.n_in_, self.d, self.layers, self.drop)
        mask_rng = np.random.default_rng(rs)
        perm_rng = np.random.RandomState(rs)
        x = torch.from_numpy(pool["x"])
        m = torch.from_numpy(pool["mask"])
        t_off, t_on = torch.from_numpy(pool["t_off"]), torch.from_numpy(pool["t_on"])
        w_off, w_on = torch.from_numpy(pool["m_off"]), torch.from_numpy(pool["m_on"])
        n = len(x)
        steps = int(np.ceil(n / PRE_BS))
        opt = torch.optim.AdamW(enc.parameters(), lr=PRE_LR, weight_decay=PRE_WD)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=PRE_LR, total_steps=steps * self.pre_epochs,
                                                    pct_start=0.1)
        for _ in range(self.pre_epochs):
            enc.train()
            xm, pos = _mask_tokens(x, m, keep, mask_rng)
            perm = perm_rng.permutation(n)
            for s in range(steps):
                ib = perm[s * PRE_BS:(s + 1) * PRE_BS]
                bi = torch.from_numpy(ib)
                loc = np.flatnonzero(pos[ib] >= 0)
                if not len(loc):
                    continue
                # the masked slots of this batch: local positions index the network output, global
                # ones index the targets (the two differ by the batch's start row)
                li = torch.from_numpy(loc)
                gi = torch.from_numpy(ib[loc])
                pt = torch.from_numpy(pos[ib][loc])
                pred = enc.head(enc(xm[bi], m[bi])[li, pt])
                wo, won = w_off[gi, pt], w_on[gi, pt]
                denom = wo.sum() + won.sum()
                if float(denom) == 0.0:
                    continue
                loss = (((pred[:, 0] - t_off[gi, pt]) ** 2 * wo).sum()
                        + ((pred[:, 1] - t_on[gi, pt]) ** 2 * won).sum()) / denom
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
                sched.step()
        return enc

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> Self:
        import torch
        import torch.nn as nn

        M = self._matrix(X)
        self.norm_ = _RankNormaliser().fit(M)
        Z = self._input_columns(self.norm_.transform(M))
        self.n_in_ = Z.shape[1]
        if self.pretrain == "mvm":
            # The pool is exactly the rows `fit` received — the probe's `--pool train` — packed once
            # and reused by every rep: same matrix and normaliser as the fine-tune inputs, plus the
            # mask-indicator column and the masked visit's `log1p(off)` / `log1p(on)` targets.
            keep = _keep_vector(list(M.columns), self.norm_.ind)
            pool = _pack(X, Z, None, _base_curve(X), extra=_score_targets(X))
        P = _pack(X, Z, np.asarray(y, dtype=np.float32), _base_curve(X), self._residual_scale(X))
        x, m = torch.from_numpy(P["x"]), torch.from_numpy(P["mask"])
        yt, b = torch.from_numpy(P["y"]), torch.from_numpy(P["base"])
        rsc = torch.from_numpy(P["rs"]) if "rs" in P else None
        self.states_ = []
        with _torch_threads(self.threads):
            for rep in range(self.reps):
                rs = 1000 * rep + 17 + 7919 * self.seed
                if self.pretrain == "mvm":
                    # Pre-train under `rs`, then fine-tune a fresh net under `rs + 1` with the
                    # pre-trained encoder copied in (`inp`, `rnn`; the head keeps its own
                    # initialisation) — the probe's `finetune_net` / `train_from`. `arch` is 'gru'
                    # here: `__init__` refuses anything else with this pretrain mode.
                    enc = self._pretrain(pool, keep, rs)
                    torch.manual_seed(rs + 1)
                    rng = np.random.RandomState(rs + 1)
                    net = _build_net(self.n_in_, self.d, self.layers, self.heads, self.drop, self.arch)
                    net.inp.load_state_dict(enc.inp.state_dict())
                    net.rnn.load_state_dict(enc.rnn.state_dict())
                else:
                    torch.manual_seed(rs)
                    rng = np.random.RandomState(rs)
                    net = _build_net(self.n_in_, self.d, self.layers, self.heads, self.drop, self.arch)
                opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                steps = int(np.ceil(len(x) / self.batch_size))
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps * self.epochs,
                                                            pct_start=0.1)
                for _ in range(self.epochs):
                    net.train()
                    perm = rng.permutation(len(x))
                    for s in range(steps):
                        bi = torch.from_numpy(perm[s * self.batch_size:(s + 1) * self.batch_size])
                        scale = self.RESIDUAL_SCALE if rsc is None else rsc[bi]
                        pred = b[bi] + scale * net(x[bi], m[bi])
                        w = (~m[bi]).float()
                        loss = ((pred - yt[bi]) ** 2 * w).sum() / w.sum()
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()
                        sched.step()
                self.states_.append({k: v.detach().clone() for k, v in net.state_dict().items()})
        return self

    # -- prediction -------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        Z = self._input_columns(self.norm_.transform(self._matrix(X)))
        P = _pack(X, Z, None, _base_curve(X), self._residual_scale(X))
        x, m, b = torch.from_numpy(P["x"]), torch.from_numpy(P["mask"]), torch.from_numpy(P["base"])
        rsc = torch.from_numpy(P["rs"]) if "rs" in P else None
        outs = []
        with _torch_threads(self.threads), torch.no_grad():
            for state in self.states_:
                net = _build_net(self.n_in_, self.d, self.layers, self.heads, self.drop, self.arch)
                net.load_state_dict(state)
                net.eval()
                pv = [(b[s:s + 512] + (self.RESIDUAL_SCALE if rsc is None else rsc[s:s + 512])
                       * net(x[s:s + 512], m[s:s + 512])).numpy()
                      for s in range(0, len(x), 512)]
                outs.append(np.concatenate(pv))
        packed = np.mean(outs, axis=0)
        pred = np.full(len(X), np.nan)
        sel = P["row"] >= 0
        pred[P["row"][sel]] = packed[sel]
        return clip_to_valid(pred) if self.clip else pred

    def get_params(self) -> dict[str, Any]:
        out = {"reps": self.reps, "epochs": self.epochs, "d": self.d, "layers": self.layers,
               "heads": self.heads, "drop": self.drop, "lr": self.lr, "seed": self.seed}
        if self.arch != "set":
            out["arch"] = self.arch
        if self.level_norm:
            out["level_norm"] = True
        if self.neighbour_cols:
            out["neighbour_cols"] = True
        if self.pretrain != "none":
            out["pretrain"] = self.pretrain
            out["pre_epochs"] = self.pre_epochs
        return out


@register
class VisitGRUModel(SetTransformerModel):
    """The same per-visit tokens, read as an age-ordered sequence by a bidirectional GRU.

    Alone it is better than the attention model (dev ≈ 5.99 against 6.09 with two reps, three CV seeds)
    and its errors are a little less tied to the boosters', which is what a blend member is for
    (`reports/setnn-2026-10-09/README.md`). Defaults: 3 GRU layers, width 128, everything else as the
    set-transformer."""

    name = "visit_gru"

    def __init__(
        self,
        reps: int = 2,
        epochs: int = 40,
        d: int = 128,
        layers: int = 3,
        drop: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        seed: int = 0,
        threads: int = 4,
        clip: bool = True,
        level_norm: bool = False,
        pretrain: str = "none",
        pre_epochs: int = 20,
    ) -> None:
        super().__init__(reps=reps, epochs=epochs, d=d, layers=layers, heads=4, drop=drop, lr=lr,
                         weight_decay=weight_decay, batch_size=batch_size, seed=seed, threads=threads,
                         clip=clip, arch="gru", level_norm=level_norm, pretrain=pretrain,
                         pre_epochs=pre_epochs)
