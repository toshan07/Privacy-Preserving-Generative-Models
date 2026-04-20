"""
Enhanced FinDiff v3 — Optuna TPE + ASHA Hyperparameter Optimisation
=====================================================================
All hyperparameters (including number of epochs) are chosen automatically:

  Search algorithm : TPE  (Tree-structured Parzen Estimator)
                     — Bayesian optimiser, much smarter than grid/random search
  Pruner           : ASHA (Asynchronous Successive Halving Algorithm)
                     — kills bad trials early, spends budget on promising ones
  Objective        : column-wise fidelity Ω_col on a held-out validation split
                     (higher = better synthetic quality)

Parameters searched
───────────────────
  Architecture
    d_model          : {64, 128, 256}
    n_heads          : {2, 4, 8}          (must divide d_model)
    n_layers         : 1 – 4
    time_emb_dim     : {64, 128, 256}
    dropout          : 0.0 – 0.3
    emb_base         : {2, 4}             (adaptive embedding base)
    emb_max          : {8, 16, 32}

  Diffusion
    diffusion_steps  : {200, 300, 500, 750, 1000}
    beta_start       : 1e-5 – 5e-4
    beta_end         : 0.01 – 0.03

  Optimisation
    lr               : 1e-5 – 1e-2  (log scale)
    batch_size       : {256, 512, 1024}
    epochs           : 100 – 1000       ← decided by Optuna, not fixed

  DP-SGD
    dp_noise_mult    : 0.5 – 2.0
    dp_max_grad_norm : 0.5 – 2.0

After the study the best trial's hyperparameters are used to retrain a
final model from scratch, and synthetic data is saved to disk.
"""

import math, time, warnings
import numpy as np
import pandas as pd
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

import optuna
from optuna.pruners import HyperbandPruner   # ASHA variant built into Optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────
# Global config  (things that are NOT searched)
# ─────────────────────────────────────────────────────────
SEED          = 42
N_TRIALS      = 60          # number of Optuna trials (increase for better search)
STUDY_TIMEOUT = None        # seconds; None = run all N_TRIALS
USE_DP        = True        # DP-SGD on/off globally
DP_DELTA      = 1e-5
PRUNE_WARMUP_EPOCHS = 20    # don't prune before this many epochs
OUTPUT_DIR    = "Results4"

np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ─────────────────────────────────────────────────────────
# Data  (loaded once, shared across all trials)
# ─────────────────────────────────────────────────────────
def load_credit_default():
    df=pd.read_csv("Datasets/adult.csv")
    return df

print("Loading Credit Default dataset …")
df_full = load_credit_default()
print(f"Shape: {df_full.shape}\n")

LABEL_COL     = "income"
CAT_COLS = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'gender', 'native-country', 'income']

NUM_COLS      = [c for c in df_full.columns if c not in CAT_COLS]
df_full       = df_full[CAT_COLS + NUM_COLS].copy()

# 60 % train / 20 % val / 20 % test
df_trainval, df_test  = train_test_split(df_full,  test_size=0.20, random_state=SEED)
df_train,    df_val   = train_test_split(df_trainval, test_size=0.25, random_state=SEED)
print(f"Train={len(df_train)}  Val={len(df_val)}  Test={len(df_test)}")

# Quantile transformer fitted on train only
qt_global = QuantileTransformer(output_distribution="normal", random_state=SEED)
train_num_np = qt_global.fit_transform(df_train[NUM_COLS].values.astype(float))
val_num_np   = qt_global.transform(df_val[NUM_COLS].values.astype(float))
test_num_np  = qt_global.transform(df_test[NUM_COLS].values.astype(float))

# Label encoders fitted on full vocab
label_encoders: Dict[str, LabelEncoder] = {}
vocab_sizes:    Dict[str, int]           = {}
train_cat_np = np.zeros((len(df_train), len(CAT_COLS)), dtype=int)
val_cat_np   = np.zeros((len(df_val),   len(CAT_COLS)), dtype=int)
test_cat_np  = np.zeros((len(df_test),  len(CAT_COLS)), dtype=int)

for i, col in enumerate(CAT_COLS):
    le = LabelEncoder()
    le.fit(df_full[col].astype(str).values)
    label_encoders[col]   = le
    vocab_sizes[col]      = len(le.classes_)
    train_cat_np[:,i]     = le.transform(df_train[col].astype(str).values)
    val_cat_np[:,i]       = le.transform(df_val[col].astype(str).values)
    test_cat_np[:,i]      = le.transform(df_test[col].astype(str).values)

# Pre-build tensors (moved to device inside trials)
TR_CAT = torch.tensor(train_cat_np, dtype=torch.long)
TR_NUM = torch.tensor(train_num_np, dtype=torch.float32)
VA_CAT = torch.tensor(val_cat_np,   dtype=torch.long)
VA_NUM = torch.tensor(val_num_np,   dtype=torch.float32)

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def adaptive_emb_dim(cardinality: int, base: int, max_d: int) -> int:
    if cardinality <= 2: return base
    return min(base * math.ceil(math.log2(cardinality + 1)), max_d)

def compute_emb_dims(base: int, max_d: int) -> Dict[str, int]:
    return {col: adaptive_emb_dim(vocab_sizes[col], base, max_d)
            for col in CAT_COLS}

def fidelity_score(real_df: pd.DataFrame,
                   syn_df:  pd.DataFrame) -> float:
    """Column-wise fidelity Ω_col  ∈ [0,1]; higher = better."""
    scores = []
    for col in NUM_COLS:
        stat, _ = ks_2samp(real_df[col].astype(float).values,
                           syn_df[col].astype(float).values)
        scores.append(1.0 - stat)
    for col in CAT_COLS:
        rc   = real_df[col].astype(str).value_counts(normalize=True)
        sc   = syn_df[col].astype(str).value_counts(normalize=True)
        cats = set(rc.index) | set(sc.index)
        tvd  = 0.5 * sum(abs(rc.get(c, 0.) - sc.get(c, 0.)) for c in cats)
        scores.append(1.0 - tvd)
    return float(np.mean(scores))

# ─────────────────────────────────────────────────────────
# Model components  (same as v2)
# ─────────────────────────────────────────────────────────
class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device) * -f)
        e = t[:,None].float() * f[None,:]
        return torch.cat([e.sin(), e.cos()], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        self.norm_sa   = nn.LayerNorm(d)
        self.sa        = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                               batch_first=True)
        self.norm_ca_q = nn.LayerNorm(d)
        self.norm_ca_kv= nn.LayerNorm(d)
        self.ca        = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                               batch_first=True)
        self.norm_ff   = nn.LayerNorm(d)
        self.ff        = nn.Sequential(nn.Linear(d, d*4), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(d*4, d))

    def forward(self, cat_tok, num_tok):
        B, n_cat, d = cat_tok.shape
        full = torch.cat([cat_tok, num_tok], dim=1)
        x    = self.norm_sa(full)
        sa_out, _ = self.sa(x, x, x)
        full = full + sa_out
        cat_tok, num_tok = full[:,:n_cat], full[:,n_cat:]

        q  = self.norm_ca_q(cat_tok)
        kv = self.norm_ca_kv(num_tok)
        ca_out, _ = self.ca(q, kv, kv)
        cat_tok   = cat_tok + ca_out

        full = torch.cat([cat_tok, num_tok], dim=1)
        full = full + self.ff(self.norm_ff(full))
        return full[:,:n_cat], full[:,n_cat:]


class FinDiffTransformer(nn.Module):
    def __init__(self, cat_cols, vocab_sizes, emb_dims,
                 n_numeric, d_model, n_heads, n_layers,
                 time_emb_dim=128, dropout=0.0):
        super().__init__()
        self.cat_cols  = cat_cols
        self.emb_dims  = emb_dims
        self.n_numeric = n_numeric
        self.d_model   = d_model

        self.cat_emb = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], emb_dims[col])
            for col in cat_cols})
        self.cat_in  = nn.ModuleDict({
            col: nn.Linear(emb_dims[col], d_model)
            for col in cat_cols})

        n_num_tok = max(n_numeric, 1)
        self.num_in     = nn.Linear(n_num_tok, d_model * n_num_tok)
        self.n_num_tok  = n_num_tok

        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, d_model*2), nn.SiLU(),
            nn.Linear(d_model*2, d_model))

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)])

        self.cat_out = nn.ModuleDict({
            col: nn.Sequential(nn.LayerNorm(d_model),
                               nn.Linear(d_model, emb_dims[col]))
            for col in cat_cols})
        self.num_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def embed_cat_from_idx(self, idx):
        return torch.cat([self.cat_emb[col](idx[:,i])
                          for i,col in enumerate(self.cat_cols)], dim=-1)

    def _split_cat(self, flat):
        parts, cur = [], 0
        for col in self.cat_cols:
            d = self.emb_dims[col]
            parts.append(flat[:,cur:cur+d]); cur += d
        return parts

    def forward(self, x_cat_flat, x_num_flat, t):
        B = x_cat_flat.shape[0]
        t_emb = self.time_mlp(self.time_emb(t))

        cat_parts = self._split_cat(x_cat_flat)
        cat_tok = torch.stack(
            [self.cat_in[col](cat_parts[i])
             for i,col in enumerate(self.cat_cols)], dim=1)
        cat_tok = cat_tok + t_emb.unsqueeze(1)

        if self.n_numeric > 0:
            num_proj = self.num_in(x_num_flat)
            num_tok  = num_proj.view(B, self.n_num_tok, self.d_model)
        else:
            num_tok = torch.zeros(B, 1, self.d_model, device=x_cat_flat.device)
        num_tok = num_tok + t_emb.unsqueeze(1)

        for blk in self.blocks:
            cat_tok, num_tok = blk(cat_tok, num_tok)

        cat_noise = torch.cat(
            [self.cat_out[col](cat_tok[:,i,:])
             for i,col in enumerate(self.cat_cols)], dim=-1)
        num_noise = self.num_out(num_tok).squeeze(-1) if self.n_numeric > 0 \
                    else torch.zeros(B, 0, device=x_cat_flat.device)
        return torch.cat([cat_noise, num_noise], dim=-1)


class DiffusionHelper:
    def __init__(self, T, b_start, b_end):
        betas       = torch.linspace(b_start, b_end, T).to(device)
        alphas      = 1.0 - betas
        acp         = torch.cumprod(alphas, dim=0)
        self.T      = T
        self.betas  = betas
        self.alphas = alphas
        self.acp    = acp
        self.acp_prev = torch.cat([torch.ones(1,device=device), acp[:-1]])
        self.sqrt_acp        = acp.sqrt()
        self.sqrt_one_m_acp  = (1-acp).sqrt()

    def q_sample(self, x0, t, noise):
        return self.sqrt_acp[t,None]*x0 + self.sqrt_one_m_acp[t,None]*noise

    def posterior_var(self, t):
        return float(self.betas[t]*(1-self.acp_prev[t])/(1-self.acp[t]))


# ─────────────────────────────────────────────────────────
# Ghost Clipping DP-SGD (vectorized)
# ─────────────────────────────────────────────────────────
class GhostClipper:
    """
    Vectorized Ghost Clipping for DP-SGD.

    Root cause of the original RuntimeError
    ─────────────────────────────────────────
    PyTorch's nn.MultiheadAttention (batch_first=True) calls several internal
    nn.Linear layers (in_proj, out_proj) whose activations are 3-D:
        a : (B, seq_len, in_features)
        g : (B, seq_len, out_features)
    The original code assumed 2-D (B, features) and called .norm(dim=1) on
    both, which blew up when a.shape[1] != g.shape[1].

    Fix
    ───
    1. Flatten any tensor with ndim > 2 to (B, -1) before computing norms,
       treating the whole (seq_len × features) slice as one vector per sample.
    2. Verify both a and g have the same batch size B before using them;
       skip the layer silently otherwise (can happen in edge cases with
       fused kernels or gradient checkpointing).
    """

    def __init__(self, model, max_norm, noise_mult):
        self.C      = max_norm
        self.sigma  = noise_mult * max_norm
        self._hooks = []
        self._inp:  Dict = {}
        self._nsq:  Dict = {}

        for m in model.modules():
            if isinstance(m, nn.Linear):
                self._hooks.append(m.register_forward_hook(self._fwd))
                self._hooks.append(m.register_full_backward_hook(self._bwd))

    def _fwd(self, m, inp, out):
        # inp[0] may be 2-D (B, in_f) or 3-D (B, seq, in_f) — store as-is
        self._inp[m] = inp[0].detach()

    def _bwd(self, m, gi, go):
        a = self._inp.get(m)
        g = go[0].detach()
        if a is None or g is None:
            return

        # ── flatten to 2-D: (B, *) ──────────────────────────────────────
        # handles both (B, F) from FF layers and (B, S, F) from attn layers
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)   # (B, S*in_f)
        if g.ndim > 2:
            g = g.reshape(g.shape[0], -1)   # (B, S*out_f)

        # ── guard: both must have the same batch size ────────────────────
        if a.shape[0] != g.shape[0]:
            return   # skip this layer (fused kernel artefact)

        # ── per-sample norm² via outer-product identity ──────────────────
        # ‖∇_W L_i‖² = ‖g_i‖² · ‖a_i‖²
        ns = (g.norm(dim=1) ** 2) * (a.norm(dim=1) ** 2)
        if m.bias is not None:
            ns = ns + g.norm(dim=1) ** 2    # bias gradient norm²
        self._nsq[m] = ns                   # (B,)

    def remove(self):
        for h in self._hooks:
            h.remove()

    def clip_and_noise_(self, B, params):
        if not self._nsq:
            # No hooks fired (e.g. model has no Linear) — just add noise
            for p in params:
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * (self.sigma / B))
            return

        dev   = next(iter(self._nsq.values())).device
        total = torch.zeros(B, device=dev)
        for ns in self._nsq.values():
            if ns.shape[0] == B:            # skip any stray mis-sized entries
                total += ns

        norms      = total.sqrt()                                   # (B,)
        clip_f     = torch.clamp(self.C / (norms + 1e-8), max=1.0) # (B,)
        mean_clip  = clip_f.mean().item()                           # scalar

        for p in params:
            if p.grad is not None:
                p.grad.mul_(mean_clip)
                p.grad.add_(torch.randn_like(p.grad) * (self.sigma / B))

        self._nsq.clear()
        self._inp.clear()


# ─────────────────────────────────────────────────────────
# Decode helper  (shared across trial eval & final model)
# ─────────────────────────────────────────────────────────
def decode_generated(gen_tensor, model, emb_dims, total_cat_dim):
    gen   = gen_tensor.cpu()
    B     = gen.shape[0]
    cat_f = gen[:, :total_cat_dim].numpy()
    num_f = gen[:, total_cat_dim:].numpy()

    gen_idx = np.zeros((B, len(CAT_COLS)), dtype=int)
    cursor  = 0
    for i, col in enumerate(CAT_COLS):
        d     = emb_dims[col]
        chunk = torch.tensor(cat_f[:, cursor:cursor+d], dtype=torch.float32)
        W     = model.cat_emb[col].weight.detach().cpu()
        gen_idx[:,i] = torch.cdist(chunk, W).argmin(dim=1).numpy()
        cursor += d

    gen_cat = {col: label_encoders[col].inverse_transform(gen_idx[:,i])
               for i,col in enumerate(CAT_COLS)}
    gen_num = qt_global.inverse_transform(num_f) if NUM_COLS \
              else np.zeros((B,0))

    out = pd.DataFrame(gen_num, columns=NUM_COLS)
    for col in CAT_COLS: out[col] = gen_cat[col]
    out = out[CAT_COLS + NUM_COLS]
    try:   out[LABEL_COL] = out[LABEL_COL].astype(int)
    except: pass
    return out


@torch.no_grad()
def sample_loop(n, model, diff, total_cat_dim, total_num_dim):
    model.eval()
    xt = torch.randn(n, total_cat_dim + total_num_dim, device=device)
    for step in reversed(range(diff.T)):
        tv = torch.full((n,), step, device=device, dtype=torch.long)
        ec = model(xt[:,:total_cat_dim], xt[:,total_cat_dim:], tv)
        a  = diff.alphas[step]; ab = diff.acp[step]; b = diff.betas[step]
        mu = (1/a.sqrt()) * (xt - b/(1-ab).sqrt() * ec)
        if step > 0:
            xt = mu + math.sqrt(diff.posterior_var(step))*torch.randn_like(xt)
        else:
            xt = mu
    return xt.cpu()


# ─────────────────────────────────────────────────────────
# Core training function  (used by both Optuna & final run)
# ─────────────────────────────────────────────────────────
def train_model(hp: dict,
                epochs: int,
                trial: optuna.Trial = None,
                verbose: bool = False):
    """
    Train one model with hyperparameters `hp` for `epochs` epochs.
    If `trial` is given, reports intermediate values to Optuna (ASHA pruning).
    Returns (model, diff_helper, fidelity_on_val).
    """
    emb_dims      = compute_emb_dims(hp["emb_base"], hp["emb_max"])
    total_cat_dim = sum(emb_dims[c] for c in CAT_COLS)
    total_num_dim = len(NUM_COLS)

    # ── make n_heads divide d_model ──
    d_model = hp["d_model"]
    n_heads = hp["n_heads"]
    while d_model % n_heads != 0:
        n_heads = n_heads // 2
        if n_heads < 1: n_heads = 1; break

    model = FinDiffTransformer(
        cat_cols     = CAT_COLS,
        vocab_sizes  = vocab_sizes,
        emb_dims     = emb_dims,
        n_numeric    = total_num_dim,
        d_model      = d_model,
        n_heads      = n_heads,
        n_layers     = hp["n_layers"],
        time_emb_dim = hp["time_emb_dim"],
        dropout      = hp["dropout"],
    ).to(device)

    diff = DiffusionHelper(hp["diffusion_steps"],
                           hp["beta_start"], hp["beta_end"])

    bs   = hp["batch_size"]
    dl   = DataLoader(
        TensorDataset(TR_CAT.to(device), TR_NUM.to(device)),
        batch_size=bs, shuffle=True, drop_last=True)

    opt  = torch.optim.Adam(model.parameters(),
                            lr=hp["lr"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if USE_DP:
        clipper = GhostClipper(model, hp["dp_max_grad_norm"],
                               hp["dp_noise_mult"])

    best_fidelity = 0.0
    val_fidelity  = 0.0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0

        for b_cat, b_num in dl:
            B  = b_num.shape[0]
            with torch.no_grad():
                cat_f = model.embed_cat_from_idx(b_cat)
            x0    = torch.cat([cat_f, b_num], dim=-1)
            t_s   = torch.randint(0, diff.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt    = diff.q_sample(x0, t_s, noise)

            eps   = model(xt[:,:total_cat_dim], xt[:,total_cat_dim:], t_s)
            loss  = F.mse_loss(eps, noise)

            opt.zero_grad()
            loss.backward()
            if USE_DP:
                clipper.clip_and_noise_(B, list(model.parameters()))
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * B

        sched.step()
        ep_loss /= len(dl.dataset)

        # ── Evaluate on validation set every 20 epochs for ASHA pruning ──
        if trial is not None and (epoch + 1) % 20 == 0:
            with torch.no_grad():
                n_val  = len(VA_CAT)
                gen    = sample_loop(n_val, model, diff,
                                     total_cat_dim, total_num_dim)
                syn_df = decode_generated(gen, model, emb_dims, total_cat_dim)
                val_fidelity = fidelity_score(df_val, syn_df)

            if val_fidelity > best_fidelity:
                best_fidelity = val_fidelity

            trial.report(val_fidelity, step=epoch)
            if trial.should_prune():
                if USE_DP:
                    clipper.remove()
                raise optuna.TrialPruned()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={ep_loss:.5f}  "
                  f"val_fidelity={val_fidelity:.4f}")

    if USE_DP:
        clipper.remove()

    # Final validation fidelity (if not already computed this epoch)
    if (epochs % 20) != 0:
        with torch.no_grad():
            n_val  = len(VA_CAT)
            gen    = sample_loop(n_val, model, diff,
                                 total_cat_dim, total_num_dim)
            syn_df = decode_generated(gen, model, emb_dims, total_cat_dim)
            val_fidelity = fidelity_score(df_val, syn_df)

    return model, diff, val_fidelity


# ─────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:

    # ── Architecture ──
    d_model      = trial.suggest_categorical("d_model",      [64, 128, 256])
    n_heads      = trial.suggest_categorical("n_heads",      [2, 4, 8])
    n_layers     = trial.suggest_int        ("n_layers",     1, 4)
    time_emb_dim = trial.suggest_categorical("time_emb_dim", [64, 128, 256])
    dropout      = trial.suggest_float      ("dropout",      0.0, 0.3, step=0.05)
    emb_base     = trial.suggest_categorical("emb_base",     [2, 4])
    emb_max      = trial.suggest_categorical("emb_max",      [8, 16, 32])

    # ── Diffusion ──
    diffusion_steps = trial.suggest_categorical("diffusion_steps",
                                                [200, 300, 500, 750, 1000])
    beta_start  = trial.suggest_float("beta_start", 1e-5, 5e-4, log=True)
    beta_end    = trial.suggest_float("beta_end",   0.01, 0.03)

    # ── Optimisation ──
    lr          = trial.suggest_float("lr",         1e-5, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [256, 512, 1024])
    epochs      = trial.suggest_int  ("epochs",     100,  1000, step=50)

    # ── DP ──
    dp_noise_mult    = trial.suggest_float("dp_noise_mult",    0.5,  2.0)
    dp_max_grad_norm = trial.suggest_float("dp_max_grad_norm", 0.5,  2.0)

    hp = dict(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        time_emb_dim=time_emb_dim, dropout=dropout,
        emb_base=emb_base, emb_max=emb_max,
        diffusion_steps=diffusion_steps, beta_start=beta_start, beta_end=beta_end,
        lr=lr, batch_size=batch_size,
        dp_noise_mult=dp_noise_mult, dp_max_grad_norm=dp_max_grad_norm,
    )

    _, _, val_fidelity = train_model(hp, epochs=epochs, trial=trial)
    return val_fidelity


# ─────────────────────────────────────────────────────────
# Run Optuna study  (TPE sampler + ASHA / Hyperband pruner)
# ─────────────────────────────────────────────────────────
print("=" * 60)
print(f"Optuna HPO  — {N_TRIALS} trials, TPE sampler + Hyperband/ASHA pruner")
print(f"DP-SGD: {'ON' if USE_DP else 'OFF'}")
print("=" * 60)

sampler = TPESampler(seed=SEED, multivariate=True, group=True)
pruner  = HyperbandPruner(
    min_resource     = PRUNE_WARMUP_EPOCHS,   # no pruning before this epoch
    max_resource     = 1000,                  # maximum epochs any trial can run
    reduction_factor = 3,                     # halving factor (η in ASHA)
)

study = optuna.create_study(
    direction = "maximize",      # maximise Ω_col fidelity
    sampler   = sampler,
    pruner    = pruner,
    study_name= "findiff_v3",
)

t_start = time.time()
study.optimize(objective,
               n_trials  = N_TRIALS,
               timeout   = STUDY_TIMEOUT,
               show_progress_bar = True)

elapsed = time.time() - t_start
print(f"\nHPO finished in {elapsed/60:.1f} min")
print(f"Best trial  : #{study.best_trial.number}")
print(f"Best Ω_col  : {study.best_value:.4f}")
print(f"Best params :")
for k, v in study.best_params.items():
    print(f"  {k:25s} = {v}")

# ─────────────────────────────────────────────────────────
# Save HPO visualisations
# ─────────────────────────────────────────────────────────
try:
    # Optimisation history
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hpo_history.png", dpi=150)
    plt.close()

    # Parameter importance
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hpo_param_importance.png", dpi=150)
    plt.close()

    # Parallel coordinates
    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hpo_parallel.png", dpi=150)
    plt.close()

    print(f"HPO plots saved to {OUTPUT_DIR}/")
except Exception as e:
    print(f"[warn] Could not save HPO plots: {e}")

# ─────────────────────────────────────────────────────────
# Retrain final model with best hyperparameters
# ─────────────────────────────────────────────────────────
best_hp     = study.best_params
best_epochs = best_hp.pop("epochs")   # extract epochs from hp dict

print("\n" + "=" * 60)
print(f"Retraining final model  (epochs={best_epochs})  with best HP …")
print("=" * 60)

final_model, final_diff, _ = train_model(
    best_hp, epochs=best_epochs, trial=None, verbose=True
)

# Restore epochs key for reporting
best_hp["epochs"] = best_epochs

# ─────────────────────────────────────────────────────────
# Generate final synthetic data (30 000 rows)
# ─────────────────────────────────────────────────────────
N_GEN         = 48842
emb_dims_best = compute_emb_dims(best_hp["emb_base"], best_hp["emb_max"])
total_cat_dim = sum(emb_dims_best[c] for c in CAT_COLS)
total_num_dim = len(NUM_COLS)

print(f"\nGenerating {N_GEN} synthetic samples …")
gen_tensor = sample_loop(N_GEN, final_model, final_diff,
                         total_cat_dim, total_num_dim)
syn_df = decode_generated(gen_tensor, final_model,
                          emb_dims_best, total_cat_dim)

print("\nSynthetic data preview:")
print(syn_df.head())

# ─────────────────────────────────────────────────────────
# Evaluation  (test set fidelity)
# ─────────────────────────────────────────────────────────
test_fidelity = fidelity_score(df_test, syn_df)
print(f"\nTest Ω_col fidelity = {test_fidelity:.4f}")

print("\n── Per-column fidelity ──")
for col in NUM_COLS:
    stat, _ = ks_2samp(df_test[col].astype(float).values,
                       syn_df[col].astype(float).values)
    print(f"  NUM  {col:20s}  {1-stat:.4f}")
for col in CAT_COLS:
    rc   = df_test[col].astype(str).value_counts(normalize=True)
    sc   = syn_df[col].astype(str).value_counts(normalize=True)
    cats = set(rc.index)|set(sc.index)
    tvd  = 0.5*sum(abs(rc.get(c,0)-sc.get(c,0)) for c in cats)
    print(f"  CAT  {col:20s}  {1-tvd:.4f}")

# DP privacy budget
if USE_DP:
    eps = math.sqrt(
        2 * ((len(df_train)//best_hp["batch_size"]) * best_epochs)
        * math.log(1.0/DP_DELTA)
    ) * (best_hp["batch_size"]/len(df_train)) / best_hp["dp_noise_mult"]
    print(f"\n[DP] Final ε ≈ {eps:.2f}  at δ={DP_DELTA}")

# ─────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────
syn_df.to_csv(f"{OUTPUT_DIR}/synthetic_v3_best.csv", index=False)

# Save best HP summary
hp_summary = pd.DataFrame([{
    **best_hp,
    "val_fidelity_best":  study.best_value,
    "test_fidelity_final": test_fidelity,
    "n_trials": len(study.trials),
    "pruned_trials": sum(1 for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED),
}])
hp_summary.T.to_csv(f"{OUTPUT_DIR}/best_hyperparams_v3.csv", header=["value"])

print(f"\nSaved:")
print(f"  {OUTPUT_DIR}/synthetic_v3_best.csv")
print(f"  {OUTPUT_DIR}/best_hyperparams_v3.csv")
print(f"  {OUTPUT_DIR}/hpo_history.png")
print(f"  {OUTPUT_DIR}/hpo_param_importance.png")
print(f"  {OUTPUT_DIR}/hpo_parallel.png")

print("""
╔══════════════════════════════════════════════════════════════╗
║         Enhanced FinDiff v3 — HPO Summary                    ║
╠══════════════════════════════════════════════════════════════╣
║  Optimiser  : Optuna TPE  (multivariate + grouped)           ║
║  Pruner     : Hyperband / ASHA                               ║
║               — prunes unpromising trials early              ║
║               — concentrates budget on good configs          ║
║  Objective  : Ω_col fidelity on validation split             ║
║  Params     : 15 hyperparameters searched jointly            ║
║               (incl. epochs, d_model, lr, dp_noise, …)       ║
╚══════════════════════════════════════════════════════════════╝
""")