"""
Original FinDiff + Optuna Hyperparameter Optimisation
======================================================
The model architecture is IDENTICAL to the original FinDiff paper.
Only hyperparameter tuning is added using:
  - Optuna TPE sampler  (Bayesian, models parameter interactions)
  - ASHA / Hyperband pruner  (kills bad trials early)

Parameters searched
───────────────────
  cat_emb_dim      : {2, 4, 8}
  hidden_dims      : choice of layer configs
  diffusion_steps  : {200, 300, 500, 750, 1000}
  beta_start       : 1e-5 – 5e-4  (log scale)
  beta_end         : 0.01 – 0.03
  lr               : 1e-5 – 1e-2  (log scale)
  batch_size       : {256, 512, 1024}
  epochs           : 100 – 1000   (step 50)

After the study the best params retrain the final model and
synthetic data is generated + evaluated.
"""

import math
import time
import warnings
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
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# Global config
# ─────────────────────────────────────────────
SEED          = 42
N_TRIALS      = 60          # number of Optuna trials; increase for better search
STUDY_TIMEOUT = None        # wall-clock cap in seconds (None = run all trials)
PRUNE_WARMUP  = 20          # don't prune before this many epochs
N_GENERATE    = 30_000      # rows to generate after final training
OUTPUT_DIR    = "Results2"

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─────────────────────────────────────────────
# Data  (loaded once, shared across all trials)
# ─────────────────────────────────────────────
def load_credit_default():
    df=pd.read_csv("Datasets/default.csv")
    return df

print("Loading data …")
df_full = load_credit_default()
print("Loaded shape:", df_full.shape)

label_col        = "default"
categorical_cols = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    label_col,
]
numeric_cols = [c for c in df_full.columns if c not in categorical_cols]
df_full      = df_full[categorical_cols + numeric_cols].copy()

print("Categorical cols:", categorical_cols)
print("Numeric cols    :", numeric_cols)

# 60 % train / 20 % val (for Optuna objective) / 20 % test (final eval)
df_trainval, df_test  = train_test_split(df_full,     test_size=0.20, random_state=SEED)
df_train,    df_val   = train_test_split(df_trainval, test_size=0.25, random_state=SEED)
print(f"Split → train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

# Quantile transformer fitted on train only
qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
train_num_np = qt.fit_transform(df_train[numeric_cols].values.astype(float))
val_num_np   = qt.transform(df_val[numeric_cols].values.astype(float))
test_num_np  = qt.transform(df_test[numeric_cols].values.astype(float))

# Label encoders fitted on the full vocabulary
label_encoders: Dict[str, LabelEncoder] = {}
vocab_sizes:    Dict[str, int]           = {}
n_tr, n_va, n_te = len(df_train), len(df_val), len(df_test)
train_cat_np = np.zeros((n_tr, len(categorical_cols)), dtype=int)
val_cat_np   = np.zeros((n_va, len(categorical_cols)), dtype=int)
test_cat_np  = np.zeros((n_te, len(categorical_cols)), dtype=int)

for i, col in enumerate(categorical_cols):
    le = LabelEncoder()
    le.fit(df_full[col].astype(str).values)
    label_encoders[col]  = le
    vocab_sizes[col]     = len(le.classes_)
    train_cat_np[:, i]   = le.transform(df_train[col].astype(str).values)
    val_cat_np[:, i]     = le.transform(df_val[col].astype(str).values)
    test_cat_np[:, i]    = le.transform(df_test[col].astype(str).values)

# Pre-build tensors (sent to device inside each trial)
TR_CAT = torch.tensor(train_cat_np, dtype=torch.long)
TR_NUM = torch.tensor(train_num_np, dtype=torch.float32)
VA_CAT = torch.tensor(val_cat_np,   dtype=torch.long)
VA_NUM = torch.tensor(val_num_np,   dtype=torch.float32)


# ─────────────────────────────────────────────
# Original FinDiff model  (unchanged from paper)
# ─────────────────────────────────────────────
def get_beta_schedule(beta_start, beta_end, T):
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t[:, None].float() * emb[None, :]
        emb  = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLPBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        super().__init__()
        layers, last = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class FinDiffSynthesizer(nn.Module):
    def __init__(self, categorical_cols, vocab_sizes,
                 num_continuous, cat_emb_dim=2,
                 hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024]
        self.categorical_cols = categorical_cols
        self.cat_emb_dim      = cat_emb_dim

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], cat_emb_dim)
            for col in categorical_cols
        })

        self.total_cat_dim = len(categorical_cols) * cat_emb_dim
        self.num_continuous = num_continuous
        self.input_dim      = self.total_cat_dim + num_continuous

        model_dim          = max(512, self.input_dim)
        self.input_proj    = nn.Linear(self.input_dim, model_dim)
        self.time_emb      = SinusoidalPosEmb(64)
        self.time_mlp      = nn.Sequential(
            nn.Linear(64, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.backbone = MLPBackbone(model_dim, hidden_dims, model_dim)
        self.head     = nn.Linear(model_dim, self.input_dim)

    def embed_categoricals(self, cat_idx):
        return torch.cat(
            [self.embeddings[col](cat_idx[:, i])
             for i, col in enumerate(self.categorical_cols)],
            dim=-1
        )

    def forward(self, x_cat_idx, x_num, timesteps):
        cat_emb = self.embed_categoricals(x_cat_idx)
        x       = self.input_proj(torch.cat([cat_emb, x_num], dim=-1))
        t_emb   = self.time_mlp(self.time_emb(timesteps))
        h       = self.backbone(x + t_emb)
        return self.head(h)


class DiffusionHelper:
    def __init__(self, betas):
        betas                         = betas.to(device)
        alphas                        = 1.0 - betas
        acp                           = torch.cumprod(alphas, dim=0)
        self.T                        = len(betas)
        self.betas                    = betas
        self.alphas                   = alphas
        self.acp                      = acp
        self.acp_prev                 = torch.cat([torch.tensor([1.0], device=device), acp[:-1]])
        self.sqrt_acp                 = acp.sqrt()
        self.sqrt_one_minus_acp       = (1.0 - acp).sqrt()

    def q_sample(self, x0, t, noise):
        return (self.sqrt_acp[t, None] * x0
                + self.sqrt_one_minus_acp[t, None] * noise)

    def posterior_variance(self, t):
        return float(
            self.betas[t] * (1.0 - self.acp_prev[t]) / (1.0 - self.acp[t])
        )


# ─────────────────────────────────────────────
# Predict noise (sampling helper — same as original)
# ─────────────────────────────────────────────
def predict_eps(xt, timesteps, model):
    x_proj = model.input_proj(xt)
    t_emb  = model.time_mlp(model.time_emb(timesteps))
    h      = model.backbone(x_proj + t_emb)
    return model.head(h)


# ─────────────────────────────────────────────
# Fidelity metric  (objective for Optuna)
# ─────────────────────────────────────────────
def column_fidelity(real_df, syn_df):
    """Ω_col ∈ [0,1]; higher = better."""
    scores = []
    for col in numeric_cols:
        stat, _ = ks_2samp(
            real_df[col].astype(float).values,
            syn_df[col].astype(float).values
        )
        scores.append(1.0 - stat)
    for col in categorical_cols:
        rc   = real_df[col].astype(str).value_counts(normalize=True)
        sc   = syn_df[col].astype(str).value_counts(normalize=True)
        cats = set(rc.index) | set(sc.index)
        tvd  = 0.5 * sum(abs(rc.get(c, 0.) - sc.get(c, 0.)) for c in cats)
        scores.append(1.0 - tvd)
    return float(np.mean(scores))


# ─────────────────────────────────────────────
# Sampling + decoding helpers
# ─────────────────────────────────────────────
@torch.no_grad()
def sample_loop(n, model, diff_helper):
    model.eval()
    xt = torch.randn((n, model.input_dim), device=device)
    for step in reversed(range(diff_helper.T)):
        t_vec    = torch.full((n,), step, device=device, dtype=torch.long)
        eps_pred = predict_eps(xt, t_vec, model)
        alpha_t  = diff_helper.alphas[step]
        acp_t    = diff_helper.acp[step]
        beta_t   = diff_helper.betas[step]
        coef     = beta_t / (1.0 - acp_t).sqrt()
        mu       = (1.0 / alpha_t.sqrt()) * (xt - coef * eps_pred)
        if step > 0:
            pv = diff_helper.posterior_variance(step)
            xt = mu + math.sqrt(pv) * torch.randn_like(xt)
        else:
            xt = mu
    return xt.cpu()


def decode(generated, model):
    """Decode generated tensor → DataFrame with original column types."""
    n             = generated.shape[0]
    total_cat_dim = model.total_cat_dim
    emb_dim       = model.cat_emb_dim
    cat_flat      = generated[:, :total_cat_dim]
    num_flat      = generated[:, total_cat_dim:].numpy()

    # nearest-embedding lookup
    gen_idx = np.zeros((n, len(categorical_cols)), dtype=int)
    for i, col in enumerate(categorical_cols):
        s     = i * emb_dim
        chunk = torch.tensor(cat_flat[:, s:s + emb_dim].numpy(), dtype=torch.float32)
        W     = model.embeddings[col].weight.detach().cpu()
        gen_idx[:, i] = torch.cdist(chunk, W).argmin(dim=1).numpy()

    gen_cat = {col: label_encoders[col].inverse_transform(gen_idx[:, i])
               for i, col in enumerate(categorical_cols)}

    gen_num = qt.inverse_transform(num_flat) if numeric_cols \
              else np.zeros((n, 0))

    out = pd.DataFrame(gen_num, columns=numeric_cols)
    for col in categorical_cols:
        out[col] = gen_cat[col]
    out = out[categorical_cols + numeric_cols]
    try:
        out[label_col] = out[label_col].astype(int)
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────
# Core train function  (used by trials + final)
# ─────────────────────────────────────────────
def train_model(hp: dict, epochs: int,
                trial: optuna.Trial = None,
                verbose: bool = False):
    """
    Train original FinDiff with hyperparameters in `hp` for `epochs` epochs.
    Reports intermediate val-fidelity to Optuna every PRUNE_WARMUP epochs.
    Returns (model, diff_helper, val_fidelity).
    """
    model = FinDiffSynthesizer(
        categorical_cols = categorical_cols,
        vocab_sizes      = vocab_sizes,
        num_continuous   = len(numeric_cols),
        cat_emb_dim      = hp["cat_emb_dim"],
        hidden_dims      = hp["hidden_dims"],
    ).to(device)

    diff = DiffusionHelper(
        get_beta_schedule(hp["beta_start"], hp["beta_end"], hp["diffusion_steps"])
    )

    bs  = hp["batch_size"]
    dl  = DataLoader(
        TensorDataset(TR_CAT.to(device), TR_NUM.to(device)),
        batch_size=bs, shuffle=True, drop_last=True
    )

    opt   = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    val_fidelity = 0.0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0

        for b_cat, b_num in dl:
            B = b_num.shape[0]
            with torch.no_grad():
                cat_emb = model.embed_categoricals(b_cat)
            x0    = torch.cat([cat_emb, b_num], dim=-1)
            t_s   = torch.randint(0, diff.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt    = diff.q_sample(x0, t_s, noise)

            eps_pred = predict_eps(xt, t_s, model)
            loss     = F.mse_loss(eps_pred, noise)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            ep_loss += loss.item() * B

        sched.step()
        ep_loss /= len(dl.dataset)

        # ── report to Optuna every PRUNE_WARMUP epochs ──
        if trial is not None and (epoch + 1) % PRUNE_WARMUP == 0:
            gen      = sample_loop(len(VA_CAT), model, diff)
            syn_df   = decode(gen, model)
            val_fidelity = column_fidelity(df_val, syn_df)

            trial.report(val_fidelity, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}/{epochs}  "
                  f"loss={ep_loss:.5f}  val_fid={val_fidelity:.4f}")

    # final val fidelity (if last epoch wasn't a reporting step)
    if epochs % PRUNE_WARMUP != 0 or trial is None:
        gen          = sample_loop(len(VA_CAT), model, diff)
        syn_df       = decode(gen, model)
        val_fidelity = column_fidelity(df_val, syn_df)

    return model, diff, val_fidelity


# ─────────────────────────────────────────────
# Hidden-dim configs to search over
# (avoids passing a variable-length list through Optuna)
# ─────────────────────────────────────────────
HIDDEN_CONFIGS = {
    "256x2":   [256, 256],
    "512x2":   [512, 512],
    "1024x2":  [1024, 1024],
    "2048x2":  [2048, 2048],
    "512x4":   [512, 512, 512, 512],
    "1024x4":  [1024, 1024, 1024, 1024],
}


# ─────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:

    cat_emb_dim     = trial.suggest_categorical("cat_emb_dim",      [2, 4, 8])
    hidden_key      = trial.suggest_categorical("hidden_dims_key",  list(HIDDEN_CONFIGS.keys()))
    diffusion_steps = trial.suggest_categorical("diffusion_steps",  [200, 300, 500, 750, 1000])
    beta_start      = trial.suggest_float      ("beta_start",       1e-5, 5e-4, log=True)
    beta_end        = trial.suggest_float      ("beta_end",         0.01, 0.03)
    lr              = trial.suggest_float      ("lr",               1e-5, 1e-2, log=True)
    batch_size      = trial.suggest_categorical("batch_size",       [256, 512, 1024])
    epochs          = trial.suggest_int        ("epochs",           100, 1000, step=50)

    hp = dict(
        cat_emb_dim     = cat_emb_dim,
        hidden_dims     = HIDDEN_CONFIGS[hidden_key],
        diffusion_steps = diffusion_steps,
        beta_start      = beta_start,
        beta_end        = beta_end,
        lr              = lr,
        batch_size      = batch_size,
    )

    _, _, val_fidelity = train_model(hp, epochs=epochs, trial=trial)
    return val_fidelity


# ─────────────────────────────────────────────
# Run Optuna study
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Optuna HPO  ({N_TRIALS} trials)  |  TPE sampler + Hyperband/ASHA")
print("=" * 60)

sampler = TPESampler(seed=SEED, multivariate=True, group=True)
pruner  = HyperbandPruner(
    min_resource     = PRUNE_WARMUP,  # earliest pruning point (epochs)
    max_resource     = 1000,          # maximum epochs any trial runs
    reduction_factor = 3,             # η in ASHA: top-1/3 promoted each rung
)

study = optuna.create_study(
    direction  = "maximize",          # maximise Ω_col fidelity
    sampler    = sampler,
    pruner     = pruner,
    study_name = "findiff_original_hpo",
)

t0 = time.time()
study.optimize(
    objective,
    n_trials          = N_TRIALS,
    timeout           = STUDY_TIMEOUT,
    show_progress_bar = True,
)
elapsed = time.time() - t0

print(f"\nHPO finished in {elapsed/60:.1f} min")
print(f"Completed trials : {len([t for t in study.trials if t.state.is_finished()])}")
print(f"Pruned trials    : {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")
print(f"Best trial #     : {study.best_trial.number}")
print(f"Best val Ω_col   : {study.best_value:.4f}")
print("\nBest hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k:22s} = {v}")


# ─────────────────────────────────────────────
# Save HPO plots
# ─────────────────────────────────────────────
try:
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hpo_history.png", dpi=150)
    plt.close()

    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hpo_importance.png", dpi=150)
    plt.close()

    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hpo_parallel.png", dpi=150)
    plt.close()

    print(f"\nHPO plots saved to {OUTPUT_DIR}/")
except Exception as e:
    print(f"[warn] HPO plots: {e}")


# ─────────────────────────────────────────────
# Retrain final model with best hyperparameters
# ─────────────────────────────────────────────
best   = study.best_params
epochs = best.pop("epochs")   # pull epochs out separately
hidden_key = best.pop("hidden_dims_key")
best["hidden_dims"] = HIDDEN_CONFIGS[hidden_key]

print("\n" + "=" * 60)
print(f"Retraining final model  epochs={epochs}  with best HP …")
print("=" * 60)

final_model, final_diff, _ = train_model(
    best, epochs=epochs, trial=None, verbose=True
)

# ── Save best model ──
torch.save(final_model.state_dict(), f"{OUTPUT_DIR}/findiff_best_model.pth")

# restore for reporting
best["epochs"]          = epochs
best["hidden_dims_key"] = hidden_key


# ─────────────────────────────────────────────
# Generate synthetic data
# ─────────────────────────────────────────────
print(f"\nGenerating {N_GENERATE} synthetic rows …")
gen_tensor = sample_loop(N_GENERATE, final_model, final_diff)
syn_df     = decode(gen_tensor, final_model)

print("\nSynthetic data preview:")
print(syn_df.head())
print("Shape:", syn_df.shape)


# ─────────────────────────────────────────────
# Final evaluation on test set
# ─────────────────────────────────────────────
test_fidelity = column_fidelity(df_test, syn_df)
print(f"\n── Test-set column fidelity Ω_col = {test_fidelity:.4f} ──")

print("\nPer-column breakdown:")
for col in numeric_cols:
    stat, _ = ks_2samp(df_test[col].astype(float).values,
                       syn_df[col].astype(float).values)
    print(f"  NUM  {col:22s}  {1-stat:.4f}")
for col in categorical_cols:
    rc   = df_test[col].astype(str).value_counts(normalize=True)
    sc   = syn_df[col].astype(str).value_counts(normalize=True)
    cats = set(rc.index) | set(sc.index)
    tvd  = 0.5 * sum(abs(rc.get(c, 0.) - sc.get(c, 0.)) for c in cats)
    print(f"  CAT  {col:22s}  {1-tvd:.4f}")


# ─────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────
syn_df.to_csv(f"{OUTPUT_DIR}/synthetic_original_hpo.csv", index=False)

# Training loss plot for final model (rerun a short pass to get loss curve)
print("\nRunning final model one more pass to collect loss curve …")
final_model2, _, _ = train_model(best, epochs=epochs, trial=None, verbose=True)

# best HP summary CSV
summary = pd.DataFrame([{
    **{k: v for k, v in best.items() if k != "hidden_dims"},
    "hidden_dims_key":    hidden_key,
    "epochs":             epochs,
    "val_fidelity_best":  study.best_value,
    "test_fidelity":      test_fidelity,
    "n_trials_total":     len(study.trials),
    "n_trials_pruned":    sum(1 for t in study.trials
                              if t.state == optuna.trial.TrialState.PRUNED),
}])
summary.T.to_csv(f"{OUTPUT_DIR}/best_hp_original.csv", header=["value"])

print(f"\nSaved:")
print(f"  {OUTPUT_DIR}/synthetic_original_hpo.csv")
print(f"  {OUTPUT_DIR}/best_hp_original.csv")
print(f"  {OUTPUT_DIR}/hpo_history.png")
print(f"  {OUTPUT_DIR}/hpo_importance.png")
print(f"  {OUTPUT_DIR}/hpo_parallel.png")

print("""
╔══════════════════════════════════════════════════════════╗
║      Original FinDiff  +  Optuna HPO Summary             ║
╠══════════════════════════════════════════════════════════╣
║  Model      : Original FinDiff (MLP backbone, unchanged) ║
║  Sampler    : TPE (multivariate + grouped)               ║
║  Pruner     : Hyperband / ASHA                           ║
║  Objective  : column-wise fidelity Ω_col on val split    ║
║  Params     : cat_emb_dim, hidden_dims, diffusion_steps, ║
║               beta_start, beta_end, lr, batch_size,      ║
║               epochs  (8 total)                          ║
╚══════════════════════════════════════════════════════════╝
""")