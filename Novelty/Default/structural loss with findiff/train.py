import math
import time
import json
import warnings
from itertools import combinations
from typing import List, Dict, Tuple, Optional
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import (
    QuantileTransformer, LabelEncoder,
    OrdinalEncoder, OneHotEncoder, StandardScaler
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import ks_2samp, pearsonr
from lightgbm import LGBMClassifier, LGBMRegressor

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Hyper-parameters ───────────────────────────────────────────────────────
EMB_DIM         = 2           # per-attribute categorical embedding dim (FinDiff paper)
DIFFUSION_STEPS = 500         # T in DDPM schedule
BATCH_SIZE      = 512
EPOCHS          = 500         # reduce to ~100 for a quick test run
LR              = 1e-4
HIDDEN_DIMS     = [1024, 1024]
BETA_START      = 1e-4
BETA_END        = 0.02

# ── λ_sf sweep values ──────────────────────────────────────────────────────
# 0.0 → vanilla FinDiff; increase to add structural fidelity regularisation
LAMBDA_SF_VALUES = [0.0,0.1,0.25,0.5,1]

N_GENERATE = 30_000           # synthetic rows to generate per run
OUT_DIR    = "Results5"
print("Config ready.")


def load_taiwan_credit(
    path: str = "Datasets/default.csv"
) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_column_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = [
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "default"
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols     = [c for c in df.columns if c not in categorical_cols]
    return categorical_cols, numeric_cols


def preprocess(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    quantile_transformer: Optional[QuantileTransformer] = None,
    label_encoders: Optional[Dict] = None,
    fit: bool = True,
):
    """Return tensors + fitted transformers."""
    # numeric
    if fit:
        qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
        num_arr = qt.fit_transform(df[numeric_cols].values.astype(float)) \
                  if numeric_cols else np.zeros((len(df), 0))
    else:
        qt = quantile_transformer
        num_arr = qt.transform(df[numeric_cols].values.astype(float)) \
                  if numeric_cols else np.zeros((len(df), 0))

    # categorical
    if fit:
        les: Dict[str, LabelEncoder] = {}
        vocab_sizes: Dict[str, int]  = {}
    else:
        les = label_encoders
        vocab_sizes = {col: len(les[col].classes_) for col in categorical_cols}

    cat_arr = np.zeros((len(df), len(categorical_cols)), dtype=int)
    for i, col in enumerate(categorical_cols):
        if fit:
            le = LabelEncoder().fit(df[col].astype(str).values)
            les[col] = le
            vocab_sizes[col] = len(le.classes_)
        cat_arr[:, i] = les[col].transform(df[col].astype(str).values)

    x_cat = torch.tensor(cat_arr, dtype=torch.long).to(device)
    x_num = torch.tensor(num_arr, dtype=torch.float32).to(device)
    return x_cat, x_num, qt, les, vocab_sizes


# ── Load data ──────────────────────────────────────────────────────────────
df = load_taiwan_credit()
print(f"Loaded: {df.shape}")

categorical_cols, numeric_cols = build_column_lists(df)
label_col = "default"
print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric    ({len(numeric_cols)}): {numeric_cols}")

# train / test splits
train_df, test_df   = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
real_train, real_test = train_test_split(train_df, test_size=0.2, random_state=SEED)

x_cat_tr, x_num_tr, quantile, label_encoders, vocab_sizes = \
    preprocess(train_df, categorical_cols, numeric_cols, fit=True)

train_dataset = TensorDataset(x_cat_tr, x_num_tr)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=True)

print(f"Train loader: {len(train_loader)} batches of {BATCH_SIZE}")

def get_beta_schedule(beta_start: float, beta_end: float, T: int) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t[:, None].float() * emb[None, :]
        emb  = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], out_dim: int):
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
    """
    Core FinDiff model — architecture identical to the original.
    Structural fidelity regularisation is applied externally in the
    training loop via compute_sf_loss(), keeping this class unmodified.
    """
    def __init__(self,
                 categorical_cols: List[str],
                 vocab_sizes: Dict[str, int],
                 num_continuous: int,
                 cat_emb_dim: int = 2,
                 hidden_dims: List[int] = [1024, 1024]):
        super().__init__()
        self.categorical_cols = categorical_cols
        self.cat_emb_dim      = cat_emb_dim
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], cat_emb_dim)
            for col in categorical_cols
        })
        self.total_cat_dim  = len(categorical_cols) * cat_emb_dim
        self.num_continuous = num_continuous
        self.input_dim      = self.total_cat_dim + num_continuous

        model_dim = max(512, self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, model_dim)
        self.time_emb   = SinusoidalPosEmb(64)
        self.time_mlp   = nn.Sequential(
            nn.Linear(64, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.backbone = MLPBackbone(model_dim, hidden_dims, model_dim)
        self.head     = nn.Linear(model_dim, self.input_dim)

    def embed_categoricals(self, cat_idx: torch.LongTensor) -> torch.FloatTensor:
        return torch.cat(
            [self.embeddings[col](cat_idx[:, i])
             for i, col in enumerate(self.categorical_cols)], dim=-1
        )

    def forward(self, xt: torch.Tensor, timesteps: torch.LongTensor) -> torch.Tensor:
        x   = self.input_proj(xt)
        t_e = self.time_mlp(self.time_emb(timesteps))
        h   = self.backbone(x + t_e)
        return self.head(h)


class DiffusionHelper:
    def __init__(self, betas: torch.Tensor):
        self.betas              = betas.to(device)
        alphas                  = 1.0 - self.betas
        self.alphas_cumprod     = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]
        ])
        self.sqrt_alphas_cumprod           = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        self.T = len(betas)

    def q_sample(self, x0, t, noise):
        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return a * x0 + b * noise

    def predict_x0(self, xt, t, eps):
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        return (xt - b * eps) / (a + 1e-8)

    def posterior_variance(self, t: int) -> float:
        return float(
            self.betas[t] * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t] + 1e-8)
        )


betas = get_beta_schedule(BETA_START, BETA_END, DIFFUSION_STEPS)
print("Model components defined.")

def batch_correlation_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Pearson correlation matrix over the feature dimension.

    Args:
        x : Tensor [B, D]  (batch × feature-dim)
    Returns:
        corr : Tensor [D, D]

    Matching this matrix between x0_pred and x0_real encourages
    the denoiser to preserve inter-feature co-dependence —
    the empirical proxy for causal structure used in TabStruct.
    """
    x   = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=eps)
    x   = x / std
    return (x.T @ x) / (x.shape[0] - 1 + eps)


def compute_sf_loss(
    x0_pred: torch.Tensor,
    x0_real: torch.Tensor,
    global_utility: Optional[float] = None,
    lambda_global_utility: float = 0.0,
    target_global_utility: float = 1.0,
) -> torch.Tensor:
    """
    Structural Fidelity Loss (novel term).

    ||corr(x0_pred) - corr(x0_real)||^2_F

    λ_sf = 0  →  this term is multiplied by 0 → identical to vanilla FinDiff
    λ_sf > 0  →  model penalised for breaking inter-feature correlation structure
    """
    corr_pred = batch_correlation_matrix(x0_pred)
    corr_real = batch_correlation_matrix(x0_real.detach())  # no grad through real data
    sf_loss = F.mse_loss(corr_pred, corr_real)

    # Optional non-differentiable utility regularizer from compute_global_utility.
    # Pass global_utility as either a float or utility_dict["global_utility"].

    if lambda_global_utility > 0.0 and global_utility is not None:
        if isinstance(global_utility, dict):
            gu_value = float(global_utility.get("global_utility", 0.0))
        else:
            gu_value = float(global_utility)

        gu_penalty = torch.tensor(
            (target_global_utility - gu_value) ** 2,
            device=sf_loss.device,
            dtype=sf_loss.dtype,
        )
        # my_loss=lambda_global_utility * gu_penalty
        # sf_loss = sf_loss + my_loss

    return gu_penalty

def train_findiff(
    train_loader: DataLoader,
    model: FinDiffSynthesizer,
    diff: DiffusionHelper,
    epochs: int,
    lr: float,
    lambda_sf: float = 0.0,
    global_utility: Optional[float] = None,
    lambda_global_utility: float = 0.0,
) -> List[float]:
    """
    Train FinDiff with optional structural fidelity regularisation.

    Total loss = L_diffusion  +  λ_sf · L_sf

    λ_sf = 0  →  pure FinDiff, bit-for-bit identical to original
    λ_sf > 0  →  SF-regularised FinDiff (novel)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []

    print(f"\n{'='*55}")
    print(f"  Training  λ_sf={lambda_sf:.2f}   epochs={epochs}")
    print(f"{'='*55}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_cat_idx, batch_num in train_loader:
            B = batch_num.shape[0]

            # build clean embedding x0
            with torch.no_grad():
                cat_emb = model.embed_categoricals(batch_cat_idx)
            x0 = torch.cat([cat_emb, batch_num], dim=-1)   # [B, D]

            # forward diffusion
            t     = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(x0)
            xt    = diff.q_sample(x0, t, noise)

            # denoising
            eps_pred = model(xt, t)

            # diffusion loss (standard MSE on noise)
            loss_diff = F.mse_loss(eps_pred, noise)

            # structural fidelity loss (novel)
            if lambda_sf > 0.0:
                x0_pred  = diff.predict_x0(xt, t, eps_pred)
                loss_sf  = compute_sf_loss(
                    x0_pred,
                    x0,
                    global_utility=global_utility,
                    lambda_global_utility=lambda_global_utility,
                )
                loss     = loss_diff + lambda_sf * loss_sf
            else:
                loss     = loss_diff    # λ_sf=0 → identical to vanilla FinDiff

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item() * B

        scheduler.step()
        epoch_loss /= len(train_loader.dataset)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  "
                  f"loss={epoch_loss:.6f}  time={time.time()-t0:.1f}s")

    return loss_history


print("Training function defined.")

@torch.no_grad()
def sample_loop(
    n_samples: int,
    model: FinDiffSynthesizer,
    diff: DiffusionHelper,
) -> torch.Tensor:
    """DDPM ancestral sampling — identical to original FinDiff."""
    model.eval()
    xt = torch.randn((n_samples, model.input_dim), device=device)

    for step in reversed(range(diff.T)):
        t           = torch.full((n_samples,), step, device=device, dtype=torch.long)
        eps_pred    = model(xt, t)
        beta_t      = diff.betas[step]
        alpha_cum_t = diff.alphas_cumprod[step]
        coef        = beta_t / torch.sqrt(1.0 - alpha_cum_t)
        mu_theta    = (1.0 / torch.sqrt(1.0 - beta_t)) * (xt - coef * eps_pred)
        if step > 0:
            pv = diff.posterior_variance(step)
            xt = mu_theta + math.sqrt(pv) * torch.randn_like(xt)
        else:
            xt = mu_theta

    return xt.cpu()


def decode_samples(
    generated: torch.Tensor,
    model: FinDiffSynthesizer,
    categorical_cols: List[str],
    numeric_cols: List[str],
    label_encoders: Dict[str, LabelEncoder],
    quantile: QuantileTransformer,
    label_col: str = "default",
) -> pd.DataFrame:
    """Map flat embeddings back to a human-readable DataFrame."""
    n, emb_dim   = generated.shape[0], model.cat_emb_dim
    cat_flat     = generated[:, :model.total_cat_dim]
    num_flat     = generated[:, model.total_cat_dim:]

    gen_cat = np.zeros((n, len(categorical_cols)), dtype=int)
    for i, col in enumerate(categorical_cols):
        start  = i * emb_dim
        g_emb  = cat_flat[:, start:start + emb_dim].numpy()
        w      = model.embeddings[col].weight.detach().cpu()
        dists  = torch.cdist(torch.tensor(g_emb, dtype=torch.float32), w).numpy()
        gen_cat[:, i] = np.argmin(dists, axis=1)

    gen_num = quantile.inverse_transform(num_flat.numpy()) if numeric_cols \
              else np.zeros((n, 0))

    out = pd.DataFrame(gen_num, columns=numeric_cols)
    for i, col in enumerate(categorical_cols):
        out[col] = label_encoders[col].inverse_transform(gen_cat[:, i])
    out = out[categorical_cols + numeric_cols]
    try:
        out[label_col] = out[label_col].astype(int)
    except Exception:
        pass
    return out


print("Sampling & decoding functions defined.")


def plot_loss_curves(loss_histories: Dict[float, List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for lam, losses in loss_histories.items():
        label = f"λ_sf={lam}" if lam > 0 else "λ_sf=0 (vanilla FinDiff)"
        ax.plot(losses, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
    ax.set_title("FinDiff Training Loss vs Structural Fidelity Regularisation")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curves.png", dpi=150)
    plt.show()


# ── Global Utility (TabStruct §3.3) ───────────────────────────────────────────
def compute_global_utility(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols: List[str],
) -> Dict[str, float]:
    """
    TabStruct Global Utility (Eq. 4):

        For each variable x_j:
          - train predictor on synthetic (all other features → x_j)
          - evaluate on real test data
          - normalise against the reference predictor trained on real_train

        Utility_j = perf_syn / perf_ref   (categorical, higher acc = better)
                  = perf_ref / perf_syn   (numeric, lower RMSE = better)

        Global Utility = mean(Utility_j)  over all j

    Fix (v2):
        • Cast categorical columns to str before OrdinalEncoder fit/transform.
          The Taiwan dataset stores categoricals as integers (SEX=1/2,
          PAY_0=-2..8, etc.).  sklearn's OrdinalEncoder internally calls
          np.isnan() when checking for unknowns, which raises TypeError on
          integer arrays → fix is to always work with str dtype.
        • enc.fit() is now called ONCE per target (outside _prep) instead of
          being re-called on every _prep() invocation.
    """
    results = {}
    all_cols = list(real_test.columns)

    for target in all_cols:
        feat_cols = [c for c in all_cols if c != target]
        cat_feat  = [c for c in categorical_cols if c != target]

        # ── Fit encoder ONCE on synthetic data (str cast fixes TypeError) ────
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if cat_feat:
            enc.fit(synth_df[cat_feat].astype(str))

        def _prep(df_):
            X = df_[feat_cols].copy()
            if cat_feat:
                # cast to str so OrdinalEncoder never sees integer arrays
                X[cat_feat] = enc.transform(X[cat_feat].astype(str))
            return X.astype(float)

        X_te     = _prep(real_test);  y_te  = real_test[target]
        X_syn_tr = _prep(synth_df);   y_syn = synth_df[target]
        X_ref_tr = _prep(real_train); y_ref = real_train[target]
        is_cat   = target in categorical_cols

        def _score(X_tr, y_tr):
            if is_cat:
                m = LGBMClassifier(n_estimators=100, learning_rate=0.05,
                                    random_state=SEED, verbose=-1)
                m.fit(X_tr, y_tr)
                return float(accuracy_score(y_te, m.predict(X_te)))
            else:
                m = LGBMRegressor(n_estimators=100, learning_rate=0.05,
                                    random_state=SEED, verbose=-1)
                m.fit(X_tr, y_tr)
                return float(np.sqrt(mean_squared_error(y_te, m.predict(X_te))))
        try:
            perf_syn = _score(X_syn_tr, y_syn)
            perf_ref = _score(X_ref_tr, y_ref)
            util = (perf_syn / (perf_ref + 1e-8)) if is_cat \
                   else ((perf_ref + 1e-8) / (perf_syn + 1e-8))
        except Exception as e:
            print(f"  [WARN] global_utility target={target}: {e}")
            util = 0.0
        results[target] = float(util)

    results["global_utility"] = float(np.mean(list(results.values())))
    return results

all_results:    List[Dict]                = []
loss_histories: Dict[float, List[float]] = {}
synth_dfs:      Dict[float, pd.DataFrame] = {}
global_utility_by_lambda: Dict[float, float] = {}

# Feedback signal: use previous run's global utility inside current SF loss.
prev_global_utility: Optional[float] = None
LAMBDA_GLOBAL_UTILITY = 0.2

for lam in LAMBDA_SF_VALUES:
    print(f"\n{'#'*60}")
    print(f"#  λ_sf = {lam}")
    print(f"{'#'*60}")

    # ── build a fresh model for this λ ─────────────────────────────────────
    model = FinDiffSynthesizer(
        categorical_cols=categorical_cols,
        vocab_sizes=vocab_sizes,
        num_continuous=len(numeric_cols),
        cat_emb_dim=EMB_DIM,
        hidden_dims=HIDDEN_DIMS,
    ).to(device)
    diff = DiffusionHelper(betas)

    # ── train ───────────────────────────────────────────────────────────────
    losses = train_findiff(train_loader, model, diff,
                           epochs=EPOCHS, lr=LR, lambda_sf=lam,
                           global_utility=prev_global_utility,
                           lambda_global_utility=LAMBDA_GLOBAL_UTILITY)
    loss_histories[lam] = losses

    ckpt = f"{OUT_DIR}/model_lsf{lam:.2f}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Model saved → {ckpt}")

    # ── generate synthetic data ─────────────────────────────────────────────
    print(f"Generating {N_GENERATE} samples...")
    gen_tensor = sample_loop(N_GENERATE, model, diff)
    out_df     = decode_samples(
        gen_tensor, model, categorical_cols, numeric_cols,
        label_encoders, quantile, label_col=label_col
    )
    csv_path = f"{OUT_DIR}/synthetic_lsf{lam:.2f}.csv"
    out_df.to_csv(csv_path, index=False)
    synth_dfs[lam] = out_df
    print(f"Synthetic data saved → {csv_path}")
    print(out_df.head(3))

    # ── compute global utility and feed into next run ──────────────────────
    utility_dict = compute_global_utility(
        real_train=real_train,
        real_test=real_test,
        synth_df=out_df,
        categorical_cols=categorical_cols,
    )
    prev_global_utility = float(utility_dict["global_utility"])
    global_utility_by_lambda[lam] = prev_global_utility
    all_results.append({
        "lambda_sf": float(lam),
        "global_utility": prev_global_utility,
    })
    print(f"Global utility (λ_sf={lam:.2f}): {prev_global_utility:.4f}")

pd.DataFrame(all_results).to_csv(f"{OUT_DIR}/global_utility_feedback.csv", index=False)