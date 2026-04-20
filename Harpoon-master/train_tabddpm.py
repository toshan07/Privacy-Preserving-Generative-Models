import os, math, random, argparse, json
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import MLPDiffusion
# -------- robust local import of your diffusion class ----------
from pathlib import Path
import sys
import importlib.util
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PKG_DIR = BASE_DIR / "tabddpm"
MOD_PATH = PKG_DIR / "gaussian_multinomial_diffusionv1.py"

if not PKG_DIR.exists():
    raise RuntimeError(f"Expected package folder missing: {PKG_DIR}")
if not MOD_PATH.exists():
    raise RuntimeError(f"Expected module file missing: {MOD_PATH}")

# Try package import first, then absolute-path fallback
sys.path.insert(0, str(BASE_DIR))
try:
    from tabddpm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location("tabddpm.gaussian_multinomial_diffusion", str(MOD_PATH))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    GaussianMultinomialDiffusion = mod.GaussianMultinomialDiffusion
# ---------------------------------------------------------------

from dataset import Preprocessor  # your dataset.py


# -------------------- utils --------------------
def seed_everything(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# @torch.no_grad()
# def update_ema(ema_params, model_params, decay=0.999):
#     for e, p in zip(ema_params, model_params):
#         e.data.mul_((decay)).add_(p.data, alpha=(1.0 - decay))
#
#
# class SinusoidalTimestep(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, t: torch.Tensor):
#         half = self.dim // 2
#         freqs = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
#         args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
#         emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
#         if self.dim % 2 == 1:
#             emb = torch.nn.functional.pad(emb, (0, 1))
#         return emb
#
#
# class DenoiserMLP(nn.Module):
#     """Takes [x_num_t || log_x_cat_t] (dim = num_numeric + sum(K)), plus timestep, optional class y."""
#
#     def __init__(self, d_in: int, d_hidden: int = 1024, n_layers: int = 4,
#                  t_embed_dim: int = 128, y_num_classes: Optional[int] = None):
#         super().__init__()
#         self.time_emb = SinusoidalTimestep(t_embed_dim)
#         self.y_embed = None
#         y_dim = 0
#         if y_num_classes and y_num_classes > 0:
#             self.y_embed = nn.Embedding(y_num_classes, t_embed_dim)
#             y_dim = t_embed_dim
#         layers = []
#         d_first = d_in + t_embed_dim + y_dim
#         layers += [nn.Linear(d_first, d_hidden), nn.ReLU()]
#         for _ in range(n_layers - 1):
#             layers += [nn.Linear(d_hidden, d_hidden), nn.ReLU()]
#         layers += [nn.Linear(d_hidden, d_in)]
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None):
#         te = self.time_emb(t)
#         if self.y_embed is not None and y is not None:
#             ye = self.y_embed(y)
#             h = torch.cat([x, te, ye], dim=1)
#         else:
#             h = torch.cat([x, te], dim=1)
#         return self.net(h)


# -------------------- config --------------------
@dataclass
class CFG:
    dataname: str = "adult"
    batch_size: int = 256
    epochs: int = 5
    lr: float = 2e-3
    weight_decay: float = 1e-4
    timesteps: int = 200
    scheduler: str = "cosine"
    gaussian_loss_type: str = "mse"
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    d_hidden: int = 1024
    n_layers: int = 4
    seed: int = 42
    deterministic: bool = True

    # device / runtime
    device: str = "auto"  # auto | cpu | cuda
    gpu: int = 0
    amp: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    dry_run_steps: int = 0  # e.g. 10 to smoke-test on CPU

    # conditional
    y_num_classes: int = 0

    # fine-tune
    ckpt_path: Optional[str] = None


# -------------------- helpers --------------------
def get_K_from_preproc(pre: Preprocessor) -> List[int]:
    # Length per categorical column (works via either encoder's categories_)
    if hasattr(pre, "OrdinalEncoder") and hasattr(pre.OrdinalEncoder, "categories_"):
        return [len(c) for c in pre.OrdinalEncoder.categories_]
    return [len(c) for c in pre.OneHotEncoder.categories_]


def make_norm_stats(X_ord: np.ndarray, num_numeric: int):
    """Only normalize numericals; leave categorical indices untouched."""
    mean = X_ord[:, :num_numeric].mean(0)
    std = X_ord[:, :num_numeric].std(0)
    std = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def pick_device(cfg: CFG) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cfg.gpu}")
        print("[INFO] Requested CUDA but not available; falling back to CPU.")
        return torch.device("cpu")
    # auto
    return torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")


# -------------------- training --------------------
def main(cfg: CFG):
    seed_everything(cfg.seed, cfg.deterministic)
    device = pick_device(cfg)
    print(f"[INFO] Using device: {device}")

    # ---- data (IMPORTANT: use Ordinal so x_cat are integer indices) ----
    pre = Preprocessor(cfg.dataname)
    X_train_ohe = pre.encodeDf('OHE', pre.df_train)  # [nums | cat_indices]
    num_numeric = pre.numerical_indices_np_end
    K = get_K_from_preproc(pre)
    d_in = int(num_numeric + sum(K))

    # normalize numericals only
    mean, std = make_norm_stats(X_train_ohe, num_numeric)
    X_train_np = X_train_ohe.copy()
    X_train_np[:, :num_numeric] = (X_train_np[:, :num_numeric] - mean) / std

    # tensorize; categories stay as integers inside the float tensor and will be cast to long in diffusion
    X_train = torch.tensor(X_train_np, dtype=torch.float32)

    ds = TensorDataset(X_train)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # ---- model & diffusion ----
    # model = DenoiserMLP(
    #     d_in=d_in,
    #     d_hidden=cfg.d_hidden,
    #     n_layers=cfg.n_layers,
    #     t_embed_dim=128,
    #     y_num_classes=(cfg.y_num_classes if cfg.y_num_classes > 0 else None),
    # ).to(device)
    model = MLPDiffusion(d_in, cfg.d_hidden).to(device)

    # if cfg.ckpt_path and os.path.isfile(cfg.ckpt_path):
    #     sd = torch.load(cfg.ckpt_path, map_location="cpu")
    #     missing, unexpected = model.load_state_dict(sd, strict=False)
    #     print(f"[FT] loaded {cfg.ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")

    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array(K) if len(K) > 0 else np.array([0]),
        num_numerical_features=num_numeric,
        denoise_fn=model,
        gaussian_loss_type=cfg.gaussian_loss_type,
        num_timesteps=cfg.timesteps,
        scheduler=cfg.scheduler,
        device=device,
    ).to(device)

    # ema_model = DenoiserMLP(
    #     d_in=d_in,
    #     d_hidden=cfg.d_hidden,
    #     n_layers=cfg.n_layers,
    #     t_embed_dim=128,
    #     y_num_classes=(cfg.y_num_classes if cfg.y_num_classes > 0 else None),
    # ).to(device)
    # ema_model = MLPDiffusion(d_in, cfg.d_hidden).to(device)
    # ema_model.load_state_dict(model.state_dict(), strict=True)
    # for p in ema_model.parameters():
    #     p.requires_grad_(False)

    opt = torch.optim.AdamW(diffusion.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # ---- train ----
    global_step = 0
    out_dir = f"saved_models/{cfg.dataname}"
    os.makedirs(out_dir, exist_ok=True)
    losses = []
    pbar = tqdm(range(cfg.epochs), desc='Training')
    encode_again = False
    for epoch in pbar:
        loss = 0.0
        for batch in dl:
            x = batch[0].to(device, non_blocking=cfg.pin_memory)
            out_dict = {}  # unconditional by default; if conditional, you'd add {"y": y}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                loss_multi, loss_gauss = diffusion.mixed_loss(x, out_dict, encode_again)
                loss = loss_multi + loss_gauss
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        pbar.set_postfix(loss=loss)
    torch.save(diffusion._denoise_fn.state_dict(), f"{out_dir}/tabddpm.pt")
    # # ---- quick sampling demo (works for both conditional/unconditional) ----
    # try:
    #     B = min(128, cfg.batch_size)
    #     y_dist = (torch.ones(cfg.y_num_classes, device=device) / cfg.y_num_classes) if cfg.y_num_classes > 0 else torch.ones(1, device=device)
    #     samples, _ = diffusion.sample(B, y_dist)  # your class API
    #     samples = samples.detach().cpu().numpy()
    #
    #     # denorm numericals
    #     samples[:, :num_numeric] = samples[:, :num_numeric] * std + mean
    #
    #     # decode categories back to original string values using OrdinalEncoder.categories_
    #     if len(K) > 0:
    #         cat_idx = samples[:, num_numeric:].astype(int)
    #         cat_vals = []
    #         for col, cats in enumerate(pre.OrdinalEncoder.categories_):
    #             cat_vals.append(np.array(cats, dtype=object)[cat_idx[:, col]])
    #         cat_vals = np.stack(cat_vals, axis=1)
    #         df_cols = list(pre.df.columns[pre.info['num_col_idx']]) + list(pre.df.columns[pre.info['cat_col_idx']])
    #         nums = samples[:, :num_numeric]
    #         out_df = pd.DataFrame(np.concatenate([nums, cat_vals], axis=1), columns=df_cols)
    #     else:
    #         out_df = pd.DataFrame(samples[:, :num_numeric], columns=list(pre.df.columns[pre.info['num_col_idx']]))
    #     out_df.to_csv(f"{out_dir}/tabddpm_samples_head.csv", index=False)
    #     print(f"[OK] Saved samples to {out_dir}/tabddpm_samples_head.csv")
    # except Exception as e:
    #     print(f"[WARN] Sampling demo skipped or failed: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data / model
    p.add_argument("--dataname", type=str, default="adult")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--scheduler", type=str, default="linear")
    p.add_argument("--gaussian_loss_type", type=str, default="mse")
    # p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--d_hidden", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--y_num_classes", type=int, default=0)
    p.add_argument("--ckpt_path", type=str, default=None)
    # runtime
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--dry_run_steps", type=int, default=0)
    args = p.parse_args()
    cfg = CFG(**vars(args))
    main(cfg)
