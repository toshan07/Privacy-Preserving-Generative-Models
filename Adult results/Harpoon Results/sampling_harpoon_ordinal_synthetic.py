import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import Preprocessor
from model import MLPDiffusion
from utils import calc_diffusion_hyperparams

warnings.filterwarnings('ignore')

ADULT_COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
    'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

DEFAULT_COLUMN_NAMES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3',
    'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
    'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
    'PAY_AMT5', 'PAY_AMT6', 'default'
]

def build_device(gpu_index: int) -> str:
    if gpu_index != -1 and torch.cuda.is_available():
        return f'cuda:{gpu_index}'
    return 'cpu'

def encode_target_column(prepper: Preprocessor, df: pd.DataFrame):
    target_idx = prepper.info['target_col_idx'][0]
    target_col = prepper.df.columns[target_idx]
    target_series = prepper.df.iloc[:, target_idx]

    if pd.api.types.is_numeric_dtype(target_series):
        numeric_values = pd.to_numeric(target_series, errors='coerce').to_numpy()
        unique_vals = np.unique(numeric_values[~np.isnan(numeric_values)])
        is_integer_like = np.allclose(unique_vals, np.round(unique_vals))
        if is_integer_like and len(unique_vals) <= 20:
            target_categories = np.sort(unique_vals)
            target_codes = pd.Categorical(
                pd.to_numeric(df.iloc[:, target_idx], errors='coerce'),
                categories=target_categories
            ).codes.astype(np.float32)
            if np.any(target_codes < 0):
                raise ValueError('Found unknown numeric target labels while encoding target column.')
            return target_codes, target_col, target_categories

        return pd.to_numeric(df.iloc[:, target_idx], errors='coerce').astype(np.float32).to_numpy(), target_col, None

    target_categories = pd.unique(target_series)
    target_codes = pd.Categorical(df.iloc[:, target_idx], categories=target_categories).codes.astype(np.float32)
    if np.any(target_codes < 0):
        raise ValueError('Found unknown target labels while encoding target column.')
    return target_codes, target_col, target_categories


def decode_target_column(target_values: np.ndarray, target_categories, target_codes_train=None):
    if target_categories is None:
        return target_values

    if target_codes_train is not None and len(target_categories) == 2:
        positive_ratio = np.mean(target_codes_train == 1)
        num_rows = len(target_values)
        num_positive = int(np.round(positive_ratio * num_rows))
        num_positive = max(0, min(num_rows, num_positive))

        ranked_indices = np.argsort(target_values)
        class_indices = np.zeros(num_rows, dtype=int)
        if num_positive > 0:
            class_indices[ranked_indices[-num_positive:]] = 1
        return np.asarray(target_categories)[class_indices]

    indices = np.clip(np.rint(target_values), 0, len(target_categories) - 1).astype(int)
    return np.asarray(target_categories)[indices]


def prettify_columns(dataname: str, columns: list):
    are_numeric_headers = all(str(col).isdigit() for col in columns)
    if dataname == 'adult' and are_numeric_headers and len(columns) == len(ADULT_COLUMN_NAMES):
        return ADULT_COLUMN_NAMES.copy()
    if dataname == 'default' and are_numeric_headers and len(columns) == len(DEFAULT_COLUMN_NAMES):
        return DEFAULT_COLUMN_NAMES.copy()
    return columns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unconditional synthetic sampling with Harpoon ordinal model')
    parser.add_argument('--dataname', type=str, default='adult', help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index, -1 for CPU')
    parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension of MLPDiffusion')
    parser.add_argument('--timesteps', type=int, default=200, help='Number of reverse diffusion steps')
    parser.add_argument('--beta_0', type=float, default=0.0001, help='Initial beta in diffusion schedule')
    parser.add_argument('--beta_T', type=float, default=0.02, help='Final beta in diffusion schedule')
    parser.add_argument('--num_samples', type=int, default=45000, help='How many synthetic rows to generate; 0 uses train size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='', help='Optional output csv path')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = build_device(args.gpu)
    prepper = Preprocessor(args.dataname)

    train_X_base = prepper.encodeDf('Ordinal', prepper.df_train)
    target_codes_train, target_col_name, target_categories = encode_target_column(prepper, prepper.df_train)
    train_X = np.concatenate([train_X_base, target_codes_train.reshape(-1, 1)], axis=1)
    in_dim = train_X.shape[1]

    mean_X = np.mean(train_X, axis=0)
    std_X = np.std(train_X, axis=0)
    std_X = np.clip(std_X, 1e-6, None)

    num_samples = args.num_samples if args.num_samples > 0 else train_X.shape[0]

    diffusion_config = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_T)

    model_path = f'saved_models/{args.dataname}/harpoon_ordinal_with_target.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model checkpoint not found at {model_path}. '
            f'Train first with train_harpoon_ordinal.py.'
        )

    model = MLPDiffusion(in_dim, args.hid_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    x_t = torch.randn((num_samples, in_dim), dtype=torch.float32, device=device)

    with torch.no_grad():
        for t in tqdm(range(args.timesteps - 1, -1, -1), desc='Unconditional sampling'):
            timesteps_tensor = torch.full((num_samples,), t, device=device)

            alpha_t = diffusion_config['Alpha'][t].to(device)
            alpha_bar_t = diffusion_config['Alpha_bar'][t].to(device)
            alpha_bar_t_1 = (
                diffusion_config['Alpha_bar'][t - 1].to(device)
                if t >= 1 else torch.tensor(1.0, device=device)
            )

            sigmas_predicted = model(x_t, timesteps_tensor)

            x_t = (
                (x_t / torch.sqrt(alpha_t))
                - ((1 - alpha_t) / (torch.sqrt(alpha_t) * torch.sqrt(1 - alpha_bar_t))) * sigmas_predicted
            )

            if t > 0:
                noise = torch.normal(0, 1, size=x_t.shape, device=device)
                vari = (1 - alpha_t) * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * noise
                x_t += vari

    x_sample = x_t.cpu().numpy()
    x_sample = (x_sample * std_X) + mean_X
    decoded_features = prepper.decodeNp('Ordinal', x_sample[:, :-1])
    decoded_target = decode_target_column(x_sample[:, -1], target_categories, target_codes_train)

    original_cols = list(prepper.df.columns)
    syn_df = pd.DataFrame(index=range(num_samples), columns=original_cols, dtype=object)

    num_idx = prepper.info['num_col_idx']
    cat_idx = prepper.info['cat_col_idx']
    n_num = len(num_idx)

    for pos, col_idx in enumerate(num_idx):
        syn_df.iloc[:, col_idx] = decoded_features[:, pos]
    for pos, col_idx in enumerate(cat_idx):
        syn_df.iloc[:, col_idx] = decoded_features[:, n_num + pos]
    syn_df.iloc[:, prepper.info['target_col_idx'][0]] = decoded_target

    output_cols = prettify_columns(args.dataname, original_cols)
    syn_df.columns = output_cols
    numeric_output_cols = [output_cols[i] for i in num_idx]
    for col in numeric_output_cols:
        syn_df[col] = pd.to_numeric(syn_df[col], errors='coerce')

    target_output_col = output_cols[prepper.info['target_col_idx'][0]]
    if target_categories is None:
        syn_df[target_output_col] = pd.to_numeric(syn_df[target_output_col], errors='coerce')
    else:
        target_cat_numeric = pd.to_numeric(pd.Series(target_categories), errors='coerce')
        if target_cat_numeric.notna().all() and np.allclose(target_cat_numeric.to_numpy(), np.round(target_cat_numeric.to_numpy())):
            syn_df[target_output_col] = pd.to_numeric(syn_df[target_output_col], errors='coerce').round().astype('Int64')

    out_dir = 'experiments/synthetic_samples'
    os.makedirs(out_dir, exist_ok=True)

    out_path = args.output if args.output else f'{out_dir}/harpoon_ordinal_{args.dataname}_n{num_samples}.csv'
    syn_df.to_csv(out_path, index=False)

    real_ref_path = f'{out_dir}/real_{args.dataname}.csv'
    if not os.path.exists(real_ref_path):
        real_df = prepper.df.copy()
        real_df.columns = output_cols
        real_df.to_csv(real_ref_path, index=False)
    print(f'Saved synthetic samples to: {out_path}')
    print(f'Rows: {len(syn_df)}, Columns: {len(syn_df.columns)}')