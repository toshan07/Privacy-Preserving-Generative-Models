import os

import pandas as pd
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm

from model import MLPDiffusion, Model
from dataset import Preprocessor, get_eval
from diffusion_utils import sample_step, impute_mask
from generate_mask import generate_mask

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser(description='Missing Value Imputation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps.')
    parser.add_argument('--mask', type=str, default='MAR', help='Masking mechanisms.')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of sampling times.')
    parser.add_argument('--ratio', type=str, default="0.25", help='Masking ratio.')

    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    dataname = args.dataname
    device = args.device
    hid_dim = args.hid_dim
    num_steps = args.num_steps
    mask_type = args.mask
    ratio = float(args.ratio)
    num_trials = args.num_trials
    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('OHE', prepper.df_train)
    test_X = prepper.encodeDf('OHE', prepper.df_test)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (np.concatenate((np.mean(train_X[:, :num_numeric], axis=0),
                                     np.zeros(train_X.shape[1] - num_numeric)), axis=0),
                     np.concatenate((np.std(train_X[:, :num_numeric], axis=0),
                                     np.ones(train_X.shape[1] - num_numeric)), axis=0))

    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X

    X = torch.tensor(X)
    X_test = (test_X - mean_X) / std_X
    dimmax = np.max(X_test[:, :num_numeric])
    X_test = torch.tensor(X_test)

    test_X_ori_fmt = np.concatenate((prepper.df_test.iloc[:, prepper.info['num_col_idx']],
                                     prepper.df_test.iloc[:, prepper.info['cat_col_idx']]), axis=1)
    test_X_ordinal_fmt = prepper.encodeDf('Ordinal', prepper.df_test)
    orig_mask = generate_mask(test_X_ordinal_fmt, mask_type=mask_type, mask_num=num_trials, p=ratio)

    test_masks = prepper.extend_mask(orig_mask, encoding='OHE')
    models_dir = f'saved_models/{args.dataname}/'
    model_path = os.path.join(models_dir, "diffputer.pt")
    denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    mask_tests = torch.tensor(test_masks)
    MSEs, ACCs = [], []
    rec_Xs = []
    for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
        mask_test = mask_tests[trial]
        X_miss = (1. - mask_test.float()) * X_test
        X_miss = X_miss.to(device)
        impute_X = X_miss
        # ==========================================================
        net = model.denoise_fn_D

        num_samples, dim = X_test.shape[0], X_test.shape[1]
        X_pred = impute_mask(net, impute_X, mask_test, num_samples, dim, num_steps, device)

        mask_int = mask_test.to(torch.float).to(device)
        X_pred = X_pred * mask_int + impute_X * (1 - mask_int)
        X_pred = X_pred.cpu().numpy()
        # rec_Xs.append(rec_X)
        X_true = X_test.cpu().numpy()
        X_true_dec = prepper.decodeNp(scheme='OHE', arr=X_true)
        X_pred_dec = prepper.decodeNp(scheme='OHE', arr=X_pred)
        mse, acc = get_eval(X_pred_dec, X_true_dec, orig_mask[trial], num_numeric)
        MSEs.append(mse)
        ACCs.append(acc)
    # MSEs = np.array(MSEs)
    # ACCs = np.array(ACCs)
    # experiment_path = f'experiments/imputation.csv'
    # directory = os.path.dirname(experiment_path)
    # if directory and not os.path.exists(directory):
    #     os.makedirs(directory)
    # if not os.path.exists(experiment_path):
    #     columns = [
    #         "Dataset",
    #         "Method",
    #         "Mask Type",
    #         "Ratio",
    #         "Avg MSE",
    #         "STD of MSE",
    #         "Avg Acc",
    #         "STD of Acc"
    #     ]
    #     exp_df = pd.DataFrame(columns=columns)
    # else:
    #     exp_df = pd.read_csv(experiment_path).drop(columns=['Unnamed: 0'])
    #
    # new_row = {"Dataset": dataname,
    #            "Method": "DiffPuter",
    #            "Mask Type": args.mask,
    #            "Ratio": ratio,
    #            "Avg MSE": np.mean(MSEs),
    #            "STD of MSE": np.std(MSEs),
    #            "Avg Acc": np.mean(ACCs),
    #            "STD of Acc": np.std(ACCs)
    #            }
    # new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
    # new_df.to_csv(experiment_path)
