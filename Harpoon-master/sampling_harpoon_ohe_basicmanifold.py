import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm
from generate_mask import generate_mask
from model import MLPDiffusion
from dataset import Preprocessor, get_eval
from utils import calc_diffusion_hyperparams
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--timesteps', type=int, default=200, help='Number of diffusion steps.')
parser.add_argument('--beta_0', type=float, default=0.0001, help='initial variance schedule')
parser.add_argument('--beta_T', type=float, default=0.02, help='last variance schedule')
parser.add_argument('--mask', type=str, default='MAR', help='Masking mechanisms.')
parser.add_argument('--num_trials', type=int, default=5, help='Number of sampling times.')
parser.add_argument('--ratio', type=str, default="0.25", help='Masking ratio.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    dataname = args.dataname
    device = args.device
    hid_dim = args.hid_dim
    mask_type = args.mask
    ratio = float(args.ratio)
    num_trials = args.num_trials
    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('OHE', prepper.df_train)
    test_X = prepper.encodeDf('OHE', prepper.df_test)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (
    np.concatenate((np.mean(train_X[:, :num_numeric], axis=0), np.zeros(train_X.shape[1] - num_numeric)), axis=0),
    np.concatenate((np.std(train_X[:, :num_numeric], axis=0), np.ones(train_X.shape[1] - num_numeric)), axis=0))
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    X = torch.tensor(X)
    X_test = (test_X - mean_X) / std_X
    X_test = torch.tensor(X_test, dtype=torch.float32)

    test_X_ori_fmt = np.concatenate((prepper.df_test.iloc[:, prepper.info['num_col_idx']],
                                     prepper.df_test.iloc[:, prepper.info['cat_col_idx']]), axis=1)
    test_X_ordinal_fmt = prepper.encodeDf('Ordinal', prepper.df_test)
    orig_mask = generate_mask(test_X_ordinal_fmt, mask_type=mask_type, mask_num=num_trials, p=ratio)

    diffusion_config = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_T)
    test_masks = prepper.extend_mask(orig_mask, encoding='OHE')

    models_dir = f'saved_models/{args.dataname}/'
    model_path = os.path.join(models_dir, "diffputer_selfmade.pt")
    model = MLPDiffusion(in_dim, hid_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    mask_tests = torch.tensor(test_masks)
    MSEs, ACCs = [], []
    rec_Xs = []
    cossims = []
    X_test_gpu = X_test.to(device)
    with (torch.no_grad()):
        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            mask_test = mask_tests[trial]
            mask_float = mask_test.float().to(device)
            x_t = torch.randn_like(X_test).to(device)
            for t in range(args.timesteps - 1, -1, -1):
                timesteps = torch.full(size=(x_t.shape[0],), fill_value=t).to(device)
                alpha_t = diffusion_config['Alpha'][t].to(device)
                alpha_bar_t = diffusion_config['Alpha_bar'][t].to(device)
                alpha_bar_t_1 = diffusion_config['Alpha_bar'][t - 1].to(device) if t >= 1 else torch.tensor(1).to(
                    device)
                sigma_t = diffusion_config['Sigma'][t].to(device)
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    sigmas_predicted = model(x_t, timesteps)
                    x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * sigmas_predicted)/torch.sqrt(alpha_bar_t)
                    normal_vec = (x_0_hat - x_t).detach()
                    cond_loss = torch.sum((1 - mask_float) * (x_0_hat - X_test_gpu)**2)
                    grad = torch.autograd.grad(cond_loss, x_t, grad_outputs=torch.ones_like(cond_loss), retain_graph=True)[0]
                    grad_sigmas = torch.autograd.grad(sigmas_predicted, x_t, grad_outputs=torch.ones_like(sigmas_predicted))[0]

                # grad_tangent = grad - (torch.sum(grad * normal_vec) / torch.sum(normal_vec * normal_vec)) * normal_vec
                x_t = (x_t / torch.sqrt(alpha_t)) - (
                    (1 - alpha_t) / (torch.sqrt(alpha_t) * torch.sqrt(1 - alpha_bar_t))) * sigmas_predicted  # denoise w/o correction

                x_t += diffusion_config['Sigma'][t] * torch.randn_like(x_t)  # stochasticity
                update = -0.1 * grad
                x_t += update
                correction = (1.0/torch.sqrt(alpha_t) - (1-alpha_t)/(torch.sqrt(alpha_t) * torch.sqrt(1-alpha_bar_t)) * grad_sigmas) * update
                x_t += correction

                # print(f"cond loss: {cond_loss.cpu().numpy()}")
                # x_t -= 0.35 * grad  # correction term
                # x_t -= 0.25 * grad_tangent

                # x_cond_t = torch.sqrt(alpha_bar_t_1) * X_test_gpu + torch.sqrt(1-alpha_bar_t_1) * torch.randn_like(X_test_gpu)
                # x_t = (1-mask_float) * x_cond_t + mask_float * x_t
            # plt.plot(cossims, c='green')
            # plt.show()
            # exit()
            X_pred = (x_t * mask_float + (1-mask_float) * X_test_gpu).cpu().numpy()
            X_true = X_test.numpy()
            X_true_dec = prepper.decodeNp(scheme='OHE', arr=X_true)
            X_pred_dec = prepper.decodeNp(scheme='OHE', arr=X_pred)
            mse, acc = get_eval(X_pred_dec, X_true_dec, orig_mask[trial], num_numeric)
            MSEs.append(mse)
            ACCs.append(acc)

    MSEs = np.array(MSEs)
    ACCs = np.array(ACCs)
    experiment_path = f'experiments/imputation.csv'
    directory = os.path.dirname(experiment_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(experiment_path):
        columns = [
            "Dataset",
            "Method",
            "Mask Type",
            "Ratio",
            "Avg MSE",
            "STD of MSE",
            "Avg Acc",
            "STD of Acc"
        ]
        exp_df = pd.DataFrame(columns=columns)
    else:
        exp_df = pd.read_csv(experiment_path).drop(columns=['Unnamed: 0'])

    new_row = {"Dataset": dataname,
               "Method": "harpoon_basicmanifold_ohe",
               "Mask Type": args.mask,
               "Ratio": ratio,
               "Avg MSE": np.mean(MSEs),
               "STD of MSE": np.std(MSEs),
               "Avg Acc": np.mean(ACCs),
               "STD of Acc": np.std(ACCs)
               }
    new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(experiment_path)
