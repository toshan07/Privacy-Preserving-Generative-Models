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
parser.add_argument('--loss', type=str, default='mae', help='inference loss type')

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
    train_X = prepper.encodeDf('Ordinal', prepper.df_train)
    test_X = prepper.encodeDf('Ordinal', prepper.df_test)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (
        np.mean(train_X, axis=0), np.std(train_X, axis=0)
    )
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    # X = torch.tensor(X)
    X_test = (test_X - mean_X) / std_X

    X_test_eval = X_test.copy()
    X_test_eval[:, num_numeric:] = (X_test_eval[:, num_numeric:] * std_X[num_numeric:]) + mean_X[num_numeric:]
    X_test_eval = prepper.decodeNp('Ordinal', X_test_eval)
    test_X_ordinal_fmt = prepper.encodeDf('Ordinal', prepper.df_test)
    orig_mask = generate_mask(test_X_ordinal_fmt, mask_type=mask_type, mask_num=num_trials, p=ratio)
    diffusion_config = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_T)

    models_dir = f'saved_models/{args.dataname}/'
    model_path = os.path.join(models_dir, "harpoon_ordinal.pt")
    model = MLPDiffusion(in_dim, hid_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    MSEs, ACCs = [], []
    X_test_gpu = torch.tensor(X_test).to(device, dtype=torch.float32)
    with torch.no_grad():
        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            mask_test = torch.tensor(orig_mask[trial])
            mask_float = mask_test.float().to(device)
            x_t = torch.randn_like(torch.tensor(X_test)).to(device, dtype=torch.float32)
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
                    x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * sigmas_predicted) / torch.sqrt(alpha_bar_t)
                    normal_vec = (x_0_hat - x_t).detach()
                    loss1 = 0.0
                    if 'mae' in args.loss:
                        loss1 += torch.sum((1 - mask_float) * abs(x_0_hat - X_test_gpu), dim=1)
                    elif 'mse' in args.loss:
                        loss1 += torch.sum((1 - mask_float) * (x_0_hat - X_test_gpu)**2, dim=1)
                    cond_loss = loss1
                    grad = torch.autograd.grad(cond_loss, x_t, grad_outputs=torch.ones_like(cond_loss))[0]

                x_t = (x_t / torch.sqrt(alpha_t)) - (
                        (1 - alpha_t) / (torch.sqrt(alpha_t) * torch.sqrt(
                    1 - alpha_bar_t))) * sigmas_predicted  # denoise w/o correction

                vari = 0.0
                if t > 0:
                    vari = (1 - alpha_t) * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(0, 1,
                                                                                                    size=x_t.shape).to(
                        device)
                x_t += vari
                update = -0.2 * grad  # /torch.norm(grad, dim=1).unsqueeze(1)
                x_t += update
            X_pred = (x_t * mask_float + (1 - mask_float) * X_test_gpu).cpu().numpy()
            X_pred[:, num_numeric:] = (X_pred[:, num_numeric:] * std_X[num_numeric:]) + mean_X[num_numeric:]
            imputed_decoded = prepper.decodeNp('Ordinal', X_pred)
            mse, acc = get_eval(imputed_decoded, X_test_eval, mask_test, num_numeric)
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
               "Method": f"Harpoon_ordinal_{args.loss}",
               "Mask Type": args.mask,
               "Ratio": ratio,
               "Avg MSE": np.mean(MSEs),
               "STD of MSE": np.std(MSEs),
               "Avg Acc": np.mean(ACCs),
               "STD of Acc": np.std(ACCs)
               }
    new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(experiment_path)
