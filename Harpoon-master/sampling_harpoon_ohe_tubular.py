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
from sampling_harpoon_ohe_basicmanifold_kld import computeCatLoss
from timeit import default_timer as timer

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
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
    parser.add_argument('--runtime_test', type=bool, default=False, help='store runtime?')
    parser.add_argument('--schedule', type=bool, default=False, help="use linear eta schedule")
    parser.add_argument('--ignore_hard_fix', type=bool, default=False, help="skip injecting hard constraints at the end")

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
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
    exec_times = []
    rec_Xs = []
    cossim_grad = []
    cossim_gradtan = []
    X_test_gpu = X_test.to(device)
    with (torch.no_grad()):
        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            mask_test = mask_tests[trial]
            mask_float = mask_test.float().to(device)
            x_t = torch.randn_like(X_test).to(device)
            start = timer()
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
                        loss1 += torch.sum((1 - mask_float) * (x_0_hat - X_test_gpu) ** 2, dim=1)
                    loss2 = 0.0
                    if 'kld' in args.loss:
                        loss2 += computeCatLoss(x_0_hat, X_test_gpu, num_numeric, prepper.OneHotEncoder.categories_,
                                                mask_float)
                    # loss2 = torch.sum((1 - mask_float) * abs(x_0_hat - X_test_gpu) ** 1, dim=1)
                    cond_loss = loss1 + loss2
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
                scale = -0.2 * ((args.timesteps - 1 - t) / args.timesteps) if args.schedule else -0.2
                update = scale * grad  # /torch.norm(grad, dim=1).unsqueeze(1)
                x_t += update
            end = timer()
            diff = end - start
            exec_times.append(diff)
            if args.ignore_hard_fix:
                X_pred = x_t.cpu().numpy()
            else:
                X_pred = (x_t * mask_float + (1 - mask_float) * X_test_gpu).cpu().numpy()
            X_true = X_test.numpy()
            X_true_dec = prepper.decodeNp(scheme='OHE', arr=X_true)
            X_pred_dec = prepper.decodeNp(scheme='OHE', arr=X_pred)
            if args.ignore_hard_fix:
                mse, acc = get_eval(X_pred_dec, X_true_dec, ~orig_mask[trial], num_numeric)
            else:
                mse, acc = get_eval(X_pred_dec, X_true_dec, orig_mask[trial], num_numeric)
            MSEs.append(mse)
            ACCs.append(acc)

    MSEs = np.array(MSEs)
    ACCs = np.array(ACCs)
    arr_time = np.array(exec_times)
    method_str = f'harpoon_ohe_{args.loss}_linear' if args.schedule else f'harpoon_ohe_{args.loss}'
    if args.ignore_hard_fix:
        method_str += 'softfix'
    if args.runtime_test:
        experiment_path = f'experiments/runtime.csv'
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
                "STD of Acc",
                "Avg Runtime",
                "STD of Runtime"
            ]
            exp_df = pd.DataFrame(columns=columns)
        else:
            exp_df = pd.read_csv(experiment_path).drop(columns=['Unnamed: 0'])

        new_row = {"Dataset": dataname,
                   "Method": method_str,
                   "Mask Type": args.mask,
                   "Ratio": ratio,
                   "Avg MSE": np.mean(MSEs),
                   "STD of MSE": np.std(MSEs),
                   "Avg Acc": np.mean(ACCs),
                   "STD of Acc": np.std(ACCs),
                   "Avg Runtime": np.mean(arr_time),
                   "STD of Runtime": np.std(arr_time)
                   }
        new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
        new_df.to_csv(experiment_path)
        exit()
    experiment_path = f'experiments/imputation.csv'
    # experiment_path = f'experiments/extremeconstraint.csv'
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
               "Method": method_str,
               "Mask Type": args.mask,
               "Ratio": ratio,
               "Avg MSE": np.mean(MSEs),
               "STD of MSE": np.std(MSEs),
               "Avg Acc": np.mean(ACCs),
               "STD of Acc": np.std(ACCs)
               }
    new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(experiment_path)
