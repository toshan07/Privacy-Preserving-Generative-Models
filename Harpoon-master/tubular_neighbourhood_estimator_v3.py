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
from matplotlib import pyplot as plt
from sampling_harpoon_ohe_basicmanifold_kld import computeCatLoss

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--timesteps', type=int, default=200, help='Number of diffusion steps.')
parser.add_argument('--beta_0', type=float, default=0.0001, help='initial variance schedule')
parser.add_argument('--beta_T', type=float, default=0.02, help='last variance schedule')

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
    diffusion_config = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_T)
    models_dir = f'saved_models/{args.dataname}/'
    model_path = os.path.join(models_dir, "diffputer_selfmade.pt")
    model = MLPDiffusion(in_dim, hid_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    X_batch = X[:100, :].to(device, dtype=torch.float32)

    avg_angles_with_KLD = []
    std_angles_with_KLD = []
    avg_angles_no_KLD = []
    std_angles_no_KLD = []
    avg_angles_MAE = []
    std_angles_MAE = []
    avg_angles_MAE_CE = []
    std_angles_MAE_CE = []

    with torch.no_grad():
        for t in range(args.timesteps - 1, -1, -1):
            timesteps = torch.full(size=(X_batch.shape[0],), fill_value=t).to(device)
            alpha_t = diffusion_config['Alpha'][t].to(device)
            alpha_bar_t = diffusion_config['Alpha_bar'][t].to(device)
            alpha_bar_t_1 = diffusion_config['Alpha_bar'][t - 1].to(device) if t >= 1 else torch.tensor(1).to(
                device)
            sigmas = torch.normal(0, 1, size=X_batch.shape).to(device)
            """Forward noising"""
            coeff_1 = torch.sqrt(alpha_bar_t)
            coeff_2 = torch.sqrt(1 - alpha_bar_t)
            batch_noised = coeff_1 * X_batch + coeff_2 * sigmas
            mask = torch.ones(batch_noised.shape)
            mask[:, num_numeric:] = 0.0
            mask = mask.to(device)
            with torch.enable_grad():
                batch_noised.requires_grad_(True)
                batch_noised = batch_noised.to(device)
                sigmas_predicted = model(batch_noised, timesteps)
                x_0_hats = (batch_noised - torch.sqrt(1 - alpha_bar_t) * sigmas_predicted) / torch.sqrt(alpha_bar_t)
                loss1 = torch.sum(((x_0_hats - X_batch)**2))
                loss2 = torch.sum(computeCatLoss(x_0_hats, X_batch, num_numeric, prepper.OneHotEncoder.categories_, mask=mask))
                loss3 = torch.sum((abs(x_0_hats - X_batch)))
                cond_loss = loss1 + loss2
                cond_loss_no_KLD = loss1
                cond_loss_MAE = loss3
                cond_loss_MAE_CE = loss3 + loss2
                grads_v1 = torch.autograd.grad(cond_loss, batch_noised, grad_outputs=torch.ones_like(cond_loss), retain_graph=True)[0]
                grads_v2 = torch.autograd.grad(cond_loss_no_KLD, batch_noised, grad_outputs=torch.ones_like(cond_loss_no_KLD), retain_graph=True)[0]
                grads_v3 = torch.autograd.grad(cond_loss_MAE, batch_noised, grad_outputs=torch.ones_like(cond_loss_MAE), retain_graph=True)[0]
                grads_v4 = torch.autograd.grad(cond_loss_MAE_CE, batch_noised, grad_outputs=torch.ones_like(cond_loss_MAE_CE))[0]

            # vectors = x_0_hats - batch_noised
            vectors = torch.sqrt(1-alpha_bar_t) * sigmas_predicted
            normed_vectors = vectors / (vectors.norm(dim=1, keepdim=True))
            angles_with_KLD = torch.rad2deg(torch.acos(torch.sum(grads_v1 * vectors, dim=1)/(torch.norm(grads_v1, dim=1) * torch.norm(vectors, dim=1))))
            angles_no_KLD = torch.rad2deg(torch.acos(torch.sum(grads_v2 * vectors, dim=1)/(torch.norm(grads_v2, dim=1) * torch.norm(vectors, dim=1))))
            angles_MAE = torch.rad2deg(torch.acos(
                torch.sum(grads_v3 * vectors, dim=1) / (torch.norm(grads_v3, dim=1) * torch.norm(vectors, dim=1))))
            angles_MAE_CE = torch.rad2deg(torch.acos(
                torch.sum(grads_v4 * vectors, dim=1) / (torch.norm(grads_v4, dim=1) * torch.norm(vectors, dim=1))))

            avg_angles_with_KLD.append(torch.mean(angles_with_KLD).cpu().numpy())
            std_angles_with_KLD.append(torch.std(angles_with_KLD).cpu().numpy())

            avg_angles_no_KLD.append(torch.mean(angles_no_KLD).cpu().numpy())
            std_angles_no_KLD.append(torch.std(angles_no_KLD).cpu().numpy())

            avg_angles_MAE.append(torch.mean(angles_MAE).cpu().numpy())
            std_angles_MAE.append(torch.std(angles_MAE).cpu().numpy())

            avg_angles_MAE_CE.append(torch.mean(angles_MAE_CE).cpu().numpy())
            std_angles_MAE_CE.append(torch.std(angles_MAE_CE).cpu().numpy())

    Ts = np.arange(args.timesteps - 1, -1, -1)
    Ts = Ts/(args.timesteps-1)
    avg_angles_with_KLD = np.array(avg_angles_with_KLD)
    std_angles_with_KLD = np.array(std_angles_with_KLD)
    avg_angles_no_KLD = np.array(avg_angles_no_KLD)
    std_angles_no_KLD = np.array(std_angles_no_KLD)
    avg_angles_MAE = np.array(avg_angles_MAE)
    std_angles_MAE = np.array(std_angles_MAE)
    avg_angles_MAE_CE = np.array(avg_angles_MAE_CE)
    std_angles_MAE_CE = np.array(std_angles_MAE_CE)
    # Create the plot
    # Create side-by-side subplots

    plt.ylim(0, 180)
    # Plot the mean line

    if args.dataname in ['adult', 'default', 'shoppers']:
        plt.plot(Ts, avg_angles_with_KLD, color='green', label='with MSE + Cross Entropy')
        plt.plot(Ts, avg_angles_MAE_CE, color='red', label='with MAE + Cross Entropy')

    plt.plot(Ts, avg_angles_no_KLD, color='orange', label='with MSE only')
    plt.plot(Ts, avg_angles_MAE, color='blue', label='with MAE only')

    plt.axhline(y=90, color='black', linestyle='-', linewidth=2, label="90 degrees")

    # Add the "cloud" of standard deviation
    if args.dataname in ['adult', 'default', 'shoppers']:
        plt.fill_between(
            Ts,
            avg_angles_with_KLD - std_angles_with_KLD,
            avg_angles_with_KLD + std_angles_with_KLD,
            color='green',
            alpha=0.2,
            # label='±1 Std Dev'
        )
        plt.fill_between(
            Ts,
            avg_angles_MAE_CE - std_angles_MAE_CE,
            avg_angles_MAE_CE + std_angles_MAE_CE,
            color='red',
            alpha=0.2,
            # label='±1 Std Dev'
        )
    plt.fill_between(
        Ts,
        avg_angles_no_KLD - std_angles_no_KLD,
        avg_angles_no_KLD + std_angles_no_KLD,
        color='orange',
        alpha=0.2,
        # label='±1 Std Dev'
    )

    plt.fill_between(
        Ts,
        avg_angles_MAE - std_angles_MAE,
        avg_angles_MAE + std_angles_MAE,
        color='blue',
        alpha=0.2,
        # label='±1 Std Dev'
    )
    plt.gca().invert_xaxis()
    # Labels and title
    plt.xlabel("Diffusion step t", fontsize=15)
    plt.ylabel("Angles in degrees", fontsize=15)
    # fig.supxlabel("Diffusion step t", fontsize=15)
    # fig.supylabel("Angles in degrees", fontsize=15)
    # fig.suptitle(f"Average angle behavior for {args.dataname}", fontsize=18)
    # plt.title(
    #     f"Avg. angles between gradients and denoiser's\n ground truth estimate for {args.dataname}")
    plt.legend(fontsize=15)
    # axes[0].legend(fontsize=12)
    # axes[1].legend(fontsize=12)
    # plt.show()
    # plt.savefig(f'experiments/tubular_region_plots/gradient_angles_{args.dataname}_double.pdf', bbox_inches='tight')
    plt.savefig(f'experiments/tubular_region_plots/gradient_angles_{args.dataname}.pdf', bbox_inches='tight')
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # # Create the plot
    # # Create side-by-side subplots
    #
    # axes[0].set_ylim(0, 180)
    # axes[1].set_ylim(0, 180)
    # # plt.ylim(0, 180)
    # # Plot the mean line
    #
    # if args.dataname in ['adult', 'default', 'shoppers']:
    #     axes[1].plot(Ts, avg_angles_with_KLD, color='green', label='with MSE + Cross Entropy')
    #     axes[1].plot(Ts, avg_angles_MAE_CE, color='red', label='with MAE + Cross Entropy')
    #
    # axes[0].plot(Ts, avg_angles_no_KLD, color='orange', label='with MSE only')
    # axes[0].plot(Ts, avg_angles_MAE, color='blue', label='with MAE only')
    #
    # axes[0].axhline(y=90, color='black', linestyle='-', linewidth=2, label="90 degrees")
    # axes[1].axhline(y=90, color='black', linestyle='-', linewidth=2, label="90 degrees")
    #
    # # Add the "cloud" of standard deviation
    # if args.dataname in ['adult', 'default', 'shoppers']:
    #     axes[1].fill_between(
    #         Ts,
    #         avg_angles_with_KLD - std_angles_with_KLD,
    #         avg_angles_with_KLD + std_angles_with_KLD,
    #         color='green',
    #         alpha=0.2,
    #         # label='±1 Std Dev'
    #     )
    #     axes[1].fill_between(
    #         Ts,
    #         avg_angles_MAE_CE - std_angles_MAE_CE,
    #         avg_angles_MAE_CE + std_angles_MAE_CE,
    #         color='red',
    #         alpha=0.2,
    #         # label='±1 Std Dev'
    #     )
    # axes[0].fill_between(
    #     Ts,
    #     avg_angles_no_KLD - std_angles_no_KLD,
    #     avg_angles_no_KLD + std_angles_no_KLD,
    #     color='orange',
    #     alpha=0.2,
    #     # label='±1 Std Dev'
    # )
    #
    # axes[0].fill_between(
    #     Ts,
    #     avg_angles_MAE - std_angles_MAE,
    #     avg_angles_MAE + std_angles_MAE,
    #     color='blue',
    #     alpha=0.2,
    #     # label='±1 Std Dev'
    # )
    # axes[0].invert_xaxis()
    # axes[1].invert_xaxis()
    # # Labels and title
    # # plt.xlabel("Diffusion step t", fontsize=15)
    # # plt.ylabel("Angles in degrees", fontsize=15)
    # fig.supxlabel("Diffusion step t", fontsize=15)
    # fig.supylabel("Angles in degrees", fontsize=15)
    # fig.suptitle(f"Average angle behavior for {args.dataname}", fontsize=18)
    # # plt.title(
    # #     f"Avg. angles between gradients and denoiser's\n ground truth estimate for {args.dataname}")
    # # plt.legend(fontsize=15)
    # axes[0].legend(fontsize=12)
    # axes[1].legend(fontsize=12)
    # # plt.show()
    # plt.savefig(f'experiments/tubular_region_plots/gradient_angles_{args.dataname}_double.pdf', bbox_inches='tight')
