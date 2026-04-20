import os
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm
from utils import calc_diffusion_hyperparams
from model import MLPDiffusion
from dataset import Preprocessor

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=1000)
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
    np.random.seed(42)
    torch.manual_seed(42)
    dataname = args.dataname
    device = args.device
    hid_dim = args.hid_dim
    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('Ordinal', prepper.df_train)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (
        np.mean(train_X, axis=0), np.std(train_X, axis=0)
    )
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    X = torch.tensor(X, dtype=torch.float32)

    diffusion_config = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_T)
    print(dataname)
    models_dir = f'saved_models/{dataname}/'
    os.makedirs(f'{models_dir}') if not os.path.exists(f'{models_dir}') else None

    train_loader = DataLoader(
        X,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    model = MLPDiffusion(in_dim, hid_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, verbose=False)
    criterion = torch.nn.MSELoss()
    model.train()

    best_loss = float('inf')
    patience = 0

    pbar = tqdm(range(args.epochs), desc='Training')
    for epoch in pbar:
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            timesteps = torch.randint(args.timesteps, size=(batch.shape[0],)).to(device)
            sigmas = torch.normal(0, 1, size=batch.shape).to(device)
            """Forward noising"""
            alpha_bars = diffusion_config['Alpha_bar'].to(device)
            coeff_1 = torch.sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1)).to(device)
            coeff_2 = torch.sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1)).to(device)
            batch_noised = coeff_1 * batch + coeff_2 * sigmas
            batch_noised = batch_noised.to(device)
            sigmas_predicted = model(batch_noised, timesteps)
            loss = criterion(sigmas_predicted, sigmas)
            loss.backward()
            total_loss += loss
            optimizer.step()
        scheduler.step(total_loss)

        # if total_loss < best_loss:
        #     best_loss = total_loss
        #     patience = 0

        # else:
        #     patience += 1
        #     if patience == 50:
        #         print('Early stopping')
        #         break

        pbar.set_postfix(loss=total_loss)
    torch.save(model.state_dict(), f'{models_dir}/harpoon_ordinal.pt')