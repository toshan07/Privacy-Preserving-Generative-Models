import os
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm

from model import MLPDiffusion, Model
from dataset import Preprocessor
from diffusion_utils import sample_step, impute_mask

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'

if __name__ == '__main__':

    dataname = args.dataname
    device = args.device
    hid_dim = args.hid_dim
    num_steps = args.num_steps
    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('OHE', prepper.df_train)

    # train_X, test_X, train_num, test_num, train_cat_idx, test_cat_idx, cat_bin_num = load_dataset_nomask(
    #     dataname)
    # recovered = prepper.decodeNp('OHE', train_X)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (np.concatenate((np.mean(train_X[:, :num_numeric], axis=0), np.zeros(train_X.shape[1]-num_numeric)), axis=0),
                     np.concatenate((np.std(train_X[:, :num_numeric], axis=0), np.ones(train_X.shape[1]-num_numeric)), axis=0))
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    X = torch.tensor(X)

    models_dir = f'saved_models/{dataname}/'
    os.makedirs(f'{models_dir}') if not os.path.exists(f'{models_dir}') else None
    batch_size = 4096
    train_loader = DataLoader(
        X,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    num_epochs = 1000+1

    denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

    # print(denoise_fn)
    print(dataname)

    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, verbose=False)

    model.train()

    best_loss = float('inf')
    patience = 0

    # progress bar
    pbar = tqdm(range(num_epochs), desc='Training')
    for epoch in pbar:

        batch_loss = 0.0
        len_input = 0

        for batch in train_loader:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{models_dir}/diffputer.pt')
        else:
            patience += 1
            if patience == 50:
                print('Early stopping')
                break

        pbar.set_postfix(loss=curr_loss)


