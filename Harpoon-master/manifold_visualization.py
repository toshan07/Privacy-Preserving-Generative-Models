import os
import torch
import pandas as pd
import numpy as np
import argparse
import warnings
import time
from dataset import Preprocessor, get_eval
from utils import calc_diffusion_hyperparams
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
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

    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('OHE', prepper.df_train)
    test_X = prepper.encodeDf('OHE', prepper.df_test)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (
    np.concatenate((np.mean(train_X[:, :num_numeric], axis=0), np.zeros(train_X.shape[1] - num_numeric)), axis=0),
    np.concatenate((np.std(train_X[:, :num_numeric], axis=0), np.ones(train_X.shape[1] - num_numeric)), axis=0))
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    pca = PCA(n_components=2)
    X_0 = pca.fit_transform(X)
    diffusion_config = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_T)
    manifold_level = 199
    c1 = torch.sqrt(diffusion_config['Alpha_bar'][manifold_level])
    c2 = torch.sqrt(1-(c1**2))
    noise = torch.normal(0, 1, size=X.shape)
    X_T_gt = c1 * X + c2 * noise
    X_T = pca.fit_transform(X_T_gt)
    # Plot 2D
    plt.scatter(X_0[:, 0], X_0[:, 1], s=10, alpha=0.6)
    plt.scatter(X_T[:, 0], X_T[:, 1], s=10, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D projection")
    plt.grid(True)
    plt.show()

    # Plot 3D
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], s=10, alpha=0.6)
    # ax.scatter(X_T[:, 0], X_T[:, 1], X_T[:, 2], s=10, alpha=0.6)
    # ax.set_title("PCA 3D projection")
    # plt.show()




