import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import argparse
import warnings
import time
from tqdm import tqdm
from generate_mask import generate_mask
from dataset import Preprocessor, get_eval
from hyperimpute.utils.serialization import load, save
from hyperimpute.plugins.imputers import Imputers
import pickle
import logging
from timeit import default_timer as timer

logging.getLogger("absl").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--mask', type=str, default='MAR', help='Masking mechanisms.')
parser.add_argument('--num_trials', type=int, default=5, help='Number of sampling times.')
parser.add_argument('--ratio', type=str, default="0.25", help='Masking ratio.')
parser.add_argument('--runtime_test', type=bool, default=False, help='store runtime?')

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
    mask_type = args.mask
    ratio = float(args.ratio)
    num_trials = args.num_trials
    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    models_dir = f'saved_models/{dataname}/'

    train = pd.read_csv(f"datasets/{dataname}/train.csv")  # source train data file
    test = pd.read_csv(f"datasets/{dataname}/test.csv")  # source test data file

    # prepare the data: train and test data (true), test data (with missing values), and the mask
    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('Ordinal', prepper.df_train)  # train_X is a numpy array
    test_X = prepper.encodeDf('Ordinal', prepper.df_test)  # test_X is a numpy array
    num_numeric = prepper.numerical_indices_np_end  # index of the last numeric column

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

    imputer = Imputers().get("miracle", random_state=42)  # use Imputers to get the model

    MSEs, ACCs = [], []
    exec_times = []
    with torch.no_grad():
        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            mask_test = orig_mask[trial]
            X_test_masked = X_test.copy()
            X_test_masked[mask_test] = np.nan
            start = timer()
            imputed = imputer.fit_transform(X_test_masked).values  # Like hyperimpute, miracle also cannot adapt to new masks
            end = timer()
            diff = end - start
            exec_times.append(diff)
            imputed[:, num_numeric:] = (imputed[:, num_numeric:] * std_X[num_numeric:]) + mean_X[num_numeric:]
            imputed_decoded = prepper.decodeNp('Ordinal', imputed)
            mse, acc = get_eval(imputed_decoded, X_test_eval, mask_test, num_numeric)
            MSEs.append(mse)
            ACCs.append(acc)

    MSEs = np.array(MSEs)
    ACCs = np.array(ACCs)
    arr_time = np.array(exec_times)
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
                   "Method": "Miracle",
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
               "Method": "Miracle",
               "Mask Type": args.mask,
               "Ratio": ratio,
               "Avg MSE": np.mean(MSEs),
               "STD of MSE": np.std(MSEs),
               "Avg Acc": np.mean(ACCs),
               "STD of Acc": np.std(ACCs)
               }
    new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(experiment_path)
