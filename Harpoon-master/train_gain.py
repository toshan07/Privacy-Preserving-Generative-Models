# train_hyperimpute_min.py
import os, pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from hyperimpute.plugins.imputers import Imputers
from dataset import Preprocessor, get_eval
from generate_mask import generate_mask
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from hyperimpute.utils.serialization import load, save
import torch

import warnings
warnings.simplefilter("ignore", FutureWarning)

# --- stability on macOS / BLAS ---
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
          "CATBOOST_THREAD_COUNT", "XGB_NUM_THREADS"]:
    os.environ[k] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')


args = parser.parse_args()
# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # Initialize other args
    dataname = args.dataname
    device = args.device
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
    # X_test = (test_X - mean_X) / std_X

    mask_type = 'MCAR'  # or 'MAR', 'MCAR', 'MNAR_logistic_T2'
    ratio = 0.2  # train on MCAR 0.2

    train_mask = generate_mask(train_X, mask_type=mask_type, mask_num=1, p=ratio)[0]
    plugins = [ # or
        "gain",
    ]

    for plugin in plugins:
        print(f"Plugin: {plugin}")
        if plugin == "hyperimpute":
            imputer = Imputers().get(
                plugin,
                random_state=42,
                optimizer="hyperband",
                classifier_seed=["logistic_regression", "random_forest", "xgboost", "catboost"],
                regression_seed=["linear_regression", "random_forest_regressor", "xgboost_regressor",
                                 "catboost_regressor"],
            )
        else:
            imputer = Imputers().get(plugin, random_state=42)  # use Imputers to get the model
        # 3) Fit on training data (learn how to fill)
        X_miss = X.copy()
        X_miss[train_mask] = np.nan
        imputed = imputer.fit_transform(X_miss)
        # Save to a file

        buff = save(imputer)  # get the model as bytes
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"{plugin}.pkl")

        # after imputer.fit(...)
        with open(path, "wb") as f:
            f.write(buff)  # .model() returns bytes

        print(f"Saved imputer model: {f.name}")

