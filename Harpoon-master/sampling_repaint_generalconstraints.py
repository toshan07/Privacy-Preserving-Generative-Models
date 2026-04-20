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
from generate_mask import generate_mask, constrainDataset
from model import MLPDiffusion
from dataset import Preprocessor
from utils import calc_diffusion_hyperparams
import matplotlib.pyplot as plt
from sampling_harpoon_ohe_basicmanifold_kld import computeCatLoss
from synthcity.metrics.eval_statistical import AlphaPrecision, KolmogorovSmirnovTest
from synthcity.metrics.eval_detection import SyntheticDetectionLinear
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_privacy import IdentifiabilityScore
from synthcity.metrics.weighted_metrics import WeightedMetrics 

warnings.filterwarnings('ignore')
torch.set_num_threads(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser(description='General constraint synthesis')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--timesteps', type=int, default=200, help='Number of diffusion steps.')
    parser.add_argument('--beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('--beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('--constraint', type=str, default='range', help='Constraint choice. range, category, both')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of sampling times.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    dataname = args.dataname
    device = args.device
    hid_dim = args.hid_dim
    constraint = args.constraint
    num_trials = args.num_trials

    prepper = Preprocessor(dataname)
    const_df, mask_df, rangecol, bound_type, bound, cata = constrainDataset(dataname, constraint,
                                                                      prepper)  # constrained df and the mask
    train_X = prepper.encodeDf('OHE', prepper.df_train)
    test_X = prepper.encodeDf('OHE', const_df)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (
        np.concatenate((np.mean(train_X[:, :num_numeric], axis=0), np.zeros(train_X.shape[1] - num_numeric)), axis=0),
        np.concatenate((np.std(train_X[:, :num_numeric], axis=0), np.ones(train_X.shape[1] - num_numeric)), axis=0))
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    X = torch.tensor(X)
    X_test = (test_X - mean_X) / std_X
    X_test = torch.tensor(X_test, dtype=torch.float32)
    bound_standardized, bound_standardized_np = None, None
    if constraint != 'category':
        bound_standardized = torch.tensor((bound - mean_X[rangecol])/std_X[rangecol])
        bound_standardized_np = bound_standardized.numpy()
    test_X_ori_fmt = np.concatenate((prepper.df_test.iloc[:, prepper.info['num_col_idx']],
                                     prepper.df_test.iloc[:, prepper.info['cat_col_idx']]), axis=1)
    test_X_ordinal_fmt = prepper.encodeDf('Ordinal', prepper.df_test)
    mask_nums = mask_df.iloc[:, prepper.info['num_col_idx']]
    mask_cats = mask_df.iloc[:, prepper.info['cat_col_idx']]
    orig_mask_base = np.concatenate((mask_nums, mask_cats), axis=1)
    orig_mask = np.tile(orig_mask_base, (num_trials, 1, 1))
    diffusion_config = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_T)
    test_masks = prepper.extend_mask(orig_mask, encoding='OHE')

    models_dir = f'saved_models/{args.dataname}/'
    model_path = os.path.join(models_dir, "diffputer_selfmade.pt")
    model = MLPDiffusion(in_dim, hid_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    mask_tests = torch.tensor(test_masks)
    alpha_ps, violation_accs = [], []  # alpha precisions and violation accuracies
    detection_score, privacy_score, ks_score = [], [], []
    util_score = []  # utility scores
    X_test_gpu = X_test.to(device)

    for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
        with (torch.no_grad()):
            mask_test = mask_tests[trial]
            mask_float = mask_test.float().to(device)
            x_t = torch.randn_like(X_test).to(device)
            for t in range(args.timesteps - 1, -1, -1):
                timesteps = torch.full(size=(x_t.shape[0],), fill_value=t).to(device)
                alpha_t = diffusion_config['Alpha'][t].to(device)
                alpha_bar_t = diffusion_config['Alpha_bar'][t].to(device)
                alpha_bar_t_1 = diffusion_config['Alpha_bar'][t - 1].to(device) if t >= 1 else torch.tensor(1).to(
                    device)
                sigmas_predicted = model(x_t, timesteps)
                x_t = (x_t / torch.sqrt(alpha_t)) - (
                        (1 - alpha_t) / (torch.sqrt(alpha_t) * torch.sqrt(1 - alpha_bar_t))) * sigmas_predicted

                vari = 0.0
                if t > 0:
                    vari = (1 - alpha_t) * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(0, 1,
                                                                                                    size=x_t.shape).to(
                        device)
                x_t += vari
                x_cond_t = torch.sqrt(alpha_bar_t_1) * X_test_gpu + torch.sqrt(1 - alpha_bar_t_1) * torch.randn_like(
                    X_test_gpu)
                x_t = (1 - mask_float) * x_cond_t + mask_float * x_t
        X_pred = x_t.cpu().numpy()
        # X_pred = (x_t * mask_float + (1 - mask_float) * X_test_gpu).cpu().numpy()
        X_true = X_test.numpy()
        X_true_dec = prepper.decodeNp(scheme='OHE', arr=X_true)
        X_pred_dec = prepper.decodeNp(scheme='OHE', arr=X_pred)

        range_violations = np.zeros(len(X_pred_dec), dtype=bool)
        category_violations = np.zeros(len(X_pred_dec), dtype=bool)
        if bound_type == 'lb':
            range_violations = (X_pred_dec[:, rangecol] < bound_standardized_np)  # Is X_pred greater than or equal to lower bound constraint?
        elif bound_type == 'ub':
            range_violations = (X_pred_dec[:, rangecol] > bound_standardized_np)  # Is X_pred lesser than or equal to upper bound constraint?
        if constraint != 'range' and constraint != 'or':
            category_violations = (X_pred_dec[~orig_mask_base] != X_true_dec[~orig_mask_base])
        elif constraint != 'range' and constraint == 'or':
            category_violations = (X_pred_dec[~orig_mask_base] != cata)

        if constraint != 'or':
            overall_violations = category_violations | range_violations
        else:
            overall_violations = category_violations & range_violations
        violation_p = (np.sum(overall_violations)*100.0)/len(overall_violations)
        violation_accs.append(violation_p)
        X_true_dec_enc = prepper.encodeNp(scheme='OHE', arr=X_true_dec).astype(np.float32)
        X_pred_dec_enc = prepper.encodeNp(scheme='OHE', arr=X_pred_dec).astype(np.float32)
        evaluator = AlphaPrecision()
        evaluator_detection = SyntheticDetectionLinear()
        evaluator_resemblance = KolmogorovSmirnovTest()
        evaluator_priv = IdentifiabilityScore()
        evaluator_util_xgb = WeightedMetrics(
            metrics=[("performance", "xgb")],   # category, metric name
            weights=[1.0],                      # single metric → weight 1.0
            task_type="classification",         # or "regression"
            random_state=0,
        )
        X_syn_loader = GenericDataLoader(X_pred_dec_enc)
        X_real_loader = GenericDataLoader(X_true_dec_enc)
        X_true_dec_enc_util = prepper.encodeNp(scheme='Ordinal', arr=X_true_dec).astype(np.float32)
        X_pred_dec_enc_util = prepper.encodeNp(scheme='Ordinal', arr=X_pred_dec).astype(np.float32)
        if args.dataname == 'adult':
            target = '12'
        elif args.dataname == 'default':
            target = '16'
        else:
            target = '11'
        X_real_loader_util = GenericDataLoader(X_true_dec_enc_util, target_column=target)
        X_syn_loader_util = GenericDataLoader(X_pred_dec_enc_util, target_column=target)
        results_util = evaluator_util_xgb.evaluate(X_real_loader_util, X_syn_loader_util)  # XGBoost utility score
        results = evaluator.evaluate(X_real_loader, X_syn_loader)
        results_detection = evaluator_detection.evaluate(X_real_loader, X_syn_loader)
        results_ks = evaluator_resemblance.evaluate(X_real_loader, X_syn_loader)
        results_privacy = evaluator_priv.evaluate(X_real_loader, X_syn_loader)
        alpha = results['delta_precision_alpha_naive']
        ks_score.append(results_ks['marginal'])
        detection_score.append(results_detection['mean'])
        privacy_score.append(results_privacy['score'])
        alpha_ps.append(alpha)
        util_score.append(results_util)

    alpha_ps = np.array(alpha_ps)
    ks_es = np.array(ks_score)
    ident_s = np.array(privacy_score)
    detect_s = np.array(detection_score)
    violation_accs = np.array(violation_accs)
    experiment_path = f'experiments/general_constraints_updated_utility.csv'
    directory = os.path.dirname(experiment_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(experiment_path):
        columns = [
            "Dataset",
            "Method",
            "Constraint",
            "Avg Alpha-P",
            "Std Alpha-P",
            "Avg ViolationAcc",
            "Std ViolationAcc",
            "Avg ks",
            "Std ks",
            "Avg detect",
            "Std detect",
            "Avg identifiability",
            "Std identifiability",
            "Avg xgb utility",
            "Std xgb utility"
        ]
        exp_df = pd.DataFrame(columns=columns)
    else:
        exp_df = pd.read_csv(experiment_path).drop(columns=['Unnamed: 0'])

    new_row = {"Dataset": dataname,
               "Method": f"DiffPuter_Remastered",
               "Constraint": args.constraint,
               "Avg Alpha-P": np.mean(alpha_ps),
               "Std Alpha-P": np.std(alpha_ps),
               "Avg ViolationAcc": np.mean(violation_accs),
               "Std ViolationAcc": np.std(violation_accs),
               "Avg ks": np.mean(ks_es),
               "Std ks": np.std(ks_es),
               "Avg detect": np.mean(detect_s),
               "Std detect": np.std(detect_s),
               "Avg identifiability": np.mean(ident_s),
               "Std identifiability": np.std(ident_s),
               "Avg xgb utility": np.mean(util_score),
               "Std xgb utility": np.std(util_score)
               }
    new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(experiment_path)
