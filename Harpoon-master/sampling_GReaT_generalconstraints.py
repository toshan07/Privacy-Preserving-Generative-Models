import os
import json
import numpy.random
import torch
from be_great import GReaT
import warnings
import numpy as np
import logging
import argparse
import pandas as pd
import time
from tqdm import tqdm
from generate_mask import generate_mask, constrainDataset
from dataset import Preprocessor, get_eval
from synthcity.metrics.eval_statistical import AlphaPrecision, KolmogorovSmirnovTest
from synthcity.metrics.eval_detection import SyntheticDetectionLinear
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_privacy import IdentifiabilityScore
from synthcity.metrics.weighted_metrics import WeightedMetrics 



# TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ignore Python warnings
warnings.filterwarnings("ignore")

# Transformers logging
logging.getLogger("transformers.trainer").setLevel(logging.INFO)

# Suppress one-off warnings from tokenizer/config/etc.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

# Suppress decoder-only padding warning
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# PyTorch tracing helper warning
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--constraint', type=str, default='range', help='Constraint choice. range, category, both, or')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of sampling times.')
    torch.manual_seed(42)
    numpy.random.seed(42)
    args = parser.parse_args()
    constraint = args.constraint
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    dataname = args.dataname
    device = args.device
    num_trials = args.num_trials

    infopath = f'datasets/Info/{args.dataname}.json'
    info = None
    with open(infopath, 'r') as f:
        info = json.load(f)

    prepper = Preprocessor(dataname)
    const_df, mask_df, rangecol, bound_type, bound, cata = constrainDataset(dataname, constraint,
                                                                      prepper)  # constrained df and the mask
    train_df = pd.read_csv(f'datasets/{args.dataname}/train.csv')
    stooge = prepper.encodeDf('OHE', prepper.df_train)
    test_df = const_df.copy()
    num_cols = info['num_col_idx']
    cat_cols = info['cat_col_idx']
    num_numeric = len(num_cols)
    train_df = train_df.iloc[:, num_cols + cat_cols]
    test_df = test_df.iloc[:, num_cols + cat_cols]
    mean_X = train_df.iloc[:, :num_numeric].mean()
    std_X = train_df.iloc[:, :num_numeric].std()

    standardized_test_num = (test_df.iloc[:, :num_numeric] - mean_X)/std_X
    test_eval = np.concatenate((standardized_test_num.values, test_df.iloc[:, num_numeric:].values), axis=1)

    bound_standardized, bound_standardized_np = None, None
    if constraint != 'category':
        bound_standardized = torch.tensor((bound - mean_X[rangecol]) / std_X[rangecol])
        bound_standardized_np = bound_standardized.numpy()

    # Code for generating masks: Must be consistent across all baselines
    mask_nums = mask_df.iloc[:, prepper.info['num_col_idx']]
    mask_cats = mask_df.iloc[:, prepper.info['cat_col_idx']]
    orig_mask_base = np.concatenate((mask_nums, mask_cats), axis=1)
    orig_mask = np.tile(orig_mask_base, (num_trials, 1, 1))

    alpha_ps, violation_accs = [], []  # alpha precisions and violation accuracies
    detection_score, privacy_score, ks_score = [], [], []
    util_score = []

    models_dir = f'saved_models/{args.dataname}/GReaT'
    great_model = GReaT.load_from_dir(models_dir)
    # max_length = 200 if args.dataname not in ['bean', 'default', 'gesture'] else 350  # some datasets need more tokens
    cat_edge_case = True if args.dataname == 'adult' else False
    sz = 100 if args.dataname in ['gesture', 'news', 'default'] else 200

    for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
        torch.cuda.empty_cache()
        with torch.no_grad():
            mask = orig_mask[trial]
            masked_df = test_df.mask(mask)
            imputed_data = great_model.impute(masked_df, k=sz, max_retries=5)
            standardized_imputed_num = (imputed_data.iloc[:, :num_numeric] - mean_X)/std_X
            standardized_imputed_num = standardized_imputed_num.fillna(0.0)  # for numerical features that were not generated within the max retries
            imputed_eval = np.concatenate((standardized_imputed_num, imputed_data.iloc[:, num_numeric:]), axis=1)
        X_true_dec = test_eval.copy()
        X_pred_dec = imputed_eval.copy()
        if cat_edge_case:
            X_pred_dec[:, num_numeric:] = np.core.defchararray.add(" ", np.char.lstrip(X_pred_dec[:, num_numeric:].astype(str)))
        results = []
        for i, row in enumerate(X_pred_dec):  # GreaT sometimes samples invalid categories. In that case we simply replace it with a randomly drawn valid one
            try:
                row = row.reshape(1, -1)
                encoded = prepper.encodeNp(scheme='OHE', arr=row)
                results.append(row[0])
            except ValueError as e:
                # print(f"⚠️ Unknown category in row {i}: {row}")
                # Make a copy of the row so we can patch it
                fixed_row = row.copy()
                for col_idx, known in enumerate(prepper.OneHotEncoder.categories_):
                    if row[0, num_numeric+col_idx] not in known:
                        fixed_row[0, num_numeric+col_idx] = np.random.choice(known)  # replace unknown with default

                results.append(fixed_row[0])
        X_pred_dec = np.array(results)
        range_violations = np.zeros(len(X_pred_dec), dtype=bool)
        category_violations = np.zeros(len(X_pred_dec), dtype=bool)
        if bound_type == 'lb':
            try:
                range_violations = (X_pred_dec[:,
                                    rangecol] < bound_standardized_np)  # Is X_pred greater than or equal to lower bound constraint?
            except Exception:
                print()
        elif bound_type == 'ub':
            range_violations = (X_pred_dec[:,
                                rangecol] > bound_standardized_np)  # Is X_pred lesser than or equal to upper bound constraint?

        if constraint != 'range' and constraint != 'or':
            category_violations = (X_pred_dec[~orig_mask_base] != X_true_dec[~orig_mask_base])
        elif constraint != 'range' and constraint == 'or':
            category_violations = (X_pred_dec[~orig_mask_base] != cata)

        if constraint != 'or':
            overall_violations = category_violations | range_violations
        else:
            overall_violations = category_violations & range_violations
        violation_p = (np.sum(overall_violations) * 100.0) / len(overall_violations)
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
               "Method": f"GReaT",
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




