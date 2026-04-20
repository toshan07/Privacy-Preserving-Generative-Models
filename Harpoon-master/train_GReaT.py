import os

import pandas as pd
import torch
from be_great import GReaT
import warnings
import numpy as np
import logging
import argparse
import json


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("transformers.trainer").setLevel(logging.INFO)

# Suppress one-off warnings from tokenizer/config/etc.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

# PyTorch tracing helper warning
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--epochs', type=int, default=5)

    args = parser.parse_args()
    b_sz = 16 if args.dataname == 'news' else 32
    # from sklearn.datasets import fetch_california_housing
    #

    # data = fetch_california_housing(as_frame=True).frame
    # data = data.iloc[:100]
    infopath = f'datasets/Info/{args.dataname}.json'
    info = None
    with open(infopath, 'r') as f:
        info = json.load(f)

    #
    data = pd.read_csv(f'datasets/{args.dataname}/train.csv')
    num_cols = info['num_col_idx']
    cat_cols = info['cat_col_idx']
    data = data.iloc[:, num_cols+cat_cols]
    trainer_kwargs = dict(
        save_strategy="no",
    )
    great_model = GReaT(llm='distilgpt2', batch_size=b_sz, epochs=args.epochs,
                  fp16=True, float_precision=3, dataloader_num_workers=4, report_to=[], save_steps=100000, experiment_dir=".", save_strategy='no', logging_strategy='epoch', logging_first_step=False)
    great_model.fit(data)

    models_dir = f'saved_models/{args.dataname}/GReaT'
    os.makedirs(f'{models_dir}') if not os.path.exists(f'{models_dir}') else None
    great_model.save(models_dir)
    # with torch.no_grad():
    #     new = GReaT.load_from_dir("trainer_great/checkpoint-645")
    #     print()
