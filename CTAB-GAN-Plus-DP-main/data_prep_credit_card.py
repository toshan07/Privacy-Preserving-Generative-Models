import pandas as pd
import os

path="Real_Datasets/data.csv"
path1="Real_Datasets/clean_data.csv"
df = pd.read_csv(path,header=1,encoding='utf-8')

if "ID" in df.columns:
    df = df.drop(columns=["ID"])
elif "Id" in df.columns:
    df = df.drop(columns=["Id"])
possible_targets = ["default payment next month", "default.payment.next.month", "default", "Y"]
target_col = None
for cand in possible_targets:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    for c in df.columns:
        if "default" in str(c).lower() and "month" in str(c).lower():
            target_col = c
            break

if target_col is None:
    raise ValueError("Target column not found. Expected one of: " + ", ".join(possible_targets))

df = df.rename(columns={target_col: "default"})

categorical_cols = []
for c in df.columns:
    uc = c.upper()
    if uc in ("SEX", "EDUCATION", "MARRIAGE"):
        categorical_cols.append(c)
    if uc.startswith("PAY_") or uc.startswith("PAY") and uc[-1].isdigit():
        categorical_cols.append(c)

for c in categorical_cols:
    if c in df.columns:
        df[c] = df[c].astype("Int64")  
        df[c] = df[c].astype("category")

if "default" in df.columns:
    df["default"] = df["default"].astype("Int64")
    df["default"] = df["default"].astype("category")

os.makedirs(os.path.dirname(path1) or ".", exist_ok=True)
df.to_csv(path1, index=False)
print(f"Wrote cleaned CSV to {path1}. Shape: {df.shape}")
print("Categorical columns:", categorical_cols + ["default"])