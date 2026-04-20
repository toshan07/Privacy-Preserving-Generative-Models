import pandas as pd
import numpy as np


def fmt(val):
    """Format average value to 2 decimals."""
    if pd.isna(val): return "-"
    return f"{val:.2f}"


def fmt_std(std):
    """Format std dev as a subscript."""
    if pd.isna(std): return ""
    return f"$_{{{std:.2f}}}$"


def cell(avg, std):
    """Return LaTeX cell: avg_subscript"""
    if pd.isna(avg):
        return "-"
    return f"{fmt(avg)}{fmt_std(std)}"


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
if __name__ == '__main__':
    csv_path = "experiments/imputation_linear_schedule_ablation.csv"  # <-- change this to your CSV
    datasets = ["adult", "default", "shoppers"]
    model_map = {
        "harpoon_ohe_mae_linear": "linear",
        "harpoon_ohe_mae": "fixed",
    }

    # -----------------------------------------------------------
    # LOAD
    # -----------------------------------------------------------
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["id", "dataset", "model", "mechanism", "ratio",
               "mse_avg", "mse_std", "acc_avg", "acc_std"]
    )

    # Filter required datasets
    df = df[df["dataset"].isin(datasets)]

    # Map model names
    df["model_type"] = df["model"].map(model_map)

    # ROUNDING HELPERS
    # BUILD LATEX TABLE
    # -----------------------------------------------------------
    latex = ["\\begin{table}[t]", "\\centering", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{l l " + " ".join(["c c"] * 3) + "}", "\\toprule",
             " & & \\multicolumn{2}{c}{Adult} & \\multicolumn{2}{c}{Default} & "
             "\\multicolumn{2}{c}{Shoppers} \\\\",
             "Ratio & Model & Avg. MSE & Avg. Acc. & Avg. MSE & Avg. Acc. & Avg. MSE & Avg. Acc. \\\\", "\\midrule"]

    # Header row

    # Collect unique ratios in sorted order
    ratios = sorted(df["ratio"].unique())

    for r in ratios:
        sub = df[df["ratio"] == r]

        # start multirow for ratio
        latex.append(f"\\multirow{{2}}{{*}}{{{r: .2f}}}")

        # ---- linear ----
        lin = sub[sub["model_type"] == "linear"].set_index("dataset")
        row = [" linear "]
        for d in datasets:
            if d in lin.index:
                row.append(cell(lin.loc[d, "mse_avg"], lin.loc[d, "mse_std"]))
                row.append(cell(lin.loc[d, "acc_avg"], lin.loc[d, "acc_std"]))
            else:
                row.append("-")
                row.append("-")

        latex.append(" & " + " & ".join(row) + " \\\\")

        # ---- fixed ----
        fix = sub[sub["model_type"] == "fixed"].set_index("dataset")
        row = [" fixed "]
        for d in datasets:
            if d in fix.index:
                row.append(cell(fix.loc[d, "mse_avg"], fix.loc[d, "mse_std"]))
                row.append(cell(fix.loc[d, "acc_avg"], fix.loc[d, "acc_std"]))
            else:
                row.append("-")
                row.append("-")

        latex.append(" & " + " & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}% end resizebox")
    latex.append("\\end{table}")

    # -----------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------
    print("\n".join(latex))
