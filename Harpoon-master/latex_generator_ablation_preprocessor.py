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
    csv_path = "experiments/imputation.csv"  # <-- change this to your CSV
    model_map = {
        "harpoon_ohe_mae": "Standard",
        "harpoon_ohe_mae_quantile": "Quantile",
    }

    # -----------------------------------------------------------
    # LOAD
    # -----------------------------------------------------------
    df = pd.read_csv(
        csv_path
    ).drop(columns=['Unnamed: 0'])
    datasets = ["adult", 'default', 'shoppers']
    # Filter required datasets
    df = df[df["Dataset"].isin(datasets)]

    # Map model names
    df["Method"] = df["Method"].map(model_map)

    # ROUNDING HELPERS
    # BUILD LATEX TABLE
    # -----------------------------------------------------------
    latex = ["\\begin{table}[t]", "\\centering", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{l l " + " ".join(["c c"] * 3) + "}", "\\toprule",
             " & & \\multicolumn{2}{c}{Adult} & \\multicolumn{2}{c}{Default} & "
             "\\multicolumn{2}{c}{Shoppers} \\\\",
             "Ratio & Method & Avg. MSE & Avg. Acc & Avg. MSE & Avg. Acc & Avg. MSE & Avg. Acc \\\\", "\\midrule"]

    # Header row

    # Collect unique ratios in sorted order
    # ratios = sorted(df["ratio"].unique())
    ratios = [0.25, 0.50, 0.75]
    for r in ratios:
        sub = df[(df["Ratio"] == r) & (df["Mask Type"] == 'MAR')]

        # start multirow for ratio
        latex.append(f"\\multirow{{2}}{{*}}{{{r: .2f}}}")

        # ---- standard ----
        lin = sub[sub["Method"] == "Standard"].set_index("Dataset")
        row = [" Standard "]
        for d in datasets:
            if d in lin.index:
                row.append(cell(lin.loc[d, "Avg MSE"], lin.loc[d, "STD of MSE"]))
                row.append(cell(lin.loc[d, "Avg Acc"], lin.loc[d, "STD of Acc"]))
            else:
                row.append("-")
                row.append("-")

        latex.append(" & " + " & ".join(row) + " \\\\")

        # ---- fixed ----
        fix = sub[sub["Method"] == "Quantile"].set_index("Dataset")
        row = [" Quantile "]
        for d in datasets:
            if d in fix.index:
                row.append(cell(fix.loc[d, "Avg MSE"], fix.loc[d, "STD of MSE"]))
                row.append(cell(fix.loc[d, "Avg Acc"], fix.loc[d, "STD of Acc"]))
            else:
                row.append("-")
                row.append("-")

        latex.append(" & " + " & ".join(row) + " \\\\")
        latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}% end resizebox")
    latex.append("\\end{table}")

    # -----------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------
    print("\n".join(latex))
