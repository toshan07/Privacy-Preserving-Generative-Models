import pandas as pd
import numpy as np
import argparse


def format_for_latex(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        if abs(x) >= 100:
            digits = np.log10(x)
            return f">$10^{int(digits)}$"
        else:
            return f"{x:.2f}"
    return x


def format_for_latex_std(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        if abs(x) >= 100:
            digits = np.log10(x)
            return f""
        else:
            return f"$_{{{x:.2f}}}$"
    return x


def pivot_mask_acc(mask_type):
    method_order = [
        "One-Hot", "Integer"
    ]
    df_mask = df[df['Mask Type'] == mask_type].copy()
    df_mask = df_mask[df_mask['Dataset'].isin(['adult', 'default', 'shoppers'])]
    df_mask['Avg Acc'] = df_mask['Avg Acc'].apply(format_for_latex)
    df_mask['STD of Acc'] = df_mask['STD of Acc'].apply(format_for_latex_std)
    df_mask['Method'] = pd.Categorical(df_mask['Method'], categories=method_order, ordered=True)
    pivot_mse = df_mask.pivot_table(
        index=['Ratio', 'Method'],
        columns='Dataset',
        values='Avg Acc',
        aggfunc='first'
    )
    pivot_acc = df_mask.pivot_table(
        index=['Ratio', 'Method'],
        columns='Dataset',
        values='STD of Acc',
        aggfunc='first'
    )
    pivot_mse = pivot_mse.sort_index(level=['Ratio', 'Method'])
    pivot_acc = pivot_acc.sort_index(level=['Ratio', 'Method'])
    return pivot_mse, pivot_acc


def pivot_mask(mask_type):
    method_order = [
        "One-Hot", "Integer"
    ]
    df_mask = df[df['Mask Type'] == mask_type].copy()
    df_mask['Avg MSE'] = df_mask['Avg MSE'].apply(format_for_latex)
    df_mask['STD of MSE'] = df_mask['STD of MSE'].apply(format_for_latex_std)
    df_mask['Method'] = pd.Categorical(df_mask['Method'], categories=method_order, ordered=True)
    pivot_mse = df_mask.pivot_table(
        index=['Ratio', 'Method'],
        columns='Dataset',
        values='Avg MSE',
        aggfunc='first'
    )
    pivot_std = df_mask.pivot_table(
        index=['Ratio', 'Method'],
        columns='Dataset',
        values='STD of MSE',
        aggfunc='first'
    )
    pivot_mse = pivot_mse.sort_index(level=['Ratio', 'Method'])
    pivot_std = pivot_std.sort_index(level=['Ratio', 'Method'])
    return pivot_mse, pivot_std


# LaTeX generation with multirow for method
def generate_latex_multirow(pivot_mse, pivot_std, caption, label):
    latex_rows = []
    for ratio, group in pivot_mse.groupby(level=0):
        n_rows = len(group)
        first = True
        group_num = group.apply(pd.to_numeric, errors='coerce').fillna(np.inf)
        if 'acc' in label:
            best_indices = group_num.apply(lambda col: col.nlargest(1).index.tolist())
        else:
            best_indices = group_num.apply(lambda col: col.nsmallest(1).index.tolist())
        best_mask = pd.DataFrame(False, index=group_num.index, columns=group_num.columns)
        secondbest_mask = pd.DataFrame(False, index=group_num.index, columns=group_num.columns)
        # for column in best_mask.columns:
        #     best_loc = best_indices.loc[0, column]
        #     secondbest_loc = best_indices.loc[1, column]
        #     best_mask.loc[best_loc, column] = True
        #     secondbest_mask.loc[secondbest_loc, column] = True

        # Now build formatted strings (mean$_{std}$) and apply bold/underline
        group_fmt = group.copy()
        for idx in group_fmt.index:
            for col in group_fmt.columns:
                mean = group.loc[idx, col]
                std = pivot_std.loc[idx, col]
                cell = mean+std
                if best_mask.loc[idx, col]:
                    cell = "\\textbf{" + mean + "}"+std
                elif secondbest_mask.loc[idx, col]:
                    cell = "\\underline{" + mean + "}"+std
                group_fmt.loc[idx, col] = cell
        group = group.where(~best_mask, "\\textbf{" + group.astype(str) + "}")
        group = group.where(~secondbest_mask, "\\underline{" + group.astype(str) + "}")
        for idx, row in group_fmt.iterrows():
            if first:
                ratio_cell = f"\\multirow{{{n_rows}}}{{*}}{{{ratio}}}"
                first = False
            else:
                ratio_cell = ""

            row_values = " & ".join(row.values)
            latex_rows.append(f"{ratio_cell} & {idx[1]} & {row_values} \\\\")
    latex_body = "\n".join(latex_rows)

    # Column headers: Method | Ratio | Dataset1 | Dataset2 | ...
    columns = ["Ratio", "Method"] + list(pivot_mse.columns)
    col_str = " & ".join(columns) + " \\\\"

    latex_code = f"""
\\begin{{table}}[ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' * (len(columns))}}}
\\hline
{col_str}
\\hline
{latex_body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return latex_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Latex generator arguments')
    parser.add_argument('--mask', type=str, required=True, help='MAR, MCAR or MNAR scenario')
    parser.add_argument('--task', type=str, required=True, help='acc or mse table')
    args = parser.parse_args()
    df = pd.read_csv("experiments/imputation.csv").drop(columns=['Unnamed: 0'])
    # Only keep relevant columns
    df = df[['Dataset', 'Method', 'Mask Type', 'Ratio', 'Avg MSE', 'STD of MSE', 'Avg Acc', 'STD of Acc']]
    df = df[df['Dataset'].isin(['adult', 'default', 'shoppers'])]
    df = df[df['Method'].isin(['harpoon_ohe_mae', 'Harpoon_ordinal_mae'])]

    df = df[df['Ratio'].isin([0.25, 0.5, 0.75])]
    df['Ratio'] = df['Ratio'].map(lambda x: f"{x: .2f}")

    df['Method'] = df['Method'].replace(
        {'harpoon_ohe_mae': 'One-Hot', 'Harpoon_ordinal_mae': 'Integer'})

    # Function to create pivot table per mask type
    if args.task == 'mse':
        table_mse, table_std = pivot_mask(args.mask)
        latex = generate_latex_multirow(table_mse, table_std, f'Imputation MSE under different encodings for \\textsc{{harpoon}}.', f'tab:encodingmse')
    else:
        table_mse, table_std = pivot_mask_acc(args.mask)
        latex = generate_latex_multirow(table_mse, table_std, f'Imputation Accuracy under different encodings for \\textsc{{harpoon}}.', f'tab:encodingacc')

    print(latex)
