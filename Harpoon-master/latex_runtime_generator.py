import pandas as pd

# Example data: rows = methods, columns = datasets
# Each entry is (avg, std)
if __name__ == "__main__":
    df = pd.read_csv('experiments/runtime.csv')
    df['Method'] = df['Method'].replace({'harpoon_ohe_mae': 'Harpoon', 'DiffPuter_Remastered': 'DiffPuter'})

    methods = ['GAIN', 'Miracle', 'GReaT', 'Remasker', 'DiffPuter', 'Harpoon']
    datasets = df['Dataset'].unique()

    # Start LaTeX table
    latex = "\\begin{table}[htb]\n\\centering\n"
    latex += "\\resizebox{\\textwidth}{!}{%\n"
    latex += "\\begin{tabular}{l" + "c" * len(datasets) + "}\n"
    latex += "\\toprule\n"

    # Header
    latex += "Method"
    for ds in datasets:
        latex += f" & {ds}"
    latex += " \\\\\n\\midrule\n"

    # Rows
    for method in methods:
        latex += method
        for ds in datasets:
            mask = (df['Dataset'] == ds) & (df['Method'] == method)
            filtered = df.loc[mask]
            avg, std = filtered['Avg Runtime'].values[0], filtered['STD of Runtime'].values[0]
            latex += f" & {avg: .2f}$_{{{std: .2f}}}$"
        latex += " \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}%\n"
    latex += "}\n"  # close resizebox
    latex += "\\caption{Runtimes (seconds) of methods across datasets.}\n"
    latex += "\\label{tab:runtimes}\n\\end{table}"

    print(latex)