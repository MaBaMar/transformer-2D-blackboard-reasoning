import argparse
import os

import pandas as pd
import numpy as np
import wandb as wb
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm



IMG_PATH = "./plots/images/"
TBL_PATH = "./plots/tables/"
DATA_PATH = "./plots/data/"
DATA_FILE = "./plots/data/{file_name:s}.csv"



#
#   Get the data
#

def wandb_get_data(tag: str):
    print(f"Trying to load data with tag \"{tag}\"")
    api = wb.Api()
    runs = api.runs(
        path="blackboard-reasoning/blackboard-reasoning", 
        filters={"tags": {"$in": [tag]}}
    )

    # Iterate over runs and fetch data
    all_data = []
    print("Getting runs")
    for run in tqdm(runs):
        df_history = run.history()
        df_config = pd.concat([pd.DataFrame([run.config])] * df_history.shape[0], ignore_index=True)

        all_data.append(
            pd.concat(
                [
                    df_config,
                    df_history
                ],
                axis=1
            )
        )

    data = pd.concat(all_data)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    data.to_csv(DATA_FILE.format(file_name=tag), index=False)



def group_data(file_name):
    data = pd.read_csv(DATA_FILE.format(file_name=file_name))

    grouped_data = data.groupby(["model", "digits", "operation"], as_index=False)

    # Calculate standard error for each group
    acc_mean = grouped_data["accuracy"].mean().rename(columns={"accuracy": "acc_mean"})
    acc_stddev = grouped_data["accuracy"].sem().rename(columns={"accuracy": "acc_stddev"})

    merged = pd.merge(acc_mean, acc_stddev, on=["model", "digits", "operation"])

    return merged



#
#   Plot the results
#

def plot_performance(df):
    digits = sorted(df["digits"].unique())
    models = ["EOgar-2d", "EOgar-1d", "CoT"]
    operations = ["add", "sub", "mixed"]

    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Mean accuracy and variation of the different architectures over addition, subtraction and mixed datasets. The mean and standard deviation are multiplied by 100 to express values as percentages. All reported values are rounded to one decimal place.}")
    lines.append(r"\label{table:ood}")
    lines.append(r"\vskip 0.15in")
    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(r"\begin{sc}")

    col_spec = "l" + "c" * len(digits)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    for model in models:
        if model == models[0]:
            header = (
                r"\textbf{" + model + "}"
                + " & "
                + " & ".join(str(d) for d in digits)
                + r" \\"
            )
            lines.append(header)
        else:
            lines.append(
                rf"\multicolumn{{{len(digits)+1}}}{{l}}{{\textbf{{{model}}}}} \\"
            )

        lines.append(r"\midrule")

        for op in operations:
            row = [op.capitalize()]

            for d in digits:
                m = df[
                    (df["model"] == model)
                    & (df["operation"] == op)
                    & (df["digits"] == d)
                ]

                if len(m) == 1:
                    mean = m["acc_mean"].values[0] * 100
                    std = m["acc_stddev"].values[0] * 100
                    cell = rf"\scriptsize{{{mean:.1f}$\sigma${std:.1f}}}"
                else:
                    cell = r"\scriptsize{--}"

                row.append(cell)

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{sc}")
    lines.append(r"\end{center}")
    lines.append(r"\vskip -0.1in")
    lines.append(r"\end{table}")

    latex_table = "\n".join(lines)

    if not os.path.exists(TBL_PATH):
        os.makedirs(TBL_PATH)

    with open(TBL_PATH + "ood_evaluation.txt", "w") as f:
        f.write(latex_table)



def main(args):
    # Only download the data if the flag is set or there is no local data file
    if args.download or not os.path.exists(DATA_FILE.format(file_name=args.tag)):
        wandb_get_data(args.tag)

    # Group and plot the data
    data = group_data(args.tag)
    plot_performance(data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="OoD_Evaluation")
    parser.add_argument("--download", action="store_true")
    args, _ = parser.parse_known_args()

    main(args)
