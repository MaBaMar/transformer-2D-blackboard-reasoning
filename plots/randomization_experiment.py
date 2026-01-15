# ------------------------------------------------------------
# randomization_expermiment.py
#
# used to generate latex_table for the paper in a convenient manner
# ------------------------------------------------------------


import argparse
import os

import pandas as pd
import numpy as np
import wandb as wb
import matplotlib.pyplot as plt
import re

from datetime import datetime
from tqdm import tqdm



IMG_PATH = "./plots/images/"
TBL_PATH = "./plots/tables/"
DATA_PATH = "./plots/data/"
DATA_FILE = "./plots/data/{file_name:s}.csv"


def compare_matched_runs(df, metric_col='accuracy'):
    # 1. Identify all columns that define "other parameters"
    # We exclude the boolean we are testing, the target metric, and run-specific IDs
    ignore_cols = ['TrainPos', metric_col, 'run_name', 'run_id', 'timestamp', '_step', '_runtime', '_timestamp']
    group_cols = [c for c in df.columns if c not in ignore_cols]

    # 2. Pivot the data so TrainPos True and False are side-by-side for the same config
    # We use 'mean' in case there are multiple runs for the exact same seed/config
    pivoted = df.pivot_table(
        index=group_cols,
        columns='TrainPos',
        values=metric_col,
        aggfunc='mean'
    ).dropna() # drop cases where we don't have both True and False for a config

    # 3. Calculate how many times True > False
    total_matches = len(pivoted)
    true_higher = (pivoted[True] > pivoted[False]).sum()
    false_higher = (pivoted[False] > pivoted[True]).sum()
    ties = (pivoted[True] == pivoted[False]).sum()

    print(f"Total matched pairs found: {total_matches}")
    print(f"TrainPos=True is better: {true_higher} times ({(true_higher/total_matches)*100:.1f}%)")
    print(f"TrainPos=False is better: {false_higher} times ({(false_higher/total_matches)*100:.1f}%)")
    print(f"Ties: {ties}")

    return pivoted
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
    print("Getting runs...")

    for run in tqdm(runs):
        # 1. Get history and config
        df_history = run.history()
        df_config = pd.concat([pd.DataFrame([run.config])] * df_history.shape[0], ignore_index=True)

        # 2. Extract booleans from the run name
        # We look for 'TrainPos' or 'EvalPos' followed by True or False
        train_pos_match = re.search(r'TrainPos(True|False)', run.name)
        eval_pos_match = re.search(r'EvalPos(True|False)', run.name)

        # Convert the string match to a Python boolean
        train_pos = train_pos_match.group(1) == "True" if train_pos_match else None
        eval_pos = eval_pos_match.group(1) == "True" if eval_pos_match else None

        # 3. Combine data
        df_run = pd.concat([df_config, df_history], axis=1)

        # 4. Add the extracted metadata columns
        df_run['TrainPos'] = train_pos
        df_run['EvalPos'] = eval_pos
        df_run['run_name'] = run.name # Often helpful for debugging

        all_data.append(df_run)

    if not all_data:
        print("No runs found for the given tag.")
        return

    data = pd.concat(all_data, ignore_index=True)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    data.to_csv(DATA_FILE.format(file_name=tag), index=False)
    print(f"Successfully saved data to {DATA_FILE.format(file_name=tag)}")



def group_data(file_name):
    data = pd.read_csv(DATA_FILE.format(file_name=file_name))

    grouped_data = data.groupby(["model", "digits", "operation", "TrainPos", "EvalPos"], as_index=False)

    # train_rand_true = data[data["TrainPos"] == True].drop("TrainPos")
    # train_rand_false = data[data["TrainPos"] == False].drop("TrainPos")


    compare_matched_runs(data[data["operation"]=="add"])

    # Calculate standard error for each group
    acc_mean = grouped_data["accuracy"].mean().rename(columns={"accuracy": "acc_mean"})
    acc_stddev = grouped_data["accuracy"].std().rename(columns={"accuracy": "acc_stddev"})

    merged = pd.merge(acc_mean, acc_stddev, on=["model", "digits", "operation", "TrainPos", "EvalPos"])

    return merged



#
#   Plot the results
#

def plot_performance(df):
    digits = [11, 12] #sorted(df["digits"].unique())
    models = ["EOgar-2d", "EOgar-1d", "CoT"]
    operations = ["add", "sub", "mixed"]

    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{}")
    lines.append(r"\label{table:randexp}")
    lines.append(r"\vskip 0.15in")
    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(r"\begin{sc}")

    col_spec = "l" + "c" * 2*len(digits)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    for model in models:
        if model == models[0]:
            header = (
                r"\textbf{" + model + "}"
                + " & "
                + " & ".join(f"{d}({m})" for m in ["F", "R"] for d in digits)
                + r" \\"
            )
            lines.append(header)
        else:
            lines.append(
                rf"\multicolumn{{{len(digits)+1}}}{{l}}{{\textbf{{{model}}}}} \\"
            )

        lines.append(r"\midrule")

        for op in operations:
            for base_r in [False, True]:
                row = [op.capitalize() + "(R)" if base_r else "(F)"]

                for d in digits:
                    for eval_r in [False, True]:
                        m = df[
                            (df["model"] == model)
                            & (df["operation"] == op)
                            & (df["digits"] == d)
                            & (df["TrainPos"] == base_r)
                            & (df["EvalPos"] == eval_r)
                        ]

                        if len(m) == 1:
                            mean = m["acc_mean"].values[0] * 100
                            std = m["acc_stddev"].values[0] * 100
                            cell = rf"\scriptsize{{{mean:.1f}$\pm${std:.1f}}}"
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

    with open(TBL_PATH + "random_exp.txt", "w") as f:
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
    parser.add_argument("--tag", type=str, default="randomization_expermiment")
    parser.add_argument("--download", action="store_true")
    args, _ = parser.parse_known_args()

    main(args)
