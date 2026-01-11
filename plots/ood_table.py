import pandas as pd
import numpy as np
import wandb as wb
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm



IMG_PATH = "images/"
DATA_PATH = "data/"

#
#   Get the data
#

def wandb_get_data(tag: str):
    print(f"Trying to load data with tag \"{tag}\"")
    api = wb.Api()
    runs = api.runs(
        path="bongni/bachelor-thesis", 
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

    data.to_csv(f"./data/{tag}.csv", index=False)



def group_data(file_name):
    data = pd.read_csv(f'./data/{file_name}.csv')

    grouped_data = data.groupby(["acquisition_function", "_step"], as_index=False)

    # Calculate standard error for each group
    variance_mean = grouped_data["variance"].mean().rename(columns={"variance": "variance_mean"})
    entropy_mean = grouped_data["entropy"].mean().rename(columns={"entropy": "entropy_mean"})
    variance_stddev = grouped_data["variance"].sem().rename(columns={"variance": "variance_stddev"})
    entropy_stddev = grouped_data["entropy"].sem().rename(columns={"entropy": "entropy_stddev"})

    merged = pd.merge(variance_mean, entropy_mean, on=['acquisition_function', '_step'])
    merged = pd.merge(merged, variance_stddev, on=['acquisition_function', '_step'])
    merged = pd.merge(merged, entropy_stddev, on=['acquisition_function', '_step'])

    return merged

def group_data_by_space(file_name):
    data = pd.read_csv(f'./data/{file_name}.csv')

    grouped_data = data.groupby(["acquisition_function", "space", "_step"], as_index=False)

    # Calculate standard error for each group
    variance_mean = grouped_data["variance"].mean().rename(columns={"variance": "variance_mean"})
    entropy_mean = grouped_data["entropy"].mean().rename(columns={"entropy": "entropy_mean"})
    variance_stddev = grouped_data["variance"].sem().rename(columns={"variance": "variance_stddev"})
    entropy_stddev = grouped_data["entropy"].sem().rename(columns={"entropy": "entropy_stddev"})

    merged = pd.merge(variance_mean, entropy_mean, on=['acquisition_function', "space", '_step'])
    merged = pd.merge(merged, variance_stddev, on=['acquisition_function', "space", '_step'])
    merged = pd.merge(merged, entropy_stddev, on=['acquisition_function', "space", '_step'])

    return merged

#
#   Plot the results
#

FONTSIZE = 16

FUNCTIONS = ['uncertainty_sampling', 'alternative', 'original', 'noisy']
SPACES = ["2d-grid-overlapping", "2d-grid-disjoint", "2d-grid-target_in_sample", "2d-grid-sample_in_target"]

SPACE_TITLE = {
    "2d-grid-overlapping":          "Overlapping",
    "2d-grid-disjoint":             "Disjoint",
    "2d-grid-target_in_sample":     "Target in Sample",
    "2d-grid-sample_in_target":     "Sample in Target",
}

ALG_TITLE = {
    "original":                     "ITL noiseless",
    "alternative":                  "ITL noiseless alternative",
    "noisy":                        "ITL",
    "uncertainty_sampling":         "Uncertainty sampling",
    "uniform_sampling":             "Uniform sampling"
}

FUNC_COLORS = {
    'uncertainty_sampling' : 'gray', 
    'alternative' : 'red', 
    'original' : 'green', 
    'noisy' : 'black'
}

ALPHA = 0.15

def plotPerformance(fig, df, column: str):
    for func in FUNCTIONS:
        func_group = df.loc[df['acquisition_function'] == func]

        # Plot variance
        fig.plot(
            np.arange(100), 
            func_group[f'{column}_mean'], 
            label=ALG_TITLE[func], 
            color=FUNC_COLORS[func]
        )
        
        # Plot standard error
        n = func_group.groupby('_step').size()[0]
        fig.fill_between(
            np.arange(100), 
            func_group[f'{column}_mean'] - func_group[f'{column}_stddev'] / np.sqrt(n), 
            func_group[f'{column}_mean'] + func_group[f'{column}_stddev'] / np.sqrt(n), 
            color=FUNC_COLORS[func], 
            alpha=ALPHA
        )

    # Set labels
    fig.set_xlabel('Step', fontsize=FONTSIZE)
    fig.set_ylabel(column.capitalize(), fontsize=FONTSIZE)
    fig.set_title(f'{column.capitalize()} of Acquisition Functions', fontsize=FONTSIZE)
