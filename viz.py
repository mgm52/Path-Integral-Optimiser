import hydra
import numpy as np
import pandas as pd
import torch
from src.datamodules.datasets.opt_mini import OptMini
import os
import yaml
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath} \boldmath']
import matplotlib.cm as cm
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.markers import MarkerStyle

def read_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def get_training_metrics_df(dir_path):
    # Get all subfolders in csv folder
    csv_dir_path = os.path.join(dir_path, "csv")
    subfolders = [f.path for f in os.scandir(csv_dir_path) if f.is_dir()]
    
    for subfolder in subfolders:
        metrics_path = os.path.join(subfolder, "metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            return df
    return None

def get_final_validation_loss(dir_path):
    log_path = os.path.join(dir_path, "pis_optim_pl.log")
    final_validation_loss = None
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines:
                if 'validation loss:' in line:
                    final_validation_loss = float(line.split(':')[-1].strip())
    return final_validation_loss

def get_final_test_loss(dir_path):
    log_path = os.path.join(dir_path, "pis_optim_pl.log")
    final_test_loss = None
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines:
                if 'test loss:' in line:
                    final_test_loss = float(line.split(':')[-1].strip())
    return final_test_loss

def get_first_timestamp(dir_path):
    log_path = os.path.join(dir_path, "pis_optim_pl.log")
    first_timestamp = None
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            if lines:
                first_line = lines[0]
                first_timestamp = datetime.strptime(first_line.split('[')[1].split(']')[0], "%Y-%m-%d %H:%M:%S,%f")
    return first_timestamp

def get_final_timestamp(dir_path):
    log_path = os.path.join(dir_path, "pis_optim_pl.log")
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            if lines:
                last_line = lines[-1]
                final_timestamp = datetime.strptime(last_line.split('[')[1].split(']')[0], "%Y-%m-%d %H:%M:%S,%f")
                return final_timestamp
    return None

def get_config_yaml(dir_path):
    config_path = os.path.join(dir_path, ".hydra", "config.yaml")
    if os.path.exists(config_path):
        return read_yaml(config_path), config_path
    return None, None

def get_best_w(dir_path):
    # look for .pt file starting with best_final
    best_w_path = None
    for file in os.listdir(dir_path):
        if file.startswith("best_final"):
            best_w_path = os.path.join(dir_path, file)
            break
    # load into torch
    return torch.load(best_w_path) if best_w_path else None

def get_run_data(dir_path):
    final_val_loss = get_final_validation_loss(dir_path)
    final_test_loss = get_final_test_loss(dir_path)
    final_timestamp = get_final_timestamp(dir_path)
    metrics_df = get_training_metrics_df(dir_path)
    config, config_path = get_config_yaml(dir_path)
    best_w = get_best_w(dir_path)
    return {
        'run_folder': dir_path,
        'final_val_loss': final_val_loss,
        'final_test_loss': final_test_loss,
        'final_timestamp': final_timestamp,
        'metrics_df': metrics_df,
        'config': config,
        'config_path': config_path,
        'best_w': best_w
    }

def get_all_run_data(multirun_root_dir):
    run_datas = []

    # Get a list of all direct subdirectories
    subdirs = [d for d in os.listdir(multirun_root_dir) if os.path.isdir(os.path.join(multirun_root_dir, d))]

    # Iterate over each subdirectory and fetch the final V_min value and timestamp
    for subdir in subdirs:
        subdir_path = os.path.join(multirun_root_dir, subdir)
        run_datas.append(get_run_data(subdir_path))

    run_datas.sort(key=lambda x: x['final_timestamp'])

    for data in run_datas:
        print(f"Run folder: {data['run_folder']}")
        print(f"Final validation loss: {data['final_val_loss']}")
        print(f"Final timestamp: {data['final_timestamp']}\n")

    earliest_timestamp = get_first_timestamp(os.path.join(multirun_root_dir, "0"))

    return run_datas, earliest_timestamp

def plot_multirun_final_loss_over_time(run_datas):
    # Extract data from run_datas
    timestamps = [data['final_timestamp'] for data in run_datas]
    validation_losses = [data['final_val_loss'] for data in run_datas]

    # Convert timestamps to minutes relative to the earliest timestamp
    timestamps_in_minutes = [(t - earliest_timestamp).total_seconds() / 60 for t in timestamps]

    # Plotting
    plt.figure(figsize=(8,4))
    plt.scatter(timestamps_in_minutes, validation_losses, marker=MarkerStyle('x'))
    plt.title('Final Task Validation Loss Over Time')
    plt.xlabel(r'\textbf{Time Since Start (minutes)}')
    plt.ylabel(r'\textbf{Final Task Validation Loss}')

    # Add a trend line
    z = np.polyfit(timestamps_in_minutes, validation_losses, 1)
    p = np.poly1d(z)
    plt.plot(timestamps_in_minutes, p(timestamps_in_minutes), "r--")

    plt.savefig(f"{root_dir}/final_validation_loss_over_time.pdf")
    plt.close()

def plot_run_train_loss_over_time(run_data):
    df = run_data['metrics_df']

    x = df.index
    v_median = df['V_median']
    v_min = df['V_min']
    v_max = df['V_max']

    plt.figure(figsize=(6,4))
    plt.plot(x, v_median, label='V_median', color='blue')
    plt.fill_between(x, v_median, v_min, color='blue', alpha=0.2)
    plt.fill_between(x, v_median, v_max, color='blue', alpha=0.2)

    plt.xlabel(r"\textbf{Step}")
    plt.ylabel(r'\textbf{Loss}')
    plt.title('Median (+/- max/min over trajectories) \nTask Training Loss for Best Run')
    plt.legend()
    plt.savefig(f"{root_dir}/v_median.pdf")
    plt.close()

def plot_multirun_all_train_loss_over_time(run_datas):
    for run_data in run_datas:
        df = run_data['metrics_df']

        x = df.index
        v_median = df['V_median']
        v_min = df['V_min']
        v_max = df['V_max']

        plt.plot(x, v_median, label='V_median')
        #plt.fill_between(x, v_median, v_min, color='blue', alpha=0.2)
        #plt.fill_between(x, v_median, v_max, color='blue', alpha=0.2)

    plt.xlabel(r"\textbf{Step}")
    plt.ylabel(r'\textbf{Value}')
    plt.title('Median Task Training Loss for All Runs')
    plt.savefig(f"{root_dir}/v_median_all_runs.pdf")
    plt.close()

def plot_multirun_all_min_train_loss_over_time(run_datas):
    for run_data in run_datas:
        df = run_data['metrics_df']

        x = df.index
        v_median = df['V_median']
        v_min = df['V_min']
        v_max = df['V_max']

        plt.plot(x, v_min, label='V_min')
        #plt.fill_between(x, v_median, v_min, color='blue', alpha=0.2)
        #plt.fill_between(x, v_median, v_max, color='blue', alpha=0.2)

    plt.xlabel(r"\textbf{Step}")
    plt.ylabel(r'\textbf{Value}')
    plt.title('Median Task Training Loss for All Runs')
    plt.savefig(f"{root_dir}/v_min_all_runs.pdf")
    plt.close()

def plot_multirun_combined_train_loss_over_time(run_datas_list):
    # Loop over run_datas in run_datas_list
    # TODO: consider limiting to shortest-length run_datas
    for i, run_datas in enumerate(run_datas_list):
        v_mins = []

        for run_data in run_datas:
            df = run_data['metrics_df']
            v_mins.append(df['V_min'])

        # Assuming all runs have the same steps
        x = run_datas[0]['metrics_df'].index

        # Get minimum, median, and maximum of V_min over runs
        CI95_values = np.percentile(v_mins, [2.5, 97.5], axis=0)
        med_values = np.median(v_mins, axis=0)

        plt.plot(x, med_values, label=f'Median V_min for run_datas {i+1}')
        plt.fill_between(x, CI95_values[0], med_values, alpha=0.2, color=f'C{i}')
        plt.fill_between(x, med_values, CI95_values[1], alpha=0.2, color=f'C{i}')

    # Finalize the plot
    plt.xlabel(r"\textbf{Step}")
    plt.ylabel(r'\textbf{Value}')
    plt.title('Run-Median Trajectory-Best \n Task Training Loss for All Runs (95% CI)')
    plt.legend()
    plt.savefig(f"{root_dir}/v_min_combined_runs.pdf")
    plt.close()

def plot_run_task_viz(run_data):
    if run_data["best_w"] is None:
        print("No best w found for this run")
        return
    
    task = hydra.utils.instantiate(run_data["config"]["task"])
    ts_model = hydra.utils.instantiate(
        run_data["config"]["task_solving_model"], DATASIZE=task.datasize(), GTSIZE=task.gtsize(), _convert_="partial"
    )
    w = run_data["best_w"]

    task.viz(ts_model, w, "PIO", root_dir)

def flatten_dict(dd, separator='_', prefix=''):
    return {f"{prefix}{separator}{k}" if prefix else k : v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
           } if isinstance(dd, dict) else {prefix: dd}

def get_changing_params(run_datas):
    configs = [
        flatten_dict(rd["config"]) for rd in run_datas
    ]
    flat_configs_df = pd.DataFrame(configs)

    # might want to convert some categorical data to numerical
    # e.g. df['logger'] = df['logger'].apply(lambda x: 1 if x == 'csv' else 0)

    # Identify which parameters change between runs
    def get_hashable_columns(df):
        return [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (list, dict))).sum() == 0]

    hashable_cols = get_hashable_columns(flat_configs_df)
    hashable_df = flat_configs_df[hashable_cols]

    changing_params = hashable_df.columns[hashable_df.nunique() > 1]
    return changing_params, flat_configs_df

def plot_multirun_param_correlations(run_datas):
    changing_params, df = get_changing_params(run_datas)

    # change column names to remove prefixes
    df = df.rename(columns=lambda x: x.replace('model_', ''))
    df = df.rename(columns=lambda x: x.replace('datamodule_dataset_', ''))
    changing_params = [p.replace('model_', '') for p in changing_params]
    changing_params = [p.replace('datamodule_dataset_', '') for p in changing_params]

    # For each changing parameter, calculate the correlation with the final validation loss
    df['final_val_loss'] = [rd['final_val_loss'] for rd in run_datas]
    correlations = df[changing_params].apply(lambda x: x.corr(df['final_val_loss']))

    # Generate color palette based on the y value's magnitude
    #norm = mcolors.Normalize(correlations.values.min(), correlations.values.max())
    norm = mcolors.Normalize(-1, 1)
    #colors = cm.get_cmap('coolwarm')(norm(np.abs(correlations.values)))
    colors = cm.get_cmap('RdYlGn')(norm(correlations.values))
    plt.figure(figsize=(8, 4))
    sns.barplot(x=correlations.index, y=correlations.values, palette=colors)
    plt.ylim(-1, 1)
    plt.axhline(0, color='black', linewidth=0.5)
    #plt.title('Correlation between parameters and final validation loss')
    plt.xlabel(r"\textbf{Parameter}")
    plt.ylabel(r"\textbf{Correlation}")
    #plt.xticks(rotation=45)
    plt.savefig(f"{root_dir}/correlations.pdf")
    plt.close()

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the root directory as an argument.")
        sys.exit(1)
    root_dir = sys.argv[1]

    if "multiruns/" in root_dir:
        # Read runs
        run_datas, earliest_timestamp = get_all_run_data(root_dir)

        # Plot sweep stats
        plot_multirun_final_loss_over_time(run_datas)
        plot_multirun_param_correlations(run_datas)
        plot_multirun_all_train_loss_over_time(run_datas)
        plot_multirun_all_min_train_loss_over_time(run_datas)
        plot_multirun_combined_train_loss_over_time([run_datas])
        
        # Plot best run
        best_run = min(run_datas, key=lambda x: x['final_val_loss'])
        run_data = best_run
    else:
        run_data = get_run_data(root_dir)

    plot_run_train_loss_over_time(run_data)
    plot_run_task_viz(run_data)
