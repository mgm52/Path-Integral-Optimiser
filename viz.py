import hashlib
import shutil
import subprocess
import time
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
from src.utils.nn_creation import visualize_weights, set_params

def read_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# TODO: refactor all these... yikes
def get_training_metrics_df(dir_path):
    # Get all subfolders in csv folder
    csv_dir_path = os.path.join(dir_path, "csv")
    subfolders = [f.path for f in os.scandir(csv_dir_path) if f.is_dir()]
    
    for subfolder in subfolders:
        metrics_path = os.path.join(subfolder, "metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            df["time"] = df["time"] - df["time"].iloc[0]
            return df
    return None

def get_final_validation_loss(dir_path):
    log_path = os.path.join(dir_path, "training_coordinator.log")
    final_validation_loss = None
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines:
                if 'validation loss:' in line:
                    final_validation_loss_str = line.split(':')[-1].strip()
                    if final_validation_loss_str == "nan":
                        final_validation_loss = 99999
                    else:
                        final_validation_loss = float(line.split(':')[-1].strip())
    return final_validation_loss

def get_final_test_loss(dir_path):
    log_path = os.path.join(dir_path, "training_coordinator.log")
    final_test_loss = None
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines:
                if 'test loss:' in line:
                    final_test_loss_str = line.split(':')[-1].strip()
                    if final_test_loss_str == "nan":
                        final_test_loss = 99999
                    else:
                        final_test_loss = float(line.split(':')[-1].strip())
    return final_test_loss

def get_first_timestamp(dir_path):
    log_path = os.path.join(dir_path, "training_coordinator.log")
    first_timestamp = None
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            if lines:
                first_line = lines[0]
                first_timestamp = datetime.strptime(first_line.split('[')[1].split(']')[0], "%Y-%m-%d %H:%M:%S,%f")
    return first_timestamp

def get_final_timestamp(dir_path):
    log_path = os.path.join(dir_path, "training_coordinator.log")
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
    else:
        print(f"WARNING: config not found at {config_path}")
    return None, None

def get_best_w(dir_path):
    # look for .pt file starting with best_final
    best_w_path = None
    for file in os.listdir(dir_path):
        if file.startswith("best_final"):
            best_w_path = os.path.join(dir_path, file)
            break
    # load into torch
    return torch.load(best_w_path).squeeze() if best_w_path else None

def expand_metrics_to_V(metrics_df):
    if "train_loss" in metrics_df.keys():
        for v_param in ["V_min", "V_median", "V_max"]:
            metrics_df[v_param] = metrics_df["train_loss"]
    return metrics_df

def generate_approx_time(metrics_df, first_timestamp, final_timestamp):
    # generate a new column, which represents the difference between first_timestamp and final_timestamp
    # and is linearly spaced
    if first_timestamp and final_timestamp:
        t1 = first_timestamp.timestamp()
        t2 = final_timestamp.timestamp()
        time_diff = t2 - t1
        metrics_df["time"] = np.linspace(0, time_diff, len(metrics_df))
    return metrics_df

def replace_optimizer_name(config):
    if config['model']["optimizer"] == "pis":
        config['model']["optimizer"] = "pio"
    return config

def get_run_data(dir_path):
    final_val_loss = get_final_validation_loss(dir_path)
    final_test_loss = get_final_test_loss(dir_path)
    final_timestamp = get_final_timestamp(dir_path)
    first_timestamp = get_first_timestamp(dir_path)
    metrics_df = expand_metrics_to_V(get_training_metrics_df(dir_path))
    metrics_df = generate_approx_time(metrics_df, first_timestamp, final_timestamp)
    config, config_path = get_config_yaml(dir_path)
    config = replace_optimizer_name(config)
    best_w = get_best_w(dir_path)
    task_name = "Carrillo" if "carrillo" in config['task']['_target_'] else ("MNIST" if "mnist" in config['task']['_target_'] else ("Moons" if "moons" in config['task']['_target_'] else "Task"))
    return {
        'run_folder': dir_path,
        'final_val_loss': final_val_loss,
        'final_test_loss': final_test_loss,
        'first_timestamp': first_timestamp,
        'final_timestamp': final_timestamp,
        'metrics_df': metrics_df,
        'config': config,
        'config_path': config_path,
        'best_w': best_w,
        'task_name': task_name
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

    return run_datas

def plot_multirun_final_loss_over_time(run_datas, log_scale=True):
    # Extract data from run_datas
    timestamps = [data['final_timestamp'] for data in run_datas]
    validation_losses = [data['final_val_loss'] for data in run_datas]

    # Remove NaN and None values from both lists
    validation_losses, timestamps = zip(*[(v, t) for v, t in zip(validation_losses, timestamps) 
                                          if v is not None and not np.isnan(v)])

    # Convert timestamps to minutes relative to the earliest timestamp
    earliest_timestamp = min([data['first_timestamp'] for data in run_datas])
    timestamps_in_minutes = [(t - earliest_timestamp).total_seconds() / 60 for t in timestamps]

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.scatter(timestamps_in_minutes, validation_losses, marker=MarkerStyle('x'))
    plt.title('Final Task Validation Loss')
    plt.xlabel(r'\textbf{Time Since Start (minutes)}')
    plt.ylabel(r'\textbf{Final Task Validation Loss}')

    # Add a trend line
    z = np.polyfit(timestamps_in_minutes, validation_losses, 1)
    p = np.poly1d(z)
    plt.plot(timestamps_in_minutes, p(timestamps_in_minutes), "r--")

    plt.tight_layout()

    # Check if log scale can be applied: filter NaN, None, and non-positive values
    if log_scale and np.any(np.array(validation_losses) > 0):
        plt.yscale('log')
        plt.savefig(f"{root_dir}/final_validation_loss_over_time_log.pdf")
    else:
        if log_scale:
            print("Warning: Log scale cannot be applied due to entirely NaN or non-positive values in validation losses.")
        plt.savefig(f"{root_dir}/final_validation_loss_over_time.pdf")

    plt.close()

def plot_run_train_loss_over_time(run_data, log_scale=True, x_axis='step'):
    df = run_data['metrics_df']

    if x_axis == 'step':
        x = df.index
    elif x_axis == 'examples':
        batch_size = run_data["config"]["model"]["batch_size"]
        x = df.index * batch_size
    elif x_axis == 'time':
        x = df['time']
    else:
        raise ValueError(f"Unsupported x_axis value: {x_axis}")

    v_median = df['V_median']
    v_min = df['V_min']
    v_max = df['V_max']

    plt.figure(figsize=(6,4))
    plt.plot(x, v_median, label='V_median', color='blue')
    plt.fill_between(x, v_median, v_min, color='blue', alpha=0.2)
    plt.fill_between(x, v_median, v_max, color='blue', alpha=0.2)

    plt.xlabel(r"\textbf{" + x_axis.capitalize() + "}")
    plt.ylabel(r'\textbf{Loss}')
    plt.title('Median (+/- max/min over trajectories) \nTask Training Loss for Best Run')
    plt.legend()


    plt.tight_layout()
    if log_scale and np.any(v_median > 0):
        plt.yscale('log')
        plt.savefig(f"{root_dir}/v_median_{x_axis}_log.pdf")
    else:
        plt.savefig(f"{root_dir}/v_median_{x_axis}.pdf")

    plt.close()

def plot_run_lr_over_time(run_data, log_scale=True):
    df = run_data['metrics_df']
    x = df.index
    plt.figure(figsize=(6,4))

    lrs = df['lr']
    plt.plot(x, lrs, color='tab:blue')

    plt.grid(True, which="both", ls="-", alpha=0.3)
    if "sigma" in df:
        sigmas = df['sigma']
        plt.plot(x, sigmas, color='tab:red')
        plt.title('Learning Rate and $\sigma$')
    else:
        plt.title('Learning Rate')
    
    plt.xlabel(r"\textbf{Step}")
    plt.ylabel(r'\textbf{Value}')
    
    plt.tight_layout()
    if log_scale and np.any(lrs > 0):
        plt.yscale('log')
        plt.savefig(f"{root_dir}/lr_over_time_log.pdf")
    else:
        plt.savefig(f"{root_dir}/lr_over_time.pdf")
    plt.close()

def plot_run_gradients(run_data):
    df = run_data['metrics_df']
    x = df.index

    if "loss" in df: losses = df['loss']
    else: losses = df['train_loss']

    if not "pis_grad_max_preopt" in df:
        return
    
    maxgrads_preopt = df['pis_grad_max_preopt']
    maxgrads_postopt = df['pis_grad_max_postopt']
    medianVs = df['V_median']

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(10, 9)

    ax1.plot(x, losses, color='tab:blue')
    ax1.set_title("Loss")

    ax2.plot(x, maxgrads_preopt, color='tab:green')
    ax2.set_title("Max grad (pre-optimization)")

    ax3.plot(x, maxgrads_postopt, color='darkgreen')
    ax3.set_title("Max grad (post-optimization)")

    ax4.plot(x, medianVs, color='tab:purple')
    ax4.set_title("Median V(x) (i.e. task loss)")

    plt.xlabel(r"\textbf{Step}")

    plt.suptitle(f"Loss \& maxgrad")
    plt.tight_layout()
    plt.savefig(f"{root_dir}/gradients.pdf")
    plt.close()


def plot_multirun_all_train_loss_over_time(run_datas, log_scale=False):
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
    plt.ylabel(r'\textbf{Loss}')
    plt.title('Median Task Training Loss for All Runs')

    plt.tight_layout()
    if log_scale and np.any(v_median > 0):
        plt.yscale('log')
        plt.savefig(f"{root_dir}/v_median_all_runs_log.pdf")
    else:
        plt.savefig(f"{root_dir}/v_median_all_runs.pdf")
    plt.close()

def plot_multirun_all_min_train_loss_over_time(run_datas, log_scale=False):
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
    plt.ylabel(r'\textbf{Loss}')
    plt.title('Min Task Training Loss for All Runs')
    plt.tight_layout()
    if log_scale and np.any(v_min > 0):
        plt.yscale('log')
        plt.savefig(f"{root_dir}/v_min_all_runs_log.pdf")
    else:
        plt.savefig(f"{root_dir}/v_min_all_runs.pdf")
    plt.close()

import numpy as np
import matplotlib.pyplot as plt

def plot_multirun_combined_val_over_time(run_datas_list, labelling_config="optimizer", log_scale=False, x_axis='step', y_key="V_min", y_label="Loss"):
    shortest_run_end = None

    # Loop over run_datas in run_datas_list
    plt.figure(figsize=(4,4))

    for i, run_datas in enumerate(run_datas_list):
        v_mins = []
        x_values = []

        for run_data in run_datas:
            df = run_data['metrics_df']
            v_mins.append(df[y_key])

            if x_axis == 'step':
                x_values.append(df.index)
                # Update shortest_run_end if necessary
                if shortest_run_end is None or df.index[-1] < shortest_run_end:
                    shortest_run_end = df.index[-1]
            elif x_axis == 'examples':
                batch_size = run_data["config"]["model"]["batch_size"]
                x_values.append(df.index * batch_size)
                # Update shortest_run_end if necessary
                if shortest_run_end is None or (df.index[-1] * batch_size) < shortest_run_end:
                    shortest_run_end = df.index[-1] * batch_size
            elif x_axis == 'time':
                x_values.append(df['time'])
                # Update shortest_run_end if necessary
                if shortest_run_end is None or df['time'].iloc[-1] < shortest_run_end:
                    shortest_run_end = df['time'].iloc[-1]
            else:
                raise ValueError(f"Unsupported x_axis value: {x_axis}")

        # Assuming all runs have the same x values
        x = x_values[0]

        # Get minimum, median, and maximum of V_min over runs
        CI95_values = np.percentile(v_mins, [2.5, 97.5], axis=0)
        med_values = np.median(v_mins, axis=0)

        if labelling_config:
            plt.plot(x, med_values, label=f"{run_datas[0]['config']['model'][labelling_config]}")
        else:
            plt.plot(x, med_values, label=f"{i}")
        plt.fill_between(x, CI95_values[0], med_values, alpha=0.2, color=f"C{i}")
        plt.fill_between(x, med_values, CI95_values[1], alpha=0.2, color=f"C{i}")

    # Finalize the plot
    plt.xlabel(r"\textbf{" + x_axis.capitalize() + "}")
    plt.ylabel(r"\textbf{" + y_label + "}")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.title(f"{run_datas_list[0][0]['task_name']}:\nTask Training {y_label} over {sum([len(rd) for rd in run_datas_list])} Runs (95\% CI)")
    plt.legend()
    plt.tight_layout()
    low_y_key = y_key.lower()
    if log_scale and np.any(med_values > 0):
        plt.yscale('log')
        plt.savefig(f"{root_dir}/{low_y_key}_combined_runs_{x_axis}_log.pdf")
        print(f"{root_dir}/{low_y_key}_combined_runs_{x_axis}_log.pdf")
    else:
        plt.savefig(f"{root_dir}/{low_y_key}_combined_runs_{x_axis}.pdf")

    # If shortest_run_end is not None, create an additional plot with trimmed x-axis
    if shortest_run_end is not None:
        plt.xlim(left=0, right=shortest_run_end)  # Explicitly set the left limit to 0
        plt.title(f"Task Training {y_label} over {sum([len(rd) for rd in run_datas_list])} Runs (95\% CI)")
        if log_scale:
            plt.savefig(f"{root_dir}/{low_y_key}_combined_runs_{x_axis}_log_trimmed.pdf")
        else:
            plt.savefig(f"{root_dir}/{low_y_key}_combined_runs_{x_axis}_trimmed.pdf")

    plt.close()




def get_task_model_weights(run_data):
    task = hydra.utils.instantiate(run_data["config"]["task"])
    ts_model = hydra.utils.instantiate(
        run_data["config"]["task_solving_model"], DATASIZE=task.datasize(), GTSIZE=task.gtsize(), _convert_="partial"
    )
    w = run_data["best_w"].squeeze()
    return task, ts_model, w

def plot_run_task_viz(run_data):
    if run_data["best_w"] is None:
        print("No best w found for this run")
        return
    
    task, ts_model, w = get_task_model_weights(run_data)

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
    df = df.rename(columns=lambda x: x.replace('trainer_gradient', 'grad'))
    df = df.rename(columns=lambda x: x.replace('callbacks_', ''))
    df = df.rename(columns=lambda x: x.replace('f_func_', ''))
    changing_params = [p.replace('model_', '') for p in changing_params]
    changing_params = [p.replace('datamodule_dataset_', '') for p in changing_params]
    changing_params = [p.replace('trainer_gradient', 'grad') for p in changing_params]
    changing_params = [p.replace('callbacks', '') for p in changing_params]
    changing_params = [p.replace('f_func_', '') for p in changing_params]

    # For each changing parameter, calculate the correlation with the final validation loss
    df['final_val_loss'] = [rd['final_val_loss'] for rd in run_datas]
    correlations = df[changing_params].apply(lambda x: x.corr(df['final_val_loss']))

    if len(correlations) > 0:

        # Generate color palette based on the y value's magnitude
        #norm = mcolors.Normalize(correlations.values.min(), correlations.values.max())
        norm = mcolors.Normalize(-1, 1)
        #colors = cm.get_cmap('coolwarm')(norm(np.abs(correlations.values)))
        colors = cm.get_cmap('RdYlGn')(norm(correlations.values))
        plt.figure(figsize=(1.5 * len(correlations.index), 4))
        # add a horizontal grid
        plt.grid(axis='y', linestyle='-', alpha=0.5)
        sns.barplot(x=correlations.index, y=correlations.values, palette=colors)
        #plt.ylim(-1, 1)
        plt.axhline(0, color='black', linewidth=0.5)
        #plt.title('Correlation between parameters and final validation loss')
        plt.xlabel(r"\textbf{Parameter}")
        plt.ylabel(r"\textbf{Loss Correlation}")
        #plt.xticks(rotation=45)
        plt.savefig(f"{root_dir}/correlations.pdf")
        plt.close()
    else:
        print("WARNING: No correlations found")

import sys

def plot_run_weights(run_data):
    if run_data["best_w"] is None:
        print("No best w found for this run")
        return
    
    task, ts_model, w = get_task_model_weights(run_data)
    net = ts_model.get_trainable_net()
    if not isinstance(net, torch.nn.Parameter):
        set_params(net, w)
        visualize_weights(net, ts_model.__class__.__name__, root_dir)

def get_multirun_final_test_loss(run_datas):
    final_test_losses = [rd["final_test_loss"] for rd in run_datas]
    std = np.std(final_test_losses)
    mean = np.mean(final_test_losses)
    return mean, std, final_test_losses



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the root directory as an argument.")
        sys.exit(1)
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]

        if "multiruns/" in root_dir:
            # Read runs
            run_datas = get_all_run_data(root_dir)

            # Plot sweep stats
            plot_multirun_param_correlations(run_datas)
            for log_scale in [True, False]:
                plot_multirun_final_loss_over_time(run_datas, log_scale)
                plot_multirun_all_train_loss_over_time(run_datas, log_scale)
                plot_multirun_all_min_train_loss_over_time(run_datas, log_scale)
                for x_axis in ["step", "time", "examples"]:
                    plot_multirun_combined_val_over_time([run_datas], "optimizer", log_scale, x_axis)
                for x_axis in ["step", "time", "examples"]:
                    plot_multirun_combined_val_over_time([run_datas], "optimizer", log_scale, x_axis, "param_norm", "TSM Parameter Norm")

            mean_tl, std_tl, all_tl = get_multirun_final_test_loss(run_datas)
            with open(f"{root_dir}/final_test_loss.txt", "w") as f:
                f.write(f"{mean_tl} +/- {std_tl} over {len(run_datas)} runs ({all_tl})\n")

            # Plot best run
            best_run = min(run_datas, key=lambda x: x['final_val_loss'])
            run_data = best_run
        else:
            run_data = get_run_data(root_dir)

        for x_axis in ["step", "time", "examples"]:
            for log_scale in [True, False]:
                plot_run_train_loss_over_time(run_data, log_scale, x_axis)
        plot_run_task_viz(run_data)
        plot_run_weights(run_data)
        plot_run_gradients(run_data)
        for log_scale in [True, False]:
            plot_run_lr_over_time(run_data)
    else:
        # root_dir is a new directory
        root_dirs = sys.argv[1:]
        endings = [rd.split("/")[-1] for rd in root_dirs]
        root_dir = f"logs/multiruns/Z_{max(endings)}_comboX{len(sys.argv)-1}_{hashlib.md5(f'{endings}'.encode()).hexdigest()}"
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        

        for rd in root_dirs:
            if "pis" in rd:
                # start a viz.py subprocess for rd
                print(f"Located a PIS run: calling viz.py for {rd}")
                subprocess.run(["python", "viz.py", rd])
                # delete root_dir/pis-seed-viz
                if os.path.exists(f"{root_dir}/pis-seed-viz"):
                    shutil.rmtree(f"{root_dir}/pis-seed-viz")
                # copy rd directory to root_dir/pis-seed-viz
                shutil.copytree(rd, f"{root_dir}/pis-seed-viz")
                print(f"Copied {rd} to {root_dir}/pis-seed-viz")

        run_datas_list = [get_all_run_data(rd) for rd in root_dirs]
        # sort by optimizer name
        run_datas_list = sorted(run_datas_list, key=lambda x: x[0]["config"]["model"]["optimizer"])

        #run_datas_1, earliest_timestamp_1 = get_all_run_data(root_dir_1)
        #run_datas_2, earliest_timestamp_2 = get_all_run_data(root_dir_2)

        for x_axis in ["step", "time", "examples"]:
            for log_scale in [True, False]:
                plot_multirun_combined_val_over_time(run_datas_list, "optimizer", log_scale, x_axis)
        for x_axis in ["step", "time", "examples"]:
            for log_scale in [True, False]:
                plot_multirun_combined_val_over_time(run_datas_list, "optimizer", log_scale, x_axis, "param_norm", "TSM Parameter Norm")


        means_stds = [get_multirun_final_test_loss(run_datas) for run_datas in run_datas_list]
        with open(f"{root_dir}/final_test_losses.txt", "w") as f:
            # write each mean and std on a new line
            f.write("\n".join([f"{run_datas[0]['run_folder']}: {mean} +/- {std} over {len(run_datas)} runs \n({all_tl})\n" for (mean, std, all_tl), run_datas in zip(means_stds, run_datas_list)]))