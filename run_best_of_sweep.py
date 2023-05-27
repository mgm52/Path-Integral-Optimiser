import subprocess
import omegaconf
from viz import get_all_run_data
from src.pis_optim_pl import run_with_config

if __name__ == "__main__":
    root_dir = 'logs/multiruns/2023-05-25_08-30-09_pio-sweep_NOpt'  # replace this with your actual root directory

    # Read runs
    run_datas, earliest_timestamp = get_all_run_data(root_dir)

    # Get best run
    best_run = min(run_datas, key=lambda x: x['final_val_loss'])

    config_path = best_run['config_path']
    config_filename = root_dir.split("/")[-1]

    # copy file into configs/experiment
    subprocess.run(["cp", config_path, f"configs/experiment/{config_filename}.yaml"])

    # prepend with # @package _global_
    subprocess.run(["sed", "-i", "1i # @package _global_", f"configs/experiment/{config_filename}.yaml"])

    #config = best_run['config']
    #load from configs/logger/csv.yaml and configs/logger/wandb.yaml using omegaconf
    #config["logger"]["csv"] = omegaconf.OmegaConf.load("configs/logger/csv.yaml")
    #config["logger"]["wandb"] = omegaconf.OmegaConf.load("configs/logger/wandb.yaml")

    #print(f"Gonna run with config: {config}")
    #run_with_config(omegaconf.OmegaConf.create(config)) # type: ignore

    print(f"Gonna run with config_path: {config_path}")

    python_script_path  = "src/pis_optim_pl.py"
    subprocess.run(["python", python_script_path, f"logger=[csv,wandb]", f"experiment={config_filename}", f"trainer.max_steps=1000", f"seed=42"])