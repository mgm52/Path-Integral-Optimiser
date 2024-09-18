import subprocess
import sys
import omegaconf
from viz import get_all_run_data
from src.pis_optim_pl import run_with_config

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide the root directory and no. of seeds as arguments.")
        sys.exit(1)
    root_dir = sys.argv[1]
    num_seeds = int(sys.argv[2])
    experiment_name_extra = sys.argv[3] if len(sys.argv) > 3 else ""
    num_workers = sys.argv[4] if len(sys.argv) > 4 else 1

    # root_dir = 'logs/multiruns/2023-05-25_08-30-09_pio-sweep_NOpt'  # replace this with your actual root directory

    # Read runs
    run_datas = get_all_run_data(root_dir)

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
    # Load basic sweeper (hacky)
    #subprocess.run(["mv", "configs/config.yaml", "configs/config_nevergrad.yaml"])
    #subprocess.run(["mv", "configs/config_basic.yaml", "configs/config.yaml"])
    # hackhack
    to_string = lambda lst: f"choice({', '.join(map(str, lst))})"
    subprocess.run(["python", python_script_path, f"logger=[csv]", f"experiment={config_filename}", f"trainer.max_steps=100", f"seed={to_string(range(10000))}", f"hydra.sweeper.optim.budget={num_seeds}",  f"name='seeds-{experiment_name_extra}'", "hydra/sweeper=nevergrad", f"hydra.sweeper.optim.num_workers={num_workers}", f"-m"])
    # Restore nevergrad sweeper
    #subprocess.run(["mv", "configs/config.yaml", "configs/config_basic.yaml"])
    #subprocess.run(["mv", "configs/config_nevergrad.yaml", "configs/config.yaml"])