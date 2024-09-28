import subprocess
import sys
import os
import glob
import uuid

def newest_directory_in_logs(log_dir, keyword_match=""):
    # List all directories in the logs directory
    all_subdirs = [os.path.join(log_dir, name) for name in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, name)) and keyword_match in name]

    # Return the directory with the latest modification time
    return max(all_subdirs, key=os.path.getmtime)

def run_experiment(experiment_name, optimizer_name, extra_choice=False, mc_mega_boost=False, mc_hundred=False, force_seed_steps=""):
    # Define the root directory for logs
    log_dir = "logs/multiruns"

    if not RUN_SEEDING_ONLY:

        # Define sweep command
        random_id = str(uuid.uuid4())
        command = ["python", "src/pis_optim_pl.py", f"experiment={experiment_name}.yaml", f"model.optimizer={optimizer_name}",
                # Hyperparameters to sweep
                "trainer.gradient_clip_val=tag(log, interval(0.01,4.95))",
                # Sweeper config
                "hydra/sweeper=nevergrad", "logger=[csv, wandb]", f"name=sweep-{experiment_name}", f"+logger.wandb.name=sweep-{experiment_name}-{optimizer_name}-{random_id}", "-m"]
        
        # TODO: make this whole file a python script... or at least stop inserting at index 8...
        if optimizer_name != "pis-mc":
            command.insert(8, "model.lr=tag(log,interval(0.0001,0.1))")

        if extra_choice:
            if mc_mega_boost and optimizer_name == "pis-mc":
                command.insert(8, "model.mc_ts_per_mc_step=tag(log,interval(0.0009,1.0))")
                command.insert(8, "model.mc_max_steps_total=1024")
                command.insert(8, "trainer.max_steps=1024")
                command.insert(8, "hydra.sweeper.optim.budget=64")
            elif mc_hundred and optimizer_name == "pis-mc":
                command.insert(8, "model.mc_ts_per_mc_step=choice(1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.04, 0.02, 0.01)")
                command.insert(8, "model.mc_max_steps_total=100")
                command.insert(8, "trainer.max_steps=100")
                command.insert(8, "hydra.sweeper.optim.budget=64")
            else:
                if optimizer_name in ["pis-mc", "pis"]:
                    command.insert(8, "model.batch_laps=choice(1,4,16)")
                if optimizer_name == 'pis-mc':
                    command.insert(8, "model.mc_ts_per_mc_step=choice(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)")
                    command.insert(8, "model.mc_max_steps_total=64")
                if optimizer_name == 'pis':
                    command.insert(8, "model.t_end=interval(1.0, 10.0)")
                    command.insert(8, "model.f_func.num_layers=range(1,3)")
                if optimizer_name == 'adam':
                    command.insert(8, "model.b1=tag(log,interval(0.81,0.98))")
                    command.insert(8, "model.b2=tag(log,interval(0.91,0.9998))")
                command.insert(8, "trainer.max_steps=64")
                command.insert(8, "hydra.sweeper.optim.budget=64")
        else:
            if optimizer_name == 'pis-mc':
                command.insert(8, f"model.mc_ts_per_mc_step=0.125")
                command.insert(8, "model.mc_max_steps_total=32")
            command.insert(8, "model.batch_laps=1")
            command.insert(8, "trainer.max_steps=32")
            command.insert(8, "hydra.sweeper.optim.budget=32")

        if optimizer_name in ['pis', 'pis-mc']:
            if not "schedule" in experiment_name:
                command.insert(8, "datamodule.dataset.sigma=tag(log, interval(0.0001, 0.1))")
            else:
                command.insert(8, "callbacks.pissigma.sigma_factor=tag(log, interval(0.01, 10))")
        
        if "carrillo" in experiment_name:
            command.insert(8, "model.batch_size=1")
        else:
            command.insert(8, "model.batch_size=int(tag(log,interval(16,128)))")

        if "mnist" in experiment_name:
            command.insert(8, "hydra.sweeper.optim.num_workers=1")
            

        # Run sweep
        subprocess.run(command)

    sweep_directory = newest_directory_in_logs(log_dir, "_sweep-")
    # Check that sweep was successful by checking whether optimization_results.yaml exists:
    if not os.path.exists(os.path.join(sweep_directory, "optimization_results.yaml")):
        print("WARNING: Sweep failed!")
        #sys.exit(1)
    if not RUN_SEEDING_ONLY:
        subprocess.run(["python", "viz.py", sweep_directory])
    if RUN_SEEDING_ONLY and ("sweep" not in sweep_directory):
        print("WARNING: Most recent folder is not a sweep! Skipping seeding...")
        sys.exit(1)

    # Run seeding
    print("ABOUT TO START SEEDING")
    num_seeds = "5" if "mnist" in experiment_name else "10"
    #num_workers = "1" if "mnist" in experiment_name else "5"
    num_workers = "1"
    if mc_mega_boost and optimizer_name == "pis-mc":
        test_steps = "1024"
    else:
        test_steps = "100"
    if len(force_seed_steps) > 0:
        test_steps = force_seed_steps
    subprocess.run(["python", "run_best_of_sweep.py", sweep_directory, num_seeds, test_steps, experiment_name, num_workers])
    seed_directory = newest_directory_in_logs(log_dir, "_seeds-")
    # Check that seeding was successful by checking whether optimization_results.yaml exists:
    if not os.path.exists(os.path.join(seed_directory, "optimization_results.yaml")):
        print("WARNING: Seeding failed!")
        #sys.exit(1)
    subprocess.run(["python", "viz.py", seed_directory])

    return seed_directory

RUN_SEEDING_ONLY = False

if __name__ == "__main__":
    experiments = ["opt_carrillo", "opt_mini", "opt_mnist"]
    optimizers = ["pis"] #["pis", "pis-mc", "adam", "sgd", "adagrad"]

    extra_choices = [False, True]
    mc_mega_boost = False
    mc_hundred = False

    for extra_choice in extra_choices:
        for experiment in experiments:
            seed_directories = []
            for optimizer in optimizers:
                print(f"Running experiment {experiment} with optimizer {optimizer}")
                seed_directory = run_experiment(
                    experiment,
                    optimizer,
                    extra_choice,
                    mc_mega_boost,
                    mc_hundred,
                    force_seed_steps=(("64" if extra_choice else "32") if optimizer == "pis-mc" else ""))
                seed_directories.append(seed_directory)

            # Visualize all seeded results
            subprocess.run(["python", "viz.py"] + seed_directories)