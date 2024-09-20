import subprocess
import sys
import os
import glob

def newest_directory_in_logs(log_dir):
    # List all directories in the logs directory
    all_subdirs = [os.path.join(log_dir, name) for name in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, name))]

    # Return the directory with the latest modification time
    return max(all_subdirs, key=os.path.getmtime)

def run_experiment(experiment_name, optimizer_name, extra_choice=False):
    # Define the root directory for logs
    log_dir = "logs/multiruns"

    # Define sweep command
    command = ["python", "src/pis_optim_pl.py", f"experiment={experiment_name}.yaml", f"name=sweep-{experiment_name}", f"model.optimizer={optimizer_name}",
               # Hyperparameters to sweep
               "model.lr=tag(log,interval(0.0001,0.1))",
               "trainer.gradient_clip_val=tag(log, interval(0.01,4.95))",
               # Sweeper config
               "hydra/sweeper=nevergrad", "logger=csv", "-m"]
    
    if extra_choice:
        command.insert(9, "model.batch_laps=choice(1,4,16)")
        if optimizer_name == 'pis':
            command.insert(9, "model.dt=interval(0.01, 0.1)")
            command.insert(9, "model.t_end=interval(1.0, 10.0)")
            command.insert(9, "model.f_func.num_layers=range(1,3)")
        if optimizer_name == 'adam':
            command.insert(9, "model.b1=tag(log,interval(0.81,0.98))")
            command.insert(9, "model.b2=tag(log,interval(0.91,0.9998))")
        command.insert(9, "trainer.max_steps=64")
        command.insert(9, "hydra.sweeper.optim.budget=64")
    else:
        command.insert(9, "model.batch_laps=1")
        command.insert(9, "trainer.max_steps=32")
        command.insert(9, "hydra.sweeper.optim.budget=32")

    if optimizer_name in ['pis', 'pis-mc']:
        if not "schedule" in experiment_name:
            command.insert(9, "datamodule.dataset.sigma=tag(log, interval(0.0001, 0.1))")
        else:
            command.insert(9, "callbacks.pissigma.sigma_factor=tag(log, interval(0.01, 10))")
    
    if optimizer_name == 'pis-mc':
        command.insert(9, "model.m_monte_carlo=int(tag(log,interval(2,2048)))")

    if "carrillo" in experiment_name:
        command.insert(9, "model.batch_size=1")
    else:
        command.insert(9, "model.batch_size=int(tag(log,interval(16,128)))")

    if "mnist" in experiment_name:
        command.insert(9, "hydra.sweeper.optim.num_workers=1")
        

    # Run sweep
    subprocess.run(command)
    sweep_directory = newest_directory_in_logs(log_dir)
    # Check that sweep was successful by checking whether optimization_results.yaml exists:
    if not os.path.exists(os.path.join(sweep_directory, "optimization_results.yaml")):
        print("WARNING: Sweep failed!")
        #sys.exit(1)
    subprocess.run(["python", "viz.py", sweep_directory])

    # Run seeding
    print("ABOUT TO START SEEDING")
    num_seeds = "5" if "mnist" in experiment_name else "10"
    #num_workers = "1" if "mnist" in experiment_name else "5"
    num_workers = "1"
    subprocess.run(["python", "run_best_of_sweep.py", sweep_directory, num_seeds, experiment_name, num_workers])
    seed_directory = newest_directory_in_logs(log_dir)
    # Check that seeding was successful by checking whether optimization_results.yaml exists:
    if not os.path.exists(os.path.join(seed_directory, "optimization_results.yaml")):
        print("WARNING: Seeding failed!")
        #sys.exit(1)
    subprocess.run(["python", "viz.py", seed_directory])

    return seed_directory


if __name__ == "__main__":
    experiments = ["opt_mini"] #, ""
    optimizers = ["pis-mc"]
    extra_choice = False

    #run_experiment("opt_mnist", "sgd")
    #pass

    for experiment in experiments:
        seed_directories = []
        for optimizer in optimizers:
            print(f"Running experiment {experiment} with optimizer {optimizer}")
            seed_directory = run_experiment(experiment, optimizer, extra_choice)
            seed_directories.append(seed_directory)

        # Visualize all seeded results
        subprocess.run(["python", "viz.py"] + seed_directories)
