from typing import List
import hydra
from omegaconf import DictConfig

from src.models_tasksolve.base_ts_model import BaseTSModel
from src.tasks.base_task import BaseTask
from src.train_non_pio import start_sgd_train_loop
from src.train_pio import PISBasedOptimizer
from src.utils import lht_utils

from pytorch_lightning.loggers import LightningLoggerBase

log = lht_utils.get_logger(__name__)

# Kick off a training run!
def run_with_config(cfg: DictConfig):
    task: BaseTask = hydra.utils.instantiate(
        cfg.task, _convert_="partial"
    )
    ts_model: BaseTSModel = hydra.utils.instantiate(
        cfg.task_solving_model, DATASIZE=task.datasize(), GTSIZE=task.gtsize(), _convert_="partial"
    )
    OPTIM = cfg.model.optimizer

    val_loss = None
    if OPTIM in ["sgd", "adam", "adagrad", "pis-mc"]:
        val_loss, test_loss, w = start_sgd_train_loop(ts_model, task, cfg, OPTIM)

        ts_model.save_checkpoint(w, f"best_final_{ts_model.__class__.__name__}.pt")

        log.info("Final validation loss: " + str(val_loss))
        log.info("Final test loss: " + str(test_loss))

    elif OPTIM in ["pis"]:
        optimizer = PISBasedOptimizer(
            cfg=cfg,
            task=task,
            ts_model=ts_model)
        final_trajectories = optimizer.start_train_loop()
        final_val_task_losses = optimizer.task_weights_to_validation_loss(final_trajectories)
    
        min_val_loss = float(final_val_task_losses.min())
        best_trajectory = final_trajectories[final_val_task_losses.argmin()]

        final_test_loss_for_best_trajectory = float(optimizer.task_weights_to_test_loss(best_trajectory.unsqueeze(0)))

        task.viz(ts_model, best_trajectory, f"best_final_{ts_model.__class__.__name__}", cfg.model.fig_path)

        ts_model.save_checkpoint(final_trajectories, f"{final_trajectories.shape[0]}_final_{ts_model.__class__.__name__}.pt")
        ts_model.save_checkpoint(best_trajectory, f"best_final_{ts_model.__class__.__name__}.pt")

        log.info("Final (min over trajectories) validation loss: " + str(min_val_loss))
        log.info("Final (via trajectory w/ min val loss) test loss: " + str(final_test_loss_for_best_trajectory))
        
        val_loss = min_val_loss
        test_loss = final_test_loss_for_best_trajectory

    return val_loss

@hydra.main(config_path="../configs/", config_name="config")
def main(cfg: DictConfig):
    return run_with_config(cfg)

if __name__ == "__main__":
    main()