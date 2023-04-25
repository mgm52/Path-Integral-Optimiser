from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import pytorch_lightning as pl
from typing import Any, List
from sigfig import round
import hydra
from sklearn.datasets import make_moons
import torch
from jammy import hyd_instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from src.callbacks.non_meta_log_cb import NonMetaLogCB
from src.callbacks.pis_opt_cb import PISOptCB
from src.callbacks.pis_opt_log_cb import PISOptLogCB
from src.datamodules.datasets.opt_general import OptGeneral
from src.logger.jam_wandb import JamWandb
from src.models.base_model import BaseModel

from src.models.loss import loss_pis
from src.models.non_meta_model import NonMetaModel
from src.task_solving_models.base_ts_model import BaseTSModel
from src.task_solving_models.ff_nn import FeedForwardNN
from src.tasks.base_task import BaseTask
from src.tasks.moons import MoonsTask
from src.utils import lht_utils
from src.utils.sampling import generate_traj
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import numpy as np
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils.time_utils import TimeTester

# TODO: see if there's a better way to do this, using config & hydra
class MyDataModule(LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int = 0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def prepare_data(self):
        # Add any data preparation logic here
        pass

log = lht_utils.get_logger(__name__)

class PISBasedOptimizer:

    def __init__(self, cfg, task: BaseTask, ts_model: BaseTSModel):
        self.task = task
        self.ts_model = ts_model
        self.cfg = cfg
        self.sampling_repeats_per_data = cfg.model.sampling_repeats_per_data
        self.x = None
        self.GT = None

        # TODO: neaten this (assigning of PIS output length)
        cfg.datamodule.ndim = ts_model.param_size()
        cfg.datamodule.shape = ts_model.param_size()

        # Houses our PIS model
        self.model = BaseModel(self.cfg.model)
        # TODO: move this into task...?
        self.dataloader_iter = iter(DataLoader(self.task.dataset(), batch_size=self.cfg.model.batch_size, shuffle=True))

        # Init lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in self.cfg and self.cfg.callbacks:
            for _, cb_conf in self.cfg.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(
                        f"Instantiating callback <{cb_conf._target_}>"  # pylint: disable=protected-access
                    )
                    callbacks.append(hydra.utils.instantiate(cb_conf))
        
        callbacks.append(PISOptCB(self))
        callbacks.append(PISOptLogCB())

        # Instantiate a Trainer to use to train the PIS model
        self.trainer: Trainer = hydra.utils.instantiate(
            self.cfg.trainer, callbacks=callbacks, logger=get_logger(self.cfg), _convert_="partial"
        )

    # Train self.model to minimize V(w)=task_weights_to_loss(w)
    def start_train_loop(self):
        self.task_next_data()
        datamodule = MyDataModule(
            dataset=OptGeneral(
                len_data=self.sampling_repeats_per_data * self.cfg.datamodule.dl.batch_size,
                V=self.task_weights_to_loss,
                sigma=self.cfg.datamodule.dataset.sigma),
            batch_size=self.cfg.datamodule.dl.batch_size,
            num_workers=0)

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        JamWandb.g_cfg = self.cfg
        lht_utils.log_hyperparameters(
            config=self.cfg,
            model=self.model,
            datamodule=datamodule,
            trainer=self.trainer,
            callbacks=self.trainer.callbacks,
            logger=self.trainer.logger
        )

        # reseed before training, encounter once after instantiation, randomness disappear
        if self.cfg.get("seed"):
            seed_everything(self.cfg.seed, workers=True)

        # Train the model
        log.info("Starting training!")
        self.trainer.fit(model=self.model, datamodule=datamodule)
        print("Done training PIS model.")
    
    # Run V(w)
    def task_weights_to_loss(self, w):
        ttimer = TimeTester("V timer", disabled=True)
        ttimer.start("Forward")
        y = self.ts_model.forward(self.x, w)
        ttimer.end_prev()
        ttimer.start("Loss")
        l = self.task.loss(y, self.GT)
        ttimer.end_all()

        #l_median_index = torch.argsort(l)[l.shape[0]//2]
        #self.task.viz(self.x, y[l_median_index], "y_median", self.GT, l[l_median_index])
        return l

    # Load new V(w)
    def task_next_data(self):
        # Load from self.dataloader
        self.x, self.GT = next(self.dataloader_iter)

def get_logger(cfg):
    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(
                    f"Instantiating logger <{lg_conf._target_}>"  # pylint: disable=protected-access
                )
                logger.append(hydra.utils.instantiate(lg_conf))
    return logger

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../configs/", config_name="config")
    def main(cfg: DictConfig):
        print(cfg)

        task = MoonsTask(dataset_len=50000)
        ts_model = FeedForwardNN(task.datasize(), task.gtsize(), 10)
        OPTIM = cfg.model.optimizer

        if OPTIM == "sgd":
            model = NonMetaModel(ts_model, task, cfg.model)
            callbacks = [NonMetaLogCB()]

            trainer = pl.Trainer(max_epochs=99999999, progress_bar_refresh_rate=10, logger=get_logger(cfg), callbacks=callbacks)

            # Send some parameters from config to all lightning loggers
            log.info("Logging hyperparameters!")
            JamWandb.g_cfg = cfg
            lht_utils.log_hyperparameters(
                config=cfg,
                model=model,
                datamodule=None,
                trainer=trainer,
                callbacks=trainer.callbacks,
                logger=trainer.logger
            )

            # reseed before training, encounter once after instantiation, randomness disappear
            if cfg.get("seed"):
                seed_everything(cfg.seed, workers=True)

            # Train the model
            log.info("Starting training!")

            trainer.fit(model)
        elif OPTIM == "pis":
            optimizer = PISBasedOptimizer(
                cfg=cfg,
                task=task,
                ts_model=ts_model)
            optimizer.start_train_loop()
    main()