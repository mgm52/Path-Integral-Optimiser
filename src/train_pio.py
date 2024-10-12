from src.train_non_pio import get_logger
from typing import List
import hydra
from pytorch_lightning import Callback, LightningDataModule, Trainer, seed_everything
from src.callbacks.pis_opt_cb import PISOptCB
from src.datamodules.datasets.opt_general import OptGeneral
from src.logger.jam_wandb import JamWandb
from src.models_lightning.base_model import PIOModel

from src.models_tasksolve.base_ts_model import BaseTSModel
from src.tasks.base_task import BaseTask
from src.utils import lht_utils
from torch.utils.data import Dataset, DataLoader

from src.utils.nn_creation import flat_parameters

from src.utils.time_utils import TimeTester
import torch as th


log = lht_utils.get_logger(__name__)

class PISBasedOptimizer:

    def __init__(self, cfg, task: BaseTask, ts_model: BaseTSModel):
        self.task = task
        self.ts_model = ts_model
        self.cfg = cfg
        self.batch_laps = cfg.model.batch_laps
        self.x = None
        self.GT = None

        self.pis_dataset = OptGeneral(
                len_data=self.batch_laps * self.cfg.datamodule.dl.batch_size,
                V=self.task_weights_to_loss,
                sigma=self.cfg.datamodule.dataset.sigma)

        # TODO: neaten this (assigning of PIS output length)
        cfg.datamodule.ndim = ts_model.param_size()
        cfg.datamodule.shape = ts_model.param_size()

        # Houses our PIS model
        initial_target = flat_parameters(self.ts_model.get_trainable_net()).detach()
        self.model = PIOModel(self.cfg.model, get_sigma = self.pis_dataset.get_sigma, initial_target=initial_target)
        # TODO: move this into task...
        self.dataloader_iter = iter(DataLoader(self.task.training_dataset(), batch_size=self.cfg.model.batch_size, shuffle=True))

        # Init lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in self.cfg and self.cfg.callbacks:
            for _, cb_conf in self.cfg.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(
                        f"Instantiating callback <{cb_conf._target_}>"  # pylint: disable=protected-access
                    )
                    callbacks.append(hydra.utils.instantiate(cb_conf))
        
        # Important callback: moves on to new task minibatch
        callbacks.append(PISOptCB(self))

        # Instantiate a Trainer to use to train the PIS model
        self.trainer: Trainer = hydra.utils.instantiate(
            self.cfg.trainer, callbacks=callbacks, logger=get_logger(self.cfg), _convert_="partial"
        )

    # Train self.model to minimize V(w)=task_weights_to_loss(w)
    def start_train_loop(self):
        self.task_next_data()
        datamodule = BasicDataModule(
            dataset=self.pis_dataset,
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
        log.info(f"Done training; self.pis_dataset.recent_V.median()={self.pis_dataset.recent_V.median()}!")
        print("Done training PIS model.")
        return self.pis_dataset.recent_x

    
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

    def task_weights_to_validation_loss(self, w):
        val_dataset = self.task.validation_dataset()
        val_dataloader = DataLoader(val_dataset, batch_size=self.cfg.model.batch_size)
        
        l = th.zeros(w.shape[0], device=w.device)
        for x, GT in val_dataloader:
            y = self.ts_model.forward(x, w)
            l += self.task.loss(y, GT)
        l /= len(val_dataloader)

        return l

    def task_weights_to_test_loss(self, w):
        test_dataset = self.task.test_dataset()
        test_dataloader = DataLoader(test_dataset, batch_size=self.cfg.model.batch_size)
        
        l = th.zeros(w.shape[0], device=w.device)
        for x, GT in test_dataloader:
            y = self.ts_model.forward(x, w)
            l += self.task.loss(y, GT)
        l /= len(test_dataloader)

        return l

    # Load new V(w)
    def task_next_data(self):
        # Load from self.dataloader
        self.x, self.GT = next(self.dataloader_iter)

# TODO: replace with config / hydra access
class BasicDataModule(LightningDataModule):
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
        pass