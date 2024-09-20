import math
import pytorch_lightning as pl
from typing import Any, List
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, Trainer, seed_everything
from src.callbacks.non_meta_log_cb import NonMetaLogCB
from src.callbacks.pis_opt_cb import PISOptCB
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
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.loggers import LightningLoggerBase
from src.utils.nn_creation import flat_parameters

from src.utils.time_utils import TimeTester
import torch as th

import pdb

# TODO: see if there's a better way to do this, using config & hydra
# TODO: at the very least, rename this..!
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
        self.model = BaseModel(self.cfg.model, get_sigma = self.pis_dataset.get_sigma, initial_target=initial_target)
        # TODO: move this into task...?
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
        datamodule = MyDataModule(
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

def start_mc_train_loop_no_lightning(cfg, task: BaseTask, ts_model: BaseTSModel):
    batch_size= cfg.model.batch_size
    sigma = cfg.datamodule.dataset.sigma
    m_monte_carlo = cfg.model.m_monte_carlo # only parameter unique to MC method
    num_trajectories = cfg.datamodule.dl.batch_size
    dt = cfg.model.dt

    num_t_samples = int(1/dt)
    # update dt to ensure we reach t_end=1
    dt = 1/num_t_samples

    train_dataloader = DataLoader(task.training_dataset(), batch_size=batch_size, shuffle=True)

    def get_P(w):
        x, GT = train_dataloader.sampler().next() # replace with actual function

        y = ts_model.forward(x, w)
        l = task.loss(y, GT)

        boltz_l = th.exp(-l / sigma)
        return boltz_l
    def get_N(w):
        n = w.shape[0]
        dist = th.distributions.MultivariateNormal(th.zeros(n), th.eye(n))  # N(0, I_n)
        return th.exp(dist.log_prob(w))  # Probability density function
    def get_f(w):
        P = get_P(w)
        N = get_N(w)
        return P/N
    def get_nabla_f(f, w):
        return th.autograd.grad(f, w, retain_graph=True)[0]
    def get_drift(x, t):
        f_values = []
        for i in range(m_monte_carlo):
            Z = th.randn(ts_model.param_size())
            f = get_f(x + math.sqrt(1 - t) * Z)
            f_values.append(f)
        nabla_f_values = [get_nabla_f(f, x) for f in f_values]
        sum_nabla_f = th.stack(nabla_f_values).sum(dim=0)
        sum_f = th.stack(f_values).sum()
        drift = sum_nabla_f / sum_f
        return drift

    
    final_trajectories = []
    for _ in range(num_trajectories):
        ts = th.linspace(0, 1, num_t_samples)
        x = th.zeros(ts_model.param_size())
        dt = ts[1] - ts[0]
        for i in range(num_t_samples):
            t = ts[i]
            drift = get_drift(x, t)
            noise = th.randn_like(x) * math.sqrt(dt)
            x = x + drift * dt + noise
        # Final state X_T!
        final_trajectories.append(x)
    
    # put final_trajectories into a tensor in shape (trajectories, *data_shape)
    final_trajectories = th.stack(final_trajectories)
    return final_trajectories

def start_sgd_train_loop(ts_model, task, cfg, optimizer_name):
    #pdb.set_trace()
    model = NonMetaModel(ts_model, task, cfg.model, optimizer_name).to("cuda")
    callbacks = [NonMetaLogCB()]

    #trainer = pl.Trainer(progress_bar_refresh_rate=10, logger=get_logger(cfg), callbacks=callbacks)
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=get_logger(cfg), _convert_="partial"
    )

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

    w = model.w.detach().cuda()

    # do validation
    val_loss = th.zeros(1, device="cuda")
    for x, GT in model.val_dataloader():
        y = ts_model.forward(x, w.unsqueeze(0))
        val_loss += task.loss(y, GT)
    val_loss /= len(model.val_dataloader())

    # do testing
    test_loss = th.zeros(1, device="cuda")
    for x, GT in model.test_dataloader():
        y = ts_model.forward(x, w.unsqueeze(0))
        test_loss += task.loss(y, GT)
    test_loss /= len(model.val_dataloader())

    return float(val_loss), float(test_loss), w.squeeze()

# TODO: move this into a class, like the PIS optimizer...
# Also reconsider whether I really want task and task-solving model bundled with optimizer class...
def mc_task_weights_to_validation_loss(cfg, w, ts_model: BaseTSModel, task: BaseTask):
    val_dataset = task.validation_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size)
    
    l = th.zeros(w.shape[0], device=w.device)
    for x, GT in val_dataloader:
        y = ts_model.forward(x, w)
        l += task.loss(y, GT)
    l /= len(val_dataloader)

    return l

def mc_task_weights_to_test_loss(cfg, w, ts_model: BaseTSModel, task: BaseTask):
    test_dataset = task.test_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.model.batch_size)
    
    l = th.zeros(w.shape[0], device=w.device)
    for x, GT in test_dataloader:
        y = ts_model.forward(x, w)
        l += task.loss(y, GT)
    l /= len(test_dataloader)

    return l

def run_with_config(cfg: DictConfig):
    #print(cfg) 
    #cfg.mode.hydra.run.dir = "logs/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}"

    task = hydra.utils.instantiate(
        cfg.task, _convert_="partial"
    )
    ts_model = hydra.utils.instantiate(
        cfg.task_solving_model, DATASIZE=task.datasize(), GTSIZE=task.gtsize(), _convert_="partial"
    )
    OPTIM = cfg.model.optimizer

    val_loss = None
    if OPTIM in ["sgd", "adam", "adagrad"]:
        val_loss, test_loss, w = start_sgd_train_loop(ts_model, task, cfg, OPTIM)

        ts_model.save_checkpoint(w, f"best_final_{ts_model.__class__.__name__}.pt")

        log.info("Final validation loss: " + str(val_loss))
        log.info("Final test loss: " + str(test_loss))

    elif OPTIM in ["pis", "pis-mc"]:
        if OPTIM == "pis-mc":
            final_trajectories = start_mc_train_loop_no_lightning(cfg, task, ts_model)
            final_val_task_losses = mc_task_weights_to_validation_loss(cfg, final_trajectories, ts_model, task)
            
            min_val_loss = float(final_val_task_losses.min())
            best_trajectory = final_trajectories[final_val_task_losses.argmin()]
            
            final_test_loss_for_best_trajectory = float(mc_task_weights_to_test_loss(cfg, best_trajectory.unsqueeze(0), ts_model, task))

        else:
            optimizer = PISBasedOptimizer(
                cfg=cfg,
                task=task,
                ts_model=ts_model)
            final_trajectories = optimizer.start_train_loop()
            final_val_task_losses = optimizer.task_weights_to_validation_loss(final_trajectories)
        
            min_val_loss = float(final_val_task_losses.min())
            best_trajectory = final_trajectories[final_val_task_losses.argmin()]

            final_test_loss_for_best_trajectory = float(optimizer.task_weights_to_test_loss(best_trajectory.unsqueeze(0)))

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