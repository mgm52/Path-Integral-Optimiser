from typing import List
import hydra
from pytorch_lightning import seed_everything
from src.callbacks.non_meta_log_cb import NonMetaLogCB
from src.logger.jam_wandb import JamWandb

from src.models_lightning.non_meta_model import NonMetaModel
from src.models_tasksolve.base_ts_model import BaseTSModel
from src.tasks.base_task import BaseTask
from src.utils import lht_utils
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import LightningLoggerBase

import torch as th
from src.optimizers.pio_monte_carlo import PIOMonteCarlo


log = lht_utils.get_logger(__name__)

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

def start_sgd_train_loop(ts_model, task, cfg, optimizer_name):
    # HACKHACK to pass sigma to model config... Right now it's in datamodule for legacy PIS setup
    cfg.model.sigma = cfg.datamodule.dataset.sigma
    # HACKHACK to pass gradient_clip_val to model config... So PIS-MC can use it within optimizer
    cfg.model.gradient_clip_val = cfg.trainer.gradient_clip_val

    # TODO: Double check this is necessary...
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

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

    #JamWandb.finish()
    return float(val_loss), float(test_loss), w.squeeze()


def test_start_mc_train_loop_no_lightning(cfg, task: BaseTask, ts_model: BaseTSModel):
    print(f"Starting MC train loop with {cfg.model.m_monte_carlo} trajectories.")
    batch_size= cfg.model.batch_size
    sigma = cfg.datamodule.dataset.sigma
    m_monte_carlo = cfg.model.m_monte_carlo # only parameter unique to MC method
    num_trajectories = cfg.datamodule.dl.batch_size
    dt = cfg.model.dt

    print(f"Initialised with variables: batch_size={batch_size}, sigma={sigma}, m_monte_carlo={m_monte_carlo}, num_trajectories={num_trajectories}, dt={dt}")

    #g_coef = np.sqrt(0.2) # TODO: replace... right now this is copied from PISNN

    train_dataloader = DataLoader(task.training_dataset(), batch_size=batch_size, shuffle=True)
    
    nmm: NonMetaModel = NonMetaModel(ts_model, task, cfg.model, "replace-me-with-mc").to("cuda")
    train_dataloader = nmm.train_dataloader()
    train_iter = iter(train_dataloader)

    opt = PIOMonteCarlo(params=nmm.get_params_for_opt(), sigma=sigma, m_monte_carlo=m_monte_carlo, dt=dt, sigma_rescaling=True)

    final_trajectories = []
    for _ in range(num_trajectories):
        while not opt.completed_trajectory:
            #print(f"Not completed trajectory, running...")
            try:
                data, GT = next(train_iter)
            except StopIteration:
                # Reset iterator for a new epoch when the dataloader runs out of data
                train_iter = iter(train_dataloader)
                data, GT = next(train_iter)            
            data = data.to("cuda")
            GT = GT.to("cuda")
            def closure():
                y = nmm.forward(data)
                l = task.loss(y, GT)
                opt.zero_grad()
                l.backward() # TODO: this is unnecessary and wasteful, because optimizer already does autograd backwards... maybe i could optimize to use this somehow tho
                #print(f"Outer loss: {l.item()}")
                return l
            #print(f"Gonna step the opt...")
            opt.step(closure)
        final_trajectories.append(opt.extract_parameters_1d())

    # put final_trajectories into a tensor in shape (trajectories, *data_shape)
    final_trajectories = th.stack(final_trajectories)
    return final_trajectories

# Deprecated
def mc_task_weights_to_test_loss(cfg, w, ts_model: BaseTSModel, task: BaseTask):
    test_dataset = task.test_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.model.batch_size)
    
    l = th.zeros(w.shape[0], device=w.device)
    for x, GT in test_dataloader:
        y = ts_model.forward(x, w)
        l += task.loss(y, GT)
    l /= len(test_dataloader)

    return l

# Deprecated
def mc_task_weights_to_validation_loss(cfg, w, ts_model: BaseTSModel, task: BaseTask):
    val_dataset = task.validation_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size)
    
    l = th.zeros(w.shape[0], device=w.device)
    for x, GT in val_dataloader:
        y = ts_model.forward(x, w)
        l += task.loss(y, GT)
    l /= len(val_dataloader)

    return l