
from pytorch_lightning import Callback, Trainer
import torch


class PISOptSigmaCB(Callback):

    def __init__(self, sigma_factor=1.0):
        self.sigma_factor = sigma_factor

    # Update sigma
    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx):
        #sigma_decrease_interval_epochs = 12
        #if batch_idx==0 and trainer.current_epoch>0 and trainer.current_epoch % sigma_decrease_interval_epochs == 0:
            #pl_module.dataset.sigma = pl_module.dataset.sigma * 0.5
            #print(f"Decreasing sigma to {pl_module.dataset.sigma}")
        self.set_sigma(pl_module)
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.set_sigma(pl_module)

    def set_sigma(self, pl_module):
        pl_module.dataset.sigma = pl_module.trainer.optimizers[0].param_groups[0]["lr"] * self.sigma_factor