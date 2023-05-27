
from pytorch_lightning import Callback, Trainer
import torch


class PISOptLRBoostCB(Callback):

    def __init__(self, lr_factor, no_of_boosts):
        self.lr_factor = lr_factor
        self.no_of_boosts = no_of_boosts
        self.boost_count = 0

    def on_before_backward(self, trainer: Trainer, pl_module, loss):
        if self.boost_count < self.no_of_boosts:
            self.prev_lr = pl_module.trainer.optimizers[0].param_groups[0]["lr"]
            pl_module.trainer.optimizers[0].param_groups[0]["lr"] *= self.lr_factor
    
    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx):
        if self.boost_count < self.no_of_boosts:
            pl_module.trainer.optimizers[0].param_groups[0]["lr"] = self.prev_lr
            self.boost_count += 1
