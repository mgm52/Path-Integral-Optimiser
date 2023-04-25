
from pytorch_lightning import Callback, Trainer
import torch


class NonMetaLogCB(Callback):

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx):
        pl_module.log_dict({
            "lr": pl_module.trainer.optimizers[0].param_groups[0]["lr"],
        })