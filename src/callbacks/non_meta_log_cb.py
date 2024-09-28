
from datetime import datetime
from pytorch_lightning import Callback, Trainer
import torch


class NonMetaLogCB(Callback):

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx):
        current_time = datetime.now().timestamp()
        pl_module.log_dict({
            "lr": pl_module.trainer.optimizers[0].param_groups[0]["lr"] if "lr" in pl_module.trainer.optimizers[0].param_groups[0] else 1,
            "time": current_time,
        })