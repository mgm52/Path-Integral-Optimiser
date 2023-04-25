
from pytorch_lightning import Callback, Trainer
import torch


class PISOptCB(Callback):

    def __init__(self, pis_opt):
        self.pis_opt = pis_opt

    # Update sigma
    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx):
        sigma_decrease_interval_epochs = 12
        #if batch_idx==0 and trainer.current_epoch>0 and trainer.current_epoch % sigma_decrease_interval_epochs == 0:
            #pl_module.dataset.sigma = pl_module.dataset.sigma * 0.5
            #print(f"Decreasing sigma to {pl_module.dataset.sigma}")
        
        pl_module.dataset.sigma = pl_module.trainer.optimizers[0].param_groups[0]["lr"]
    
    # Important callback: move on to new task minibatch
    def on_epoch_end(self, trainer, pl_module):
        print("Done epoch; loading new loss function")
        self.pis_opt.task_next_data()