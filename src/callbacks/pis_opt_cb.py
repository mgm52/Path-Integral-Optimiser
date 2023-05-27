
from pytorch_lightning import Callback, Trainer
import torch


class PISOptCB(Callback):

    def __init__(self, pis_opt):
        self.pis_opt = pis_opt

    # Important callback: move on to new task minibatch
    def on_epoch_end(self, trainer, pl_module):
        #print("Done epoch; loading new loss function")
        self.pis_opt.task_next_data()