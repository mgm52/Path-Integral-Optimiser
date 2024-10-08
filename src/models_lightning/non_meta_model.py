import pdb
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from src.models_tasksolve.base_ts_model import BaseTSModel
from src.optimizers.pio_monte_carlo import PIOMonteCarlo
from src.tasks.base_task import BaseTask
from torch.utils.data import DataLoader, TensorDataset, Dataset

class NonMetaModel(pl.LightningModule):
    def __init__(self, ts_model: BaseTSModel, task: BaseTask, cfg: DictConfig, optimizer_name="sgd"):
        super().__init__()
        self.ts_model = ts_model
        self.task = task
        self.cfg = cfg
        self.optimizer_name = optimizer_name

        # PIS model has initial w found by its initial (uncontrolled?) distribution from prior.
        # Here we instead use Pytorch's default weight initialization.
        net = ts_model.get_trainable_net()
        if isinstance(net, torch.nn.parameter.Parameter):
            w = net
        else:
            w = torch.nn.parameter.Parameter(torch.cat([p.view(-1) for p in net.parameters()]))
        self.register_parameter('w', w)

        self.save_hyperparameters(logger=False)

    def forward(self, x):
        return self.ts_model.forward(x, self.w.unsqueeze(0))

    # in automatic mode: this just does batch -> loss (which we're in)
    # in manual mode: this does batch -> loss -> param update
    def training_step(self, batch, batch_idx):
        x, GT = batch
        y = self.forward(x)
        loss = self.task.loss(y, GT)
        self.log('train_loss', loss)
        self.log('param_norm', self.w.norm())
        return loss

    def get_params_for_opt(self):
        return [self.w]

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.get_params_for_opt(), lr=self.cfg.lr)
        elif self.optimizer_name == "adam":
            b1 = self.cfg.b1 if hasattr(self.cfg, "b1") else 0.9
            b2 = self.cfg.b2 if hasattr(self.cfg, "b2") else 0.999
            optimizer = torch.optim.Adam(self.get_params_for_opt(), lr=self.cfg.lr, betas=(b1, b2))
        elif self.optimizer_name == "adagrad":
            optimizer = torch.optim.Adagrad(self.get_params_for_opt(), lr=self.cfg.lr)
        elif self.optimizer_name == "pis-mc":
            optimizer = PIOMonteCarlo(
                self.get_params_for_opt(),
                sigma=self.cfg.sigma,
                mc_max_steps_total=self.cfg.mc_max_steps_total,
                mc_ts_per_mc_step=self.cfg.mc_ts_per_mc_step,
                max_grad_norm=self.cfg.gradient_clip_val)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.task.training_dataset(), batch_size=self.cfg.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.task.validation_dataset(), batch_size=self.cfg.batch_size)

    def test_dataloader(self):
        return DataLoader(self.task.test_dataset(), batch_size=self.cfg.batch_size)

    # TODO: reactivate once fixed
    def on_epoch_end2(self):
        x, GT = self.task.training_dataset()[0]
        y = self.forward(x)
        l = self.task.loss(y, GT)
        
        print(f"Step {self.current_epoch}")
        plt.hist(y[0, :, 0].cpu().detach().numpy(), bins=20, alpha=0.5, label="y")
        plt.hist(GT.cpu().detach().numpy(), bins=20, alpha=0.5, label="GT")
        plt.legend()
        plt.title(f"Histograms: best y and GT.\nBest y loss: {l.min()}.\nAvg y loss: {l.mean()}.")
        plt.savefig(f"ybest_GT_hist.pdf")
        plt.close()

