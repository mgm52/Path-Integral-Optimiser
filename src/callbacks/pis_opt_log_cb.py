
from pytorch_lightning import Callback, Trainer
import torch


class PISOptLogCB(Callback):

    def on_after_backward(self, trainer, pl_module):
            maxgrad = max([torch.max(torch.abs(p.grad)).item() for p in pl_module.sde_model.parameters() if p.requires_grad])
            mediangrad = torch.median(torch.cat([torch.flatten(torch.abs(p.grad)) for p in pl_module.sde_model.parameters() if p.requires_grad]))
            pl_module.log_dict({
                "pis_grad_max_preopt": maxgrad,
                "pis_grad_median_preopt": mediangrad,
            })

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx):
        maxgrad = max([torch.max(torch.abs(p.grad)).item() for p in pl_module.sde_model.parameters() if p.requires_grad])
        mediangrad = torch.median(torch.cat([torch.flatten(torch.abs(p.grad)) for p in pl_module.sde_model.parameters() if p.requires_grad]))
        lr = pl_module.trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log_dict({
            "lr": lr,
            "V_median": pl_module.dataset.recent_V.median(),
            "sigma": pl_module.dataset.sigma,
            "pis_grad_max_postopt": maxgrad,
            "pis_grad_median_postopt": mediangrad,
        })