
from pytorch_lightning import Callback, Trainer
import torch
from datetime import datetime

class PISOptLogCB(Callback):

    def on_after_backward(self, trainer, pl_module):
            maxgrad = max([torch.max(torch.abs(p.grad)).item() for p in pl_module.sde_model.parameters() if p.requires_grad]) # type: ignore
            mediangrad = torch.median(torch.cat([torch.flatten(torch.abs(p.grad)) for p in pl_module.sde_model.parameters() if p.requires_grad])) # type: ignore
            pl_module.log_dict({
                "pis_grad_max_preopt": maxgrad,
                "pis_grad_median_preopt": mediangrad,
            })

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx):
        maxgrad = max([torch.max(torch.abs(p.grad)).item() for p in pl_module.sde_model.parameters() if p.requires_grad])
        mediangrad = torch.median(torch.cat([torch.flatten(torch.abs(p.grad)) for p in pl_module.sde_model.parameters() if p.requires_grad]))
        lr = pl_module.trainer.optimizers[0].param_groups[0]["lr"]
        current_time = datetime.now().timestamp()
        if not (pl_module.dataset.recent_V is None):
            pl_module.log_dict({
                "lr": lr,
                "V_median": pl_module.dataset.recent_V.median(),
                "V_max": pl_module.dataset.recent_V.max(),
                "V_min": pl_module.dataset.recent_V.min(),
                "time": current_time,
                #"V_97-5": pl_module.dataset.recent_V.kthvalue(int(pl_module.dataset.recent_V.shape[0]*0.975), dim=0)[0],
                #"V_2-5": pl_module.dataset.recent_V.kthvalue(int(pl_module.dataset.recent_V.shape[0]*0.025), dim=0)[0],
                "V_std": pl_module.dataset.recent_V.std(),
                "sigma": pl_module.dataset.sigma,
                "pis_grad_max_postopt": maxgrad,
                "pis_grad_median_postopt": mediangrad,
            })

        #pl_module.dataset.sigma = pl_module.trainer.optimizers[0].param_groups[0]["lr"] * 10