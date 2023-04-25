# pylint: disable=unused-import
import pytorch_lightning as pl
from jamtorch.utils import as_numpy
from pytorch_lightning import Callback

from src.callbacks.metric_cb import VizSampleDist
from src.viz.ou import dist_plot, traj_plot
from src.viz.wandb_fig import wandb_img

import torch

class OptMiniSample(VizSampleDist):
    def viz_sample(
        self, samples, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        samples = samples[:, : pl_module.data_ndim]
        # Make sure no nan values in samples
        if torch.isfinite(samples).all():
            fname = f"x-{trainer.global_step:04d}.png"
            # TODO: re-enable this for 1D
            #dist_plot(
            #    as_numpy(samples),
            #    pl_module.nll_target_fn,
            #    pl_module.nll_prior_fn,
            #    fname,
            #)
            #wandb_img("x", fname, trainer.global_step)
            #if pl_module.data_ndim < 5:
            #    pl_module.log("ksd", trainer.datamodule.dataset.ksd(samples))
        else:
            print(f"Failed to plot samples at step {trainer.global_step} due to nan values")