import matplotlib.pyplot as plt
import torch as th
import torch.distributions as D
from jamtorch.utils import as_numpy
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet
from typing import List, Tuple, Dict, Optional, Union, Callable, Any, TypeVar, Generic, Type, cast, Iterable, Sequence, Mapping, overload

from sklearn.datasets import make_moons

class OptGeneral(BaseSet):
    # dataset: () -> x, GT
    # forward: x, w -> y
    # loss: y, GT -> loss
    def __init__(self, len_data, V, sigma=0.1):
        # len_data is the total no. of trajectories to train per "epoch" (i.e. so we do len_data/trajectories full sampling passes per epoch)
        # should probably set len_data=trajectories
        super().__init__(len_data) 
        self.sigma = sigma # treat as a kind of learning rate for optimizer
        # used to specify input shape:
        self.data = th.tensor([0.0, 0.0])  # pylint: disable= not-callable

        # function (weights -> loss)
        # V should be normalised to have min 0 (otherwise we need unusual sigma values)
        # it should return shape (trajectories) - i.e. (x.shape[0])
        self.V = V
        self.recent_V = None

    # returns negative log probability - i.e. the "discriminator" loss
    #   input in shape (trajectories, *data_shape)
    #   output in shape (trajectories)
    def get_gt_disc(self, x):
        C = 1.0
        #p = th.exp(-self.V(x) / self.sigma) / C
        #return -th.log(p).flatten()

        self.recent_V = self.V(x)
        return ((self.recent_V / self.sigma) / C).flatten()

    # just used for visualization purposes
    #   output in shape (batch_size, *data_shape)
    def sample(self, batch_size):
        print(f"!!! WARNING - CALLED SAMPLE WITH BATCH SIZE {batch_size}")
        return th.zeros_like(self.data).repeat(batch_size, 1)

    def viz_pdf(self, fsave="opt-general-density.png"):
        if self.data.shape == (1,):
            x = th.linspace(-3, 5, 1000).cuda()
            density = self.unnorm_pdf(x)
            grad_nll = self.score(x)

            x, pdf, grad_nll = as_numpy([x, density, grad_nll])

            # plot the pdf and grad_nll on different y axes on the same plot
            fig, ax1 = plt.subplots()
            ax1.plot(x, pdf, color="tab:blue")
            ax1.set_xlabel("x")
            ax1.set_ylabel("pdf", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.plot(x, grad_nll, color="tab:red")
            ax2.set_ylabel("grad_nll", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            ax2.axhline(0, linestyle="--", color="tab:red")

            
            plt.title("OptGeneral target distribution with sigma={}".format(self.sigma))
            plt.savefig(fsave)
            plt.close(fig)

    def __getitem__(self, idx):
        del idx
        return self.data[0]