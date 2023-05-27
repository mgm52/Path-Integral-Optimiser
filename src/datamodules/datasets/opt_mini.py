import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributions as D
from jamtorch.utils import as_numpy
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet
from typing import List, Tuple, Dict, Optional, Union, Callable, Any, TypeVar, Generic, Type, cast, Iterable, Sequence, Mapping, overload

from sklearn.datasets import make_moons

def dataset():
    x, GT = make_moons(n_samples=100, noise=0.1)
    return th.from_numpy(x).cuda().float(), th.from_numpy(GT).cuda().float()

def forward(x, w):
    # x is of shape (samples_in_batch, DATASIZE)
    # w is of shape (trajectories_in_batch, WEIGHTSIZE)
    # we should return shape (trajectories, samples, GTSIZE)
    y = th.matmul(x, w.t()).transpose(1, 0)
    return y

def loss(y, GT):
    loss = th.mean((y - GT) ** 2, dim=1)
    return loss

class OptMini(BaseSet):
    # dataset: () -> x, GT
    # forward: x, w -> y
    # loss: y, GT -> loss
    def __init__(self, len_data, sigma=0.1):
        super().__init__(len_data)
        self.sigma = sigma # treat as a kind of learning rate for optimizer
        # used to specify input shape:
        self.data = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # pylint: disable= not-callable

    # V should be normalised to have min 0 (otherwise we need unusual sigma values)
    # it should be of shape (trajectories) - i.e. (x.shape[0])
    def V(self, x):
        #data, GT = dataset()
        #y = forward(data, x)
        #return loss(y, GT)
        
        return th.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)
        #return th.pow(x-0.4,2) * (0.2 + th.pow(x-2,2))
        #return 3.308 + (x + th.sin(x)) * th.pow(x - 2, 3)
        #return th.pow(x-3, 2)

    # returns negative log probability - i.e. the "discriminator" loss
    #   input in shape (batch_size, *data_shape)
    #   output in shape (batch_size)
    def get_gt_disc(self, x):
        # C shouldnt matter if we're using schrodinger-follmer
        C = 1.0
        #p = th.exp(-self.V(x) / self.sigma) / C
        #return -th.log(p).flatten()

        self.recent_V = self.V(x)
        return (self.recent_V / self.sigma).flatten() - np.log(C)

    # just used for visualization purposes
    #   output in shape (batch_size, *data_shape)
    def sample(self, batch_size):
        print(f"!!! WARNING - CALLED SAMPLE WITH BATCH SIZE {batch_size}")
        return th.zeros_like(self.data).repeat(batch_size, 1)

    def viz_pdf(self, fsave="opt-mini-density.png"):
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

            
            plt.title("OptMini target distribution with sigma={}".format(self.sigma))
            plt.savefig(fsave)
            plt.close(fig)

    def __getitem__(self, idx):
        del idx
        return self.data[0]