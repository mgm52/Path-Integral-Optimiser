import matplotlib.pyplot as plt
import torch as th
import torch.distributions as D
from jamtorch.utils import as_numpy
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet
from typing import List, Tuple, Dict, Optional, Union, Callable, Any, TypeVar, Generic, Type, cast, Iterable, Sequence, Mapping, overload


class OptTest(BaseSet):
    # dataset: () -> x, GT
    # forward: x, w -> y
    # loss: y, GT -> loss
    def __init__(self, len_data,
                 dataset: Callable[[], Tuple[th.Tensor, th.Tensor]],
                 forward: Callable[[th.Tensor, th.Tensor], th.Tensor],
                 loss: Callable[[th.Tensor, th.Tensor], th.Tensor]):
        super().__init__(len_data)
        self.sigma = 0.1 # treat as a kind of learning rate for optimizer
        # used to specify input shape:
        self.data = th.tensor([0.0])  # pylint: disable= not-callable

    def V(self, w):
        x, GT = self.dataset()
        y = self.forward(x, w)
        return self.loss(y, GT)

    # returns negative log probability - i.e. the "discriminator" loss
    def get_gt_disc(self, x):
        C = 1.0
        p = th.exp(-self.V(x) / self.sigma) / C
        return -th.log(p) # TODO: consider cancelling out log and exp

    # just used for visualization purposes
    def sample(self, batch_size):
        return None # todo: let sampler be optionally specified in init

    def viz_pdf(self, fsave="ou-density.png"):
        x = th.linspace(-10, 10, 100).cuda()
        density = self.unnorm_pdf(x)
        x, pdf = as_numpy([x, density])
        fig, axs = plt.subplots(1, 1, figsize=(1 * 7, 1 * 7))
        axs.plot(x, pdf)
        fig.savefig(fsave)
        plt.close(fig)

    def __getitem__(self, idx):
        del idx
        return self.data[0]
