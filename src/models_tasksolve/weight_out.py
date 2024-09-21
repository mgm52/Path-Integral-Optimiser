from src.models_tasksolve.base_ts_model import BaseTSModel
from src.utils.nn_creation import flat_parameters, set_params
from src.utils.time_utils import TimeTester
import torch
from torch import nn
from torch.nn import parameter

class WeightOut(BaseTSModel):
    def __init__(self, DATASIZE, GTSIZE):
        super().__init__(DATASIZE, GTSIZE)

    def forward(self, x, w):
        # x is of shape (samples_in_batch, DATASIZE)
        # w is of shape (trajectories_in_batch, WEIGHTSIZE)
        # return shape (trajectories_in_batch, samples_in_batch, GTSIZE)
        print(f"DATASIZE: {self.DATASIZE}, GTSIZE: {self.GTSIZE}")
        assert x.shape[0] == 1, f"Expected shape dim 0 to be (1), actual x.shape {x.shape}" # Only one sample in batch for carrillo
        assert len(w.shape) == 2
        return w.unsqueeze(1)

    def save_checkpoint(self, w, path):
        torch.save(w, path)

    def param_size(self):
        return self.GTSIZE
    
    def get_trainable_net(self):
        normal_sample = torch.distributions.Normal(0, 1).sample(torch.Size([self.GTSIZE])).to("cuda")
        return nn.parameter.Parameter(torch.tensor(normal_sample, requires_grad=True))
