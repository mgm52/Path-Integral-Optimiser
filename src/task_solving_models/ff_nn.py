from src.task_solving_models.base_ts_model import BaseTSModel
from src.utils.nn_creation import flat_parameters, set_params
from src.utils.time_utils import TimeTester
import torch
from torch import nn

class FeedForwardNN(BaseTSModel):
    def __init__(self, DATASIZE, GTSIZE, HIDDENSIZE):
        super().__init__(DATASIZE, GTSIZE)
        self.HIDDENSIZE = HIDDENSIZE

    def forward(self, x, w):
        # x is of shape (samples_in_batch, DATASIZE)
        # w is of shape (trajectories_in_batch, WEIGHTSIZE)
        # return shape (trajectories_in_batch, samples_in_batch, GTSIZE)
        ttimer = TimeTester("Forward timer", disabled=True)

        trajectories_in_batch = w.shape[0]
        samples_in_batch = x.shape[0]

        # Instantiate trajectories_in_batch number of neural networks
        ttimer.start("Creating nets")
        nets = [
            self.get_trainable_net()
            for _ in range(trajectories_in_batch)
            ]
        ttimer.end_prev()
        #print("Created net of size ", sum(p.numel() for p in nets[0].parameters() if p.requires_grad))

        # Set the weights of each network
        ttimer.start("Setting weights")
        for i in range(trajectories_in_batch):
            set_params(nets[i], w[i])
            #visualize_weights(nets[i], f"traj_{i}")
        ttimer.end_prev()

        # TODO: consider moving this into end_of_batch callback somehow? maybe move into its own function, then insert that to PISOptimiser when its created
        #avgw = torch.mean(w, dim=0)
        #avgnet = get_net(DATASIZE, HIDDENSIZE, GTSIZE)
        #set_weights(avgnet, avgw)
        #visualize_weights(avgnet, f"avgtraj")

        # Feed forward each item in x for each network
        ttimer.start("Forwarding through nets")
        y = torch.stack([net(x) for net in nets])
        ttimer.end_prev()

        #w = torch.stack([flat_parameters(n) for n in nets])
        #w.retain_grad()

        ttimer.end_all()
        return y

    def param_size(self):
        return self.DATASIZE*self.HIDDENSIZE + self.HIDDENSIZE + self.HIDDENSIZE*self.GTSIZE + self.GTSIZE
    
    def get_trainable_net(self):
        return nn.Sequential(nn.Linear(self.DATASIZE, self.HIDDENSIZE), nn.ReLU(), nn.Linear(self.HIDDENSIZE, self.GTSIZE), nn.Sigmoid()).to("cuda")