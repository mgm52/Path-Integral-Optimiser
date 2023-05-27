from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import torch
from src.tasks.base_task import BaseTask
from sigfig import round
from torch.utils.data import Dataset, TensorDataset

class CarrilloTask(BaseTask):
    def __init__(self, d=2, B=0, C=0):
        self.d = d
        self.B = B
        self.C = C
        # datasets of zeros
        self._train_dataset, self._val_dataset, self._test_dataset = TensorDataset(torch.zeros(1,1), torch.zeros(1,1)), TensorDataset(torch.zeros(1,1), torch.zeros(1,1)), TensorDataset(torch.zeros(1,1), torch.zeros(1,1))
    
    def training_dataset(self) -> Dataset:
        return self._train_dataset
    
    def validation_dataset(self) -> Dataset:
        return self._val_dataset

    def test_dataset(self) -> Dataset:
        return self._test_dataset

    def loss(self, y, GT):
        # y is of shape (trajectories_in_batch, samples_in_batch, GTSIZE)
        # GT is of shape (samples_in_batch, GTSIZE)
        # return shape (trajectories_in_batch)

        # For the carillo function, GT is meaningless; assume we are using weight_out
        assert y.shape[-1] == self.d

        #Carrillo(y)=\frac{1}{d} \sum_{i=1}^d\left[\left(x_i-B\right)^2-10 \cos \left(2 \pi\left(x_i-B\right)\right)+10\right]+C:
        carrillo_y = torch.zeros(y.shape[0], y.shape[1]).cuda()
        for j in range(y.shape[1]):
            for i in range(y.shape[2]):
                carrillo_y[:, j] += ((y[:,:,i] - self.B)**2 - 10*torch.cos(2*np.pi*(y[:,:,i] - self.B)) + 10).squeeze()
            carrillo_y[:, j] = carrillo_y[:, j] / self.d + self.C
        
        # carrillo_y is of shape (trajectories_in_batch, samples_in_batch)
        # reduce to shape (trajectories_in_batch) by taking mean
        carrillo_y = torch.mean(carrillo_y, dim=1)

        return carrillo_y

    def datasize(self):
        return 1
    
    def gtsize(self):
        return self.d
    
    def viz(self, ts_model, w, model_name, fig_path=""):
        pass