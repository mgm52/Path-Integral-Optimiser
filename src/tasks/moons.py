from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import torch
from src.tasks.base_task import BaseTask
from sigfig import round
from torch.utils.data import Dataset, TensorDataset

class MoonsTask(BaseTask):
    def __init__(self, dataset_len=1000):
        # dataset returns shape (samples_in_batch, DATASIZE), and (samples_in_batch, GTSIZE)
        x, GT = make_moons(n_samples=dataset_len, noise=0.1)
        x, GT = torch.from_numpy(x).cuda().float(), torch.from_numpy(GT).cuda().float()
        GT = GT.unsqueeze(1)
        self._dataset = TensorDataset(x, GT)
    
    def dataset(self) -> Dataset:
        return self._dataset
    
    def loss(self, y, GT):
        # y is of shape (trajectories_in_batch, samples_in_batch, GTSIZE)
        # GT is of shape (samples_in_batch, GTSIZE)
        # return shape (trajectories_in_batch)

        #if len(GT.shape) == 1:
            #print("We are assuming output data shape to be 1")
            #GT = GT.unsqueeze(1)
        #if len(y.shape) == 2:
            #print("We are assuming no. of trajectories to be 1")
            #y = y.unsqueeze(0)
        
        criterion = torch.nn.BCELoss(reduction='none')
        GT_expanded = GT.expand(y.shape[0], -1, -1)
        l = criterion(y, GT_expanded).mean(dim=(1,2))
        return l

    def datasize(self):
        return 2
    
    def gtsize(self):
        return 1
    
    def viz(self, x, y, y_name, GT, l):
        # x is of shape (samples_in_batch, DATASIZE)
        # y is of shape (samples_in_batch, GTSIZE)
        # GT is of shape (samples_in_batch, GTSIZE)

        # Visualize moon labelling attempt
        plt.scatter(x[:, 0].cpu().detach(), x[:, 1].cpu().detach(), c=GT.cpu().detach(), cmap="bwr", s=10)
        plt.scatter(x[:, 0].cpu().detach(), x[:, 1].cpu().detach(), c=y.cpu().detach(), cmap="bwr", s=8)
        plt.title(f"{y_name} labelling attempt (loss {round(float(l), sigfigs=3)})")
        plt.savefig(f"x_{y_name}_GT.png")
        plt.close()

        # Create a histogram of y[0]'s and GT's values
        plt.hist(y.cpu().detach().numpy(), bins=20, alpha=0.5, label="y")
        plt.hist(GT.cpu().detach().numpy(), bins=20, alpha=0.5, label="GT")
        plt.legend()
        plt.title(f"Histograms: {y_name} and GT.\n{y_name} loss: {round(float(l), sigfigs=3)}.")
        plt.savefig(f"{y_name}_GT_hist.png")
        plt.close()