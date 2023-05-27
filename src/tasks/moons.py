from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import torch
from src.tasks.base_task import BaseTask
from sigfig import round
from torch.utils.data import Dataset, TensorDataset

class MoonsTask(BaseTask):
    def __init__(self, train_dataset_len=1000, val_dataset_len=100, test_dataset_len=100):
        datasets = [None, None, None]
        for i, dataset_len in enumerate([train_dataset_len, val_dataset_len, test_dataset_len]):
            x, GT = make_moons(n_samples=dataset_len, noise=0.1)
            x, GT = torch.from_numpy(x).cuda().float(), torch.from_numpy(GT).cuda().float()
            GT = GT.unsqueeze(1)
            datasets[i] = TensorDataset(x, GT)
        self._train_dataset, self._val_dataset, self._test_dataset = datasets

    
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

        criterion = torch.nn.BCELoss(reduction='none')
        GT_expanded = GT.expand(y.shape[0], -1, -1)
        l = criterion(y, GT_expanded).mean(dim=(1,2))
        return l

    def datasize(self):
        return 2
    
    def gtsize(self):
        return 1
    
    def viz(self, ts_model, w, model_name, fig_path=""):
        vis_samples = 100

        x, GT = self._val_dataset[:vis_samples]

        y = ts_model.forward(x, w.unsqueeze(0))[0]

        l = self.loss(y.unsqueeze(0), GT)

        # Visualize moon labelling attempt
        plt.scatter(x[:, 0].cpu().detach(), x[:, 1].cpu().detach(), c=y.cpu().detach(), cmap="coolwarm", s=32)
        plt.scatter(x[:, 0].cpu().detach(), x[:, 1].cpu().detach(), c=GT.cpu().detach(), cmap="coolwarm", s=1)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{model_name} labelling attempt (BCE {round(float(l), sigfigs=3)})")
        plt.savefig(f"{fig_path}/x_{model_name}_GT.png")
        plt.close()

        # Create a histogram of y[0]'s and GT's values
        plt.hist(y.cpu().detach().numpy(), bins=20, alpha=0.5, label="y")
        plt.hist(GT.cpu().detach().numpy(), bins=20, alpha=0.5, label="GT")
        plt.legend()
        plt.title(f"Histograms: {model_name} and GT.\n{model_name} BCE: {round(float(l), sigfigs=3)}.")
        plt.savefig(f"{fig_path}/{model_name}_GT_hist.png")
        plt.close()