import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, Subset
from src.tasks.base_task import BaseTask
from sigfig import round
import torch.nn as nn
from multiprocessing import Lock
lock = Lock()

import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F


#lock = Lock()

class CSVToTensorDataset(Dataset):
    def __init__(self, data, transform=None, transform_label=None):
        self.data = data
        self.transform = transform
        self.transform_label = transform_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)
        
        if self.transform_label is not None:
            label = self.transform_label(label)

        return image, label

def load_mnist(path):
    # If data doesn't exist, download it
    print(f"About to load MNIST from {os.path.abspath(path)}, while in current directory {os.getcwd()}")
    
    train_path = os.path.join(path, 'mnist_train.csv')
    test_path = os.path.join(path, 'mnist_test.csv')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1).to("cuda")),
        ])
    
    transform_label = transforms.Compose([
        lambda x: F.one_hot(torch.tensor(x), num_classes=10),
        transforms.Lambda(lambda x: x.to("cuda"))
    ])

    train_data = pd.read_csv(train_path)
    train_data = CSVToTensorDataset(train_data, transform=transform, transform_label=transform_label)
    
    test_data = pd.read_csv(test_path)
    test_data = CSVToTensorDataset(test_data, transform=transform, transform_label=transform_label)

    return train_data, test_data

class MNISTTask(BaseTask):

    def __init__(self, train_dataset_len=1000, val_dataset_len=100, test_dataset_len=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 10

        abs_path = "/home/max/pis/datasets"
        relative_path = os.path.relpath(abs_path, os.getcwd())

        train_data, test_data = load_mnist(relative_path)

        self._train_dataset = Subset(train_data, range(train_dataset_len))
        
        self._val_dataset = Subset(train_data, range(train_dataset_len, train_dataset_len + val_dataset_len))

        self._test_dataset = Subset(test_data, range(test_dataset_len))


        #self._train_dataset = Subset(full_train_dataset, range(train_dataset_len))
        #self._val_dataset = Subset(full_train_dataset, range(train_dataset_len, train_dataset_len + val_dataset_len))
        #self._test_dataset = Subset(full_test_dataset, range(test_dataset_len))



    def training_dataset(self) -> Dataset:
        return self._train_dataset
    
    def validation_dataset(self) -> Dataset:
        return self._val_dataset

    def test_dataset(self) -> Dataset:
        return self._test_dataset

    import torch.nn as nn

    def loss(self, y, GT):
        # y is of shape (trajectories_in_batch, samples_in_batch, GTSIZE)
        # GT is of shape (samples_in_batch, GTSIZE)
        # return shape (trajectories_in_batch)

        loss_func = nn.NLLLoss(reduction='none')

        GT_argmax = torch.argmax(GT, dim=1)

        losses = loss_func(y.view(-1, y.size(-1)), GT_argmax.repeat(y.size(0)))

        return losses.view(y.size(0), -1).mean(dim=1)


    def datasize(self):
        return 28 * 28 # Size of an MNIST image
    
    def gtsize(self):
        return 10 # There are 10 classes in MNIST (0 to 9)

    def viz(self, ts_model, w, model_name, fig_path=""):
        num_samples = 16  # We're going to pick 16 samples
        fig, axs = plt.subplots(4, 4, figsize=(6, 6))  # Creating 4x4 grid for images

        samples = torch.stack([self._train_dataset[i][0] for i in range(num_samples)])  # Selecting the samples
        targets = torch.stack([self._train_dataset[i][1] for i in range(num_samples)])  # Selecting the targets

        predictions = ts_model.forward(samples, w.unsqueeze(0))  # Getting the model's predictions
        predictions = torch.argmax(predictions, dim=2)  # Getting the most likely class

        for i, ax in enumerate(axs.flatten()):
            ax.imshow(samples[i].cpu().view(28, 28), cmap='gray')  # Showing the image

            # Setting the title to be the predicted class and the actual class
            ax.set_title(f'Pred: {predictions[0][i].item()}, Actual: {torch.argmax(targets[i]).item()}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{fig_path}/{model_name}_mnist_examples.pdf")


