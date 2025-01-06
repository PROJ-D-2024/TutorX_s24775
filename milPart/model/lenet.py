"""
This module contains a script for training a LeNet-like Convolutional Neural Network on a dataset.
It includes argument parsing, data preparation, model definition, and training utilities.

The script performs the following steps:
1. Parses command-line arguments to configure the training process.
2. Prepares the data loaders for training, validation, and testing datasets.
3. Defines a LeNet-like Convolutional Neural Network model.
4. Trains the model using the specified configuration.
5. Saves the trained model and training history to appropriate files.

Usage:
    To run the script, use the following command:

    python lenet.py -d <dataset_location> -b <batch_size> -t <train_split>
                                --initial_learning_rate <learning_rate> -e <epochs>

    Arguments:
        -d, --dataset_location      (str)   : Path to the dataset location (required).
        -b, --batch_size            (int)   : Batch size for data loaders (default: 64).
        -t, --train_split           (float) : Proportion of data to use for training (default: 0.7).
        --initial_learning_rate     (float) : Initial learning rate for the optimizer
                                              (default: 1e-3).
        -e, --epochs                (int)   : Number of training epochs (default: 10).

Example:
    python lenet.py -d /path/to/dataset -b 32 -t 0.8 --initial_learning_rate 0.001 -e 20


Requirements:
    - Python 3.6 or higher
    - PyTorch
    - torchvision
    - pandas
"""
# ---------------------------------------------------
#                     Imports
# ---------------------------------------------------
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import torch.cuda
from torch.nn import Module, Conv2d, Linear, CrossEntropyLoss
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam, lr_scheduler
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.datasets import ImageFolder, DatasetFolder


# ---------------------------------------------------
#                     Arguments
# ---------------------------------------------------

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_location', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-t', '--train_split', type=float, default=0.7)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    return parser.parse_args()


# ---------------------------------------------------
#                  Neural Network
# ---------------------------------------------------

class LeNet(Module):
    """
    LeNet-like Convolutional Neural Network.

    Args:
        input_shape (torch.Size): Shape of the input data.
        classes_ (int): Number of output classes.
    """

    def __init__(self, input_shape: torch.Size, classes_: int):
        super().__init__()

        channel_count = input_shape[0]
        self.cnn_branch = torch.nn.Sequential(
            Conv2d(in_channels=channel_count, out_channels=20, kernel_size=(5, 5)),
            Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        )
        
        dummy_input = torch.zeros(1, *input_shape)
        flattened_size = self.cnn_branch(dummy_input).view(1, -1).size(1)

        self.fcc_branch = torch.nn.Sequential(
            Linear(in_features=flattened_size, out_features=240),
            Linear(in_features=240, out_features=100),
            Linear(in_features=100, out_features=classes_)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.cnn_branch(x)
        x = self.fcc_branch(x.view(x.size(0), -1))
        return x


def prepare_data(batch_size: int, train_split: float, dataset_location: str):
    """
    Prepare the data loaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size for data loaders.
        train_split (float): Proportion of data to use for training.
        dataset_location (str): Path to the dataset location.

    Returns:
        tuple: Tuple containing the classes and a list of data loaders for training, validation,
        and testing.
    """
    transform = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = ImageFolder(os.path.join(dataset_location, "TRAIN"), transform=transform)
    test_data = ImageFolder(os.path.join(dataset_location, "TEST"), transform=transform)
    return get_data_loaders(batch_size, train_split, train_data, test_data)


def get_data_loaders(batch_size: int, train_split: float, train_data: DatasetFolder,
                     test_data: DatasetFolder = None):
    """
    Get data loaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size for data loaders.
        train_split (float): Proportion of data to use for training.
        train_data (Dataset): Training dataset.
        test_data (Dataset, optional): Testing dataset. Defaults to None.

    Returns:
        tuple: Tuple containing the classes and a list of data loaders for training, validation,
        and testing.
    """
    val_data = None
    classes_ = train_data.classes
    if train_split < 1.0:
        train_sample_count = int(len(train_data) * train_split)
        validation_sample_count = int(len(train_data) * (1 - train_split))
        train_sample_count += len(train_data) - (train_sample_count + validation_sample_count)
        (train_data, val_data) = random_split(train_data,
                                              [train_sample_count, validation_sample_count],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(val_data, batch_size=batch_size) if val_data else None
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return classes_, [train_loader, validation_loader, test_loader]


# ---------------------------------------------------
#                     Training
# ---------------------------------------------------

class ModelUtil:
    """
    Utility class for initialization and training of the model.

    Args:
        train_loader (DataLoader): Data loader for training data.
        validation_loader (DataLoader): Data loader for validation data.
        initial_learning_rate (float): Initial learning rate for the optimizer.
        classes_ (int): Number of output classes.
    """

    def __init__(self, train_loader: DataLoader, validation_loader: DataLoader,
                 initial_learning_rate: float, classes_: int):
        self._train_loader: DataLoader = train_loader
        self._validation_loader: DataLoader = validation_loader
        self._initial_learning_rate: float = initial_learning_rate
        self._device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self._device.type)
        input_shape = next(iter(self._train_loader))[0][0].shape
        self._model: LeNet = LeNet(input_shape, classes_).to(self._device)

    def train_model(self, epochs: int = 10):
        """
        Train the model and return the trained model and training history.

        Args:
            epochs (int): Number of training epochs.

        Returns:
            tuple: Tuple containing the trained model and training history.
        """
        if self._model is None or self._device is None:
            raise Exception("Model not yet initialized or device not yet set")

        loss_function = CrossEntropyLoss()
        optimizer = Adam(self._model.parameters(), lr=self._initial_learning_rate)
        history = self.__train_epochs__(loss_function, optimizer, epochs)

        return self._model, history

    def __train_epochs__(self, loss_function, optimizer, epochs):
        """
        Train the model for multiple epochs.

        Args:
            loss_function (Loss): Loss function.
            optimizer (Optimizer): Optimizer for model training.
            epochs (int): Number of training epochs.

        Returns:
            dict: Dictionary containing training history.
        """
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "validation_loss": [],
            "validation_acc": []
        }

        for epoch in range(0, epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            for train in [True, False]:
                if train:
                    self._model.train()
                else:
                    self._model.eval()

                run_loss, run_acc = self.__perform_epoch__(optimizer, loss_function, train)

                print(f'{"Train" if train else "Validation"} '
                      f'loss: {run_loss:.4f} acc: {run_acc:.4f}')

                if train:
                    history['train_loss'].append(run_loss)
                    history['train_acc'].append(run_acc)
                else:
                    history['validation_loss'].append(run_loss)
                    history['validation_acc'].append(run_acc)

            history['epoch'].append(epoch)
        return history

    def __perform_epoch__(self, optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module,
                          train: bool):
        """
        Perform a single epoch of training or validation.

        Args:
            optimizer (Optimizer): Optimizer for model training.
            loss_function (Loss): Loss function.
            train (bool): Whether to perform training or validation.

        Returns:
            tuple: Tuple containing the total loss and total accuracy for the epoch.
        """
        total_loss = 0.0
        total_correct = 0
        if train:
            loader = self._train_loader
        else:
            loader = self._validation_loader

        for (x, y) in loader:
            (x, y) = (x.to(self._device), y.to(self._device))
            optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                out = self._model(x)
                _, pred = torch.max(out, 1)
                loss = loss_function(out, y)

                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += torch.sum(pred == y.data).item()

        return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


# ---------------------------------------------------
#                      Main
# ---------------------------------------------------

if __name__ == "__main__":
    args_ = parse_arguments()
    classes, loaders = prepare_data(args_.batch_size, args_.train_split,
                                    args_.dataset_location)
    print("Data Prepared")
    model_trainer = ModelUtil(loaders[0], loaders[1], args_.initial_learning_rate, len(classes))
    model, history_ = model_trainer.train_model(args_.epochs)

    torch.save(model, 'lenet.pth')

    history_df = pd.DataFrame(history_)
    history_df.to_csv('training_history.csv', index=False)

    plt.plot(history_df['train_acc'], label="train_acc")
    plt.plot(history_df['validation_acc'], label="val_acc")
    plt.legend()
    plt.show()

    plt.plot(history_df['train_loss'], label="train_loss")
    plt.plot(history_df['validation_loss'], label="val_loss")
    plt.legend()
    plt.show()

