import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, CrossEntropyLoss
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam, lr_scheduler
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.datasets import ImageFolder, DatasetFolder

class BaseModel(Module):
    """
    LeNet-like Convolutional Neural Network.

    Args:
        input_shape (torch.Size): Shape of the input data.
        classes_ (int): Number of output classes.
    """

    def __init__(self, input_shape: torch.Size, classes_: int):
        super(BaseModel, self).__init__()

        channel_count = input_shape[0]
        self.conv1 = nn.Conv2d(in_channels=channel_count, out_channels=2, kernel_size=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=37632, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=classes_)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




# ---------------------------------------------------
#                      Main
# ---------------------------------------------------

