import torch
import torch.nn as nn

class MFCCModel(nn.Module):
    def __init__(self, input_shape, classes_, num_numerical_features):
        super(MFCCModel, self).__init__()
        print(input_shape)
        # CNN branch for spectrogram data
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
            nn.Flatten(),
        )
        
        # Calculate the flattened size after the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            flattened_size = self.cnn_branch(dummy_input).view(1, -1).size(1)
        
        # Linear layer for the CNN branch
        self.cnn_fc = nn.Linear(flattened_size, 100)
        
        # FCC branch for mfcc data
        self.fcc_branch = nn.Sequential(
            nn.Linear(num_numerical_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
        )
        
        # Combined branch
        self.combined_fcc = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, classes_)
        )

    def forward(self, spectrogram: torch.Tensor, mfcc: torch.Tensor):
        """
        Forward pass of the network.
        
        Args:
            spectrogram (torch.Tensor): Input tensor for the spectrogram data.
            mfcc (torch.Tensor): Input tensor for the MFCC data.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.cnn_branch(spectrogram)
        x = self.cnn_fc(x)
        y = self.fcc_branch(mfcc)
        z = torch.cat((x, y), dim=1)
        z = self.combined_fcc(z)
        return z