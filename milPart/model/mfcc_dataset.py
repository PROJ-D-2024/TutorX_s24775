import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ColorJitter, RandomApply, GaussianBlur


class SpectrogramMFCCDataset(Dataset):
    def __init__(self, csv_file, genre_to_idx, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.genre_to_idx = genre_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spectrogram_path = self.data_frame.iloc[idx]['spectrogramPath']
        mfcc_path = self.data_frame.iloc[idx]['mfccPath']
        genre = self.data_frame.iloc[idx]['genre']
        
        spectrogram = Image.open(spectrogram_path).convert('L')
        mfcc = np.load(mfcc_path)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        mfcc = torch.tensor(mfcc, dtype=torch.float32) 
        
        genre = self.genre_to_idx[genre]
        genre = torch.tensor(genre, dtype=torch.long)

        sample = {'spectrogram': spectrogram, 'mfcc': mfcc, 'genre': genre}

        return sample

def prepare_mfcc_data(batch_size: int, train_split: float, train_csv: str, test_csv: str, model: str):
    """
    Get data loaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size for data loaders.
        train_split (float): Proportion of data to use for training.
        spectrogram_csv (str): Path to the CSV file with spectrogram paths.
        mfcc_csv (str): Path to the CSV file with MFCC paths.

    Returns:
        tuple: Tuple containing the classes and a list of data loaders for training, validation, and testing.
    """
    return get_mfcc_data_loaders(batch_size, train_split, train_csv=train_csv, test_csv=test_csv, model=model)

def get_mfcc_data_loaders(batch_size, train_split, train_csv, test_csv, model: str):
    """
    Prepare data loaders for spectrograms and MFCCs.

    Args:
        batch_size (int): Batch size for the data loaders.
        train_split (float): Fraction of the data to be used as the training set.
        spectrogram_csv (str): Path to the CSV file with spectrogram paths.
        mfcc_csv (str): Path to the CSV file with MFCC paths.

    Returns:
        tuple: Tuple containing the classes and a list of data loaders for training, validation, and testing.
    """
    data_augmentation_transforms = Compose([
        RandomHorizontalFlip(),
        Resize((64, 128)),
        ToTensor(),
    ])

    transform = Compose([
        Resize((128, 385)),
        ToTensor()
    ])

    # Create a mapping from genre names to numerical labels
    data_frame = pd.read_csv(train_csv)
    genres = data_frame['genre'].unique()
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    print(genres)

    train_dataset = SpectrogramMFCCDataset(train_csv, genre_to_idx, transform=transform)
    test_dataset = SpectrogramMFCCDataset(test_csv, genre_to_idx, transform=transform)

    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    classes_ = genres

    return classes_, [train_loader, validation_loader, test_loader]
