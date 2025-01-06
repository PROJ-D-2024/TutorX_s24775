import os
import argparse

import model
from util import ModelUtil, MFCCModelUtil
from mfcc_dataset import prepare_mfcc_data

import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.datasets import ImageFolder, DatasetFolder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-d', '--dataset_location', type=str, default="../../data")
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-t', '--train_split', type=float, default=0.9)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    return parser.parse_args()

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
        Resize((128, 385)),
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = ImageFolder(os.path.join(dataset_location, "spectrogramTRAIN"), transform=transform)
    test_data = ImageFolder(os.path.join(dataset_location, "spectrogramTEST"), transform=transform)
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

def plot_confusion_matrix(model, test_loader, classes, is_mfcc):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            if len(data) == 2:
                inputs, labels = data
                outputs = model(inputs)
            else:
                if is_mfcc:
                    spec, mfcc, labels = data['spectrogram'], data['mfcc'], data['genre']
                    outputs = model(spec, mfcc)
                else:
                    spec, labels = data['spectrogram'], data['mfcc'], data['genre']
                    outputs = model(spec)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == "__main__":
    args_ = parse_arguments()

    if args_.model in ["mm", "lmm"]:
        print("Preparing MFCC Data")
        classes, loaders = prepare_mfcc_data(
            args_.batch_size, args_.train_split,
            args_.dataset_location + "/spectrogramTRAIN/train_spectrograms.csv", 
            args_.dataset_location + "/spectrogramTEST/test_spectrograms.csv", args_.model
        )
        model_trainer = MFCCModelUtil(loaders[0], loaders[1], args_.initial_learning_rate, len(classes), args_.model)
    else:
        classes, loaders = prepare_data(
            args_.batch_size, args_.train_split,
            args_.dataset_location
        )
        model_trainer = ModelUtil(loaders[0], loaders[1], args_.initial_learning_rate, len(classes), args_.model)
    print("Data Prepared")

    
    trained_model, history_ = model_trainer.train_model(args_.epochs)

    torch.save(trained_model.state_dict(), f'../../data/models/{type(trained_model).__name__}.pth')

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

    plot_confusion_matrix(trained_model, loaders[2], classes, args_.model in ["mm", "lmm"])