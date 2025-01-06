from lenet import LeNet
from model import BaseModel
from multi_model import MultiModel
from large_multi_model import LargeMultiModel

import torch
from torch.optim import Adam, Adadelta
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split, DataLoader

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
                 initial_learning_rate: float, classes_: int, model: str):
        self._train_loader: DataLoader = train_loader
        self._validation_loader: DataLoader = validation_loader
        self._initial_learning_rate: float = initial_learning_rate
        self._device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self._device.type)
        
        if model == "base":
            input_shape = next(iter(self._train_loader))[0][0].shape
            self._model: BaseModel = BaseModel(input_shape, classes_).to(self._device)
        elif model == "mm":
            input_shape = self._train_loader.dataset[0]['spectrogram'].shape
            numerical_features = self._train_loader.dataset[0]['mfcc'].shape[0]
            self._model: MultiModel = MultiModel(input_shape, classes_, numerical_features).to(self._device)
        elif model == "lmm":
            input_shape = self._train_loader.dataset[0]['spectrogram'].shape
            numerical_features = self._train_loader.dataset[0]['mfcc'].shape[0]
            self._model: LargeMultiModel = LargeMultiModel(input_shape, classes_, numerical_features).to(self._device)
        else:
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
        optimizer = Adadelta(self._model.parameters(), lr=self._initial_learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        history = self.__train_epochs__(loss_function, optimizer, scheduler, epochs)

        return self._model, history

    def __train_epochs__(self, loss_function, optimizer, scheduler, epochs):
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
            scheduler.step()
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
            (x, y) = (x.to(self._device), y.to(self._device).long())

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
    
class MFCCModelUtil(ModelUtil):
    """
    Utility class for initialization and training of the MFCC model.

    Args:
        train_loader (DataLoader): Data loader for training data.
        validation_loader (DataLoader): Data loader for validation data.
        initial_learning_rate (float): Initial learning rate for the optimizer.
        classes_ (int): Number of output classes.
    """

    def __init__(self, train_loader: DataLoader, validation_loader: DataLoader,
                 initial_learning_rate: float, classes_: int, model: str):
        super().__init__(train_loader, validation_loader, initial_learning_rate, classes_, model=model)

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

        for batch in loader:
            spec, mfcc, y = batch['spectrogram'], batch['mfcc'], batch['genre']
            (spec, mfcc, y) = (spec.to(self._device), mfcc.to(self._device), y.to(self._device))
            optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                out = self._model(spec, mfcc)
                _, pred = torch.max(out, 1)
                loss = loss_function(out, y)

                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * spec.size(0)
            total_correct += torch.sum(pred == y.data).item()

        return total_loss / len(loader.dataset), total_correct / len(loader.dataset)