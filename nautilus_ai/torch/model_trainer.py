from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from nautilus_ai.common import Logger
from nautilus_ai.torch.data_convertor import PyTorchDataConvertor
from nautilus_ai.torch.trainer_interface import PyTorchTrainerInterface
from nautilus_ai.torch.datasets import WindowDataset

logger = Logger(__name__)


class PyTorchModelTrainer(PyTorchTrainerInterface):
    """
    A trainer class for PyTorch models.

    This class manages the training loop, including data preparation, loss computation,
    optimizer updates, and optional evaluation.

    :param model: The PyTorch model to be trained.
    :param optimizer: The optimizer to use for training.
    :param criterion: The loss function to use for training.
    :param device: The device to use for training (e.g., 'cpu', 'cuda').
    :param data_convertor: A utility for converting pandas DataFrames to PyTorch tensors.
    :param model_meta_data: Metadata related to the model (optional).
    :param window_size: Window size for sequence-based datasets (default: 1).
    :param tb_logger: Logger for tracking training metrics (optional).
    :param kwargs: Additional parameters, including:
        - n_epochs: Number of training epochs (default: 10).
        - n_steps: Total number of optimizer steps. Overrides n_epochs if provided.
        - batch_size: Batch size for DataLoader (default: 64).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
        data_convertor: PyTorchDataConvertor,
        model_meta_data: Dict[str, Any] = None,
        window_size: int = 1,
        tb_logger: Any = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.data_convertor = data_convertor
        self.model_meta_data = model_meta_data or {}
        self.window_size = window_size
        self.tb_logger = tb_logger
        self.test_batch_counter = 0

        self.n_epochs: int = kwargs.get("n_epochs", 10)
        self.n_steps: int = kwargs.get("n_steps")
        self.batch_size: int = kwargs.get("batch_size", 64)

        if not self.n_steps and not self.n_epochs:
            raise ValueError("Either `n_steps` or `n_epochs` must be set.")

    def fit(self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]):
        """
        Train the model using the provided data.

        :param data_dictionary: Dictionary with training and testing data.
        :param splits: Data splits to use (e.g., ['train', 'test']).
        """
        self.model.train()
        data_loaders = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs = len(data_dictionary["train_features"])
        n_epochs = self.n_epochs or self.calc_n_epochs(n_obs)

        batch_counter = 0
        for _ in range(n_epochs):
            for _, (xb, yb) in enumerate(data_loaders["train"]):
                xb, yb = xb.to(self.device), yb.to(self.device)
                yb_pred = self.model(xb)
                loss = self.criterion(yb_pred, yb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if self.tb_logger:
                    self.tb_logger.log_scalar("train_loss", loss.item(), batch_counter)
                batch_counter += 1

            if "test" in splits:
                self.estimate_loss(data_loaders, "test")

    @torch.no_grad()
    def estimate_loss(self, data_loaders: Dict[str, DataLoader], split: str):
        """
        Estimate the model loss on a specific data split.

        :param data_loaders: Dictionary of DataLoaders for different splits.
        :param split: The data split to evaluate ('train' or 'test').
        """
        self.model.eval()
        for _, (xb, yb) in enumerate(data_loaders[split]):
            xb, yb = xb.to(self.device), yb.to(self.device)
            yb_pred = self.model(xb)
            loss = self.criterion(yb_pred, yb)

            if self.tb_logger:
                self.tb_logger.log_scalar(f"{split}_loss", loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1
        self.model.train()

    def create_data_loaders_dictionary(
        self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for training and evaluation splits.

        :param data_dictionary: Dictionary with input features and labels.
        :param splits: List of data splits to create loaders for.
        :returns: Dictionary of DataLoaders.
        """
        loaders = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            dataset = TensorDataset(x, y)
            loaders[split] = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
        return loaders

    def calc_n_epochs(self, n_obs: int) -> int:
        """
        Calculate the number of epochs based on the total observations and `n_steps`.

        :param n_obs: Total number of observations in the dataset.
        :returns: Number of epochs.
        """
        if not self.n_steps:
            raise ValueError("`n_steps` must be set to calculate epochs.")
        n_batches = n_obs // self.batch_size
        n_epochs = max(self.n_steps // n_batches, 1)
        if n_epochs <= 10:
            logger.warning(
                f"Low `n_epochs`: {n_epochs}. Consider increasing `n_steps`."
            )
        return n_epochs

    def save(self, path: Path):
        """
        Save the model, optimizer state, and metadata.

        :param path: Path to save the checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_meta_data": self.model_meta_data,
            },
            path,
        )

    def load(self, path: Path):
        """
        Load the model, optimizer state, and metadata.

        :param path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_meta_data = checkpoint.get("model_meta_data", {})
        return self


class PyTorchTransformerTrainer(PyTorchModelTrainer):
    """
    Trainer class for Transformer-based models, leveraging a sliding window dataset.
    """

    def create_data_loaders_dictionary(
        self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]
    ) -> Dict[str, DataLoader]:
        """
        Override to use WindowDataset for sequence-based training.

        :param data_dictionary: Dictionary with input features and labels.
        :param splits: List of data splits to create loaders for.
        :returns: Dictionary of DataLoaders.
        """
        loaders = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            dataset = WindowDataset(x, y, self.window_size)
            loaders[split] = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
            )
        return loaders
