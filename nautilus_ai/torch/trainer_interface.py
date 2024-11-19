from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from torch import nn


class PyTorchTrainerInterface(ABC):
    """
    Abstract Base Class for PyTorch trainers, defining the core methods required for 
    training, saving, and loading PyTorch models.

    Any custom PyTorch trainer should inherit from this class and implement the abstract methods.
    """

    @abstractmethod
    def fit(self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]) -> None:
        """
        Train the PyTorch model using the provided data.

        :param data_dictionary: Dictionary containing training and testing data/labels.
            Keys should follow the convention of `<split>_features` and `<split>_labels`
            (e.g., "train_features", "train_labels").
        :param splits: List of data splits to use during training (e.g., ["train", "test"]).
            - "train" split is mandatory for training.
            - "test" split is optional and can be included for evaluation if available.

        The training process includes:
        - Forward pass to compute predictions.
        - Loss computation between predictions and actual values.
        - Backward pass for gradient computation.
        - Parameter updates using the specified optimizer.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the model's state and metadata to a file.

        :param path: Path to the file where the model state and metadata will be saved.

        The saved file should include:
        - The model's state dictionary (`state_dict`).
        - The optimizer's state dictionary (`state_dict`).
        - Any additional metadata about the model (e.g., class names, configuration).
        """
        pass

    def load(self, path: Path) -> nn.Module:
        """
        Load a model and optimizer state from a file.

        :param path: Path to the file containing the checkpoint.
        :returns: The PyTorch model loaded with the saved state.

        The checkpoint should include:
        - Model's state dictionary (`state_dict`).
        - Optimizer's state dictionary (`state_dict`).
        - Any additional metadata.
        """
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    @abstractmethod
    def load_from_checkpoint(self, checkpoint: dict) -> nn.Module:
        """
        Load a model and optimizer state from a checkpoint dictionary.

        :param checkpoint: Dictionary containing the model and optimizer state dictionaries, 
            and any additional metadata.
        :returns: The PyTorch model initialized with the checkpoint state.

        This method supports scenarios like continual learning, where models need to be 
        reloaded and trained further using saved checkpoints.
        """
        pass
