from abc import ABC, abstractmethod
import pandas as pd
import torch


class PyTorchDataConvertor(ABC):
    """
    Abstract base class for converting pandas DataFrames into PyTorch tensors.
    
    Subclasses must implement methods for converting features (`convert_x`) 
    and labels (`convert_y`) into PyTorch tensors suitable for training or inference.
    """

    @abstractmethod
    def convert_x(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        """
        Converts a features DataFrame to a PyTorch tensor.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing feature data.
        device : str
            The device to use for the tensor (e.g., 'cpu', 'cuda').

        Returns:
        --------
        torch.Tensor
            The features as a PyTorch tensor.
        """
        pass

    @abstractmethod
    def convert_y(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        """
        Converts a labels DataFrame to a PyTorch tensor.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing label data.
        device : str
            The device to use for the tensor (e.g., 'cpu', 'cuda').

        Returns:
        --------
        torch.Tensor
            The labels as a PyTorch tensor.
        """
        pass


class DefaultPyTorchDataConvertor(PyTorchDataConvertor):
    """
    Default implementation of PyTorchDataConvertor.
    
    Converts features and labels DataFrames into tensors while retaining their original shapes. 
    Allows customization of label tensor type and optional squeezing for compatibility with loss functions.
    """

    def __init__(
        self,
        target_tensor_type: torch.dtype = torch.float32,
        squeeze_target_tensor: bool = False,
    ):
        """
        Initializes the DefaultPyTorchDataConvertor.

        Parameters:
        -----------
        target_tensor_type : torch.dtype, optional
            The data type of the label tensor. Use `torch.long` for classification 
            and `torch.float` or `torch.double` for regression. Default is `torch.float32`.
        squeeze_target_tensor : bool, optional
            Whether to squeeze the target tensor to reduce its dimensions (e.g., for 0D or 1D loss functions).
            Default is False.
        """
        self._target_tensor_type = target_tensor_type
        self._squeeze_target_tensor = squeeze_target_tensor

    def convert_x(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        """
        Converts a features DataFrame to a PyTorch tensor.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing feature data.
        device : str
            The device to use for the tensor (e.g., 'cpu', 'cuda').

        Returns:
        --------
        torch.Tensor
            The features as a PyTorch tensor.
        """
        numpy_arrays = df.values
        return torch.tensor(numpy_arrays, device=device, dtype=torch.float32)

    def convert_y(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        """
        Converts a labels DataFrame to a PyTorch tensor.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing label data.
        device : str
            The device to use for the tensor (e.g., 'cpu', 'cuda').

        Returns:
        --------
        torch.Tensor
            The labels as a PyTorch tensor, with optional squeezing applied if enabled.
        """
        numpy_arrays = df.values
        y = torch.tensor(numpy_arrays, device=device, dtype=self._target_tensor_type)
        return y.squeeze() if self._squeeze_target_tensor else y
