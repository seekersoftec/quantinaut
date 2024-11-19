import torch
from torch import nn
from nautilus_ai.common import Logger

logger = Logger(__name__)


class PyTorchMLPModel(nn.Module):
    """
    A PyTorch-based multi-layer perceptron (MLP) model.

    This model serves as a flexible example for integrating PyTorch models. 
    It is not optimized for production use and is intended for demonstration purposes.

    Parameters:
    -----------
    input_dim : int
        The number of input features for the MLP.
    output_dim : int
        The number of output dimensions (e.g., classes for classification).
    hidden_dim : int, optional (default=256)
        The number of hidden units in each hidden layer.
    dropout_percent : float, optional (default=0.2)
        The dropout rate for regularization (values between 0 and 1).
    n_layer : int, optional (default=1)
        The number of hidden layers in the model.

    Returns:
    --------
    torch.Tensor
        The output tensor with shape `(batch_size, output_dim)`.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        hidden_dim = kwargs.get("hidden_dim", 256)
        dropout_percent = kwargs.get("dropout_percent", 0.2)
        n_layer = kwargs.get("n_layer", 1)

        # Define layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[Block(hidden_dim, dropout_percent) for _ in range(n_layer)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_percent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape `(batch_size, input_dim)`.

        Returns:
        --------
        torch.Tensor
            The output tensor with shape `(batch_size, output_dim)`.
        """
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class Block(nn.Module):
    """
    A modular building block for the MLP model.

    Combines layer normalization, a feedforward network, and dropout for regularization.

    Parameters:
    -----------
    hidden_dim : int
        The number of hidden units in the block.
    dropout_percent : float
        The dropout rate for regularization (values between 0 and 1).

    Returns:
    --------
    torch.Tensor
        The output tensor with shape `(batch_size, hidden_dim)`.
    """

    def __init__(self, hidden_dim: int, dropout_percent: float):
        super().__init__()
        self.feedforward = FeedForward(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_percent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape `(batch_size, hidden_dim)`.

        Returns:
        --------
        torch.Tensor
            Output tensor with shape `(batch_size, hidden_dim)`.
        """
        x = self.feedforward(self.layer_norm(x))
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    A simple feedforward network used within a Block.

    Parameters:
    -----------
    hidden_dim : int
        The number of hidden units in the layer.

    Returns:
    --------
    torch.Tensor
        The output tensor with shape `(batch_size, hidden_dim)`.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feedforward network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape `(batch_size, hidden_dim)`.

        Returns:
        --------
        torch.Tensor
            Output tensor with shape `(batch_size, hidden_dim)`.
        """
        return self.net(x)
