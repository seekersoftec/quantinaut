import math
import torch
from torch import nn


"""
The architecture is based on the paper “Attention Is All You Need”.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Lukasz Kaiser, and Illia Polosukhin. 2017.
"""


class PyTorchTransformerModel(nn.Module):
    """
    Transformer-based model for time series forecasting using positional encoding.
    The architecture is inspired by the paper "Attention Is All You Need" by Vaswani et al., 2017.
    
    This model processes time series data and outputs predictions using a Transformer encoder.
    """

    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 7,
        hidden_dim: int = 1024,
        n_layer: int = 2,
        dropout_percent: float = 0.1,
        time_window: int = 10,
        nhead: int = 8,
    ):
        """
        Initializes the Transformer model components.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output predictions.
            hidden_dim (int): Number of hidden units in the fully connected layers.
            n_layer (int): Number of layers in the Transformer encoder.
            dropout_percent (float): Dropout rate for regularization.
            time_window (int): Number of time steps in the input sequence.
            nhead (int): Number of attention heads in the Transformer.
        """
        super().__init__()

        self.time_window = time_window

        # Ensure the input dimension to the Transformer is divisible by nhead
        self.dim_val = input_dim - (input_dim % nhead)

        # Input transformation network
        self.input_net = nn.Sequential(
            nn.Dropout(dropout_percent), 
            nn.Linear(input_dim, self.dim_val)
        )

        # Positional encoding for the time series data
        self.positional_encoding = PositionalEncoding(
            d_model=self.dim_val, 
            max_len=self.dim_val
        )

        # Transformer encoder block
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_val,
            nhead=nhead,
            dropout=dropout_percent,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=n_layer
        )

        # Fully connected layers for output decoding
        self.output_net = nn.Sequential(
            nn.Linear(self.dim_val * time_window, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, add_positional_encoding: bool = True) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, SeqLen, input_dim).
            mask (torch.Tensor, optional): Attention mask to apply (e.g., for padding). Defaults to None.
            add_positional_encoding (bool): Whether to add positional encoding to the input. Defaults to True.

        Returns:
            torch.Tensor: Output predictions of shape (Batch, output_dim).
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = x.reshape(-1, 1, self.time_window * x.shape[-1])
        x = self.output_net(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding module to inject sequence order into the model.
    Implements sine and cosine encoding as described in "Attention Is All You Need".
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encodings.

        Args:
            d_model (int): Dimensionality of the input features.
            max_len (int): Maximum expected sequence length.
        """
        super().__init__()

        # Create positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register as a non-trainable buffer
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, SeqLen, d_model).

        Returns:
            torch.Tensor: Input tensor with positional encoding added.
        """
        return x + self.pe[:, : x.size(1)]
