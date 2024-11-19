from abc import ABC, abstractmethod
import torch
from nautilus_ai.common import Logger
from nautilus_ai.interface import INautilusAIModel
from nautilus_ai.torch.PyTorchDataConvertor import PyTorchDataConvertor

logger = Logger(__name__)

class BasePyTorchModel(INautilusAIModel, ABC):
    """
    Abstract base class for PyTorch-based models in the Nautilus AI framework.
    
    Users must inherit from this class and implement the `fit`, `predict` methods, 
    and the `data_convertor` property to handle data preprocessing and conversion.
    """

    def __init__(self, **kwargs):
        """
        Initializes the BasePyTorchModel with common attributes for PyTorch models.

        Parameters:
        -----------
        **kwargs : dict
            Keyword arguments including `config`, which contains configuration details 
            for the model.
        """
        super().__init__(config=kwargs["config"])

        # Model-specific attributes
        self.data_drawer.model_type = "pytorch"  # Specify the model type as PyTorch.
        self.device = self._determine_device()

        # Data split configuration
        test_size = self.nautilus_ai_info.data_split_parameters.get("test_size", 0)
        self.splits = ["train", "test"] if test_size != 0 else ["train"]

        # Window size for convolutional models
        self.window_size = self.nautilus_ai_info.conv_width or 1

    @property
    @abstractmethod
    def data_convertor(self) -> PyTorchDataConvertor:
        """
        Abstract property that defines the data conversion logic.

        Returns:
        --------
        PyTorchDataConvertor
            A class responsible for converting `*_features` and `*_labels` pandas DataFrames
            into PyTorch tensors.

        Notes:
        ------
        This property must be implemented in subclasses.
        """
        raise NotImplementedError("The `data_convertor` property must be implemented in subclasses.")

    def _determine_device(self) -> str:
        """
        Determines the appropriate device for running PyTorch models.

        Returns:
        --------
        str
            The device to use: 'mps' (if available on macOS), 'cuda' (if available on GPUs),
            or 'cpu' as fallback.
        """
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
