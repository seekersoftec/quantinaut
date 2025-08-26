# nautilus_ai/labels/label.py
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union


class Label(metaclass=ABCMeta):
    """
    Abstract base class for label transformers.

    Subclasses should implement methods to convert input features into labels
    for supervised learning tasks (classification or regression).
    """

    @abstractmethod
    def transform_one(self, X: Optional[Dict[str, Any]] = None) -> Union[int, float]:
        """
        Transform input features for a single sample into a label.

        Parameters
        ----------
            X (Optional[Dict[str, Any]]):
                Input features for one sample.

        Returns:
            Any:
                The label value (e.g., int for classification, float for regression).
                Returns 0 if no prediction is made.
        """
        pass
    
    @abstractmethod
    def transform_many(self, X: Optional[Dict[str, Any]] = None) -> List[Union[int, float]]:
        """
        Transform input features for multiple samples into labels.

        Parameters
        ----------
            X (Optional[Dict[str, Any]]):
                Input features for multiple samples (batched or iterable).

        Returns:
            Any:
                The predicted label values (e.g., list or array for batch prediction).
        """
        pass
