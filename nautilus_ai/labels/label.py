# nautilus_ai/labels/label.py
from abc import ABCMeta, abstractmethod
from typing import Any, Dict


class Label(metaclass=ABCMeta):
    """
    Base class for label models.

    This class defines the interface for models compatible with the 
    """

    @abstractmethod
    def transform(self, X: Dict[str, Any]) -> Any:
        """
        Make a prediction for a single data point.

        Parameters
        ----------
        X : Dict[str, Any]
            The input features for a single sample.

        Returns
        -------
        Any
            The predicted value (e.g., float for regression, int for classification).
        """
        pass
