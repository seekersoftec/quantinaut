# nautilus_ai/features/feature.py
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional


class Feature(metaclass=ABCMeta):
    """
    Base class for feature models.

    This class defines the interface for models compatible with the 
    """

    @abstractmethod
    def generate(self, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

#   def _assemble_features(self) -> Dict[str, Any]:
#         """
#         Assemble features from the rolling window of prices into a dictionary.
#         """
#         features = {}
#         for i, price in enumerate(self._prices):
#             # Creates a dictionary like {'price_0': 100.1, 'price_1': 100.2, ...}
#             features[f'price_{i}'] = price
#         return features