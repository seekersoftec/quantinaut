# quantinaut/features/feature.py
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional


class Feature(metaclass=ABCMeta):
    """
    Abstract base class for feature generators.

    Subclasses should implement the `generate` method to produce features
    from a given context (e.g., a sample or data point).
    """

    @abstractmethod
    def generate(self, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract and generate features using the given context, if provided.

        Parameters
        ----------
            ctx (Optional[Dict[str, Any]]):
                Input context for a single sample (e.g., raw data, signals).

        Returns
        -------
            Dict[str, Any]:
                A dictionary of generated features for the sample.
        """
        pass
