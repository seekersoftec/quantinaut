# nautilus_ai/models/base.py
# Non-RL models should follow river's API
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union


class OnlineModel(metaclass=ABCMeta):
    """
    Base class for online machine learning models.

    This class defines the interface for models compatible with the River
    online learning paradigm, using incremental learning and prediction methods.
    """

    @abstractmethod
    def learn_one(self, X: Dict[str, Any], y: Union[float, int]) -> None:
        """
        Learn from a single data point.

        This method updates the model's internal state with a single observation,
        aligning with River's core API.

        Parameters
        ----------
        X : Dict[str, Any]
            The input features for a single sample, models expect a dictionary
            of feature names and their values.
        y : Union[float, int]
            The target variable for the single sample.
        """
        pass

    @abstractmethod
    def predict_one(self, X: Dict[str, Any]) -> Any:
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

    def save(self, path: str):
        """
        Save the model to the specified path.

        Parameters
        ----------
        path : str
            The path to save the model.
        """
        # Implement saving logic here, e.g., using pickle or a specialized
        # serialization method provided by the specific River model.
        pass

    def load(self, path: str):
        """
        Load the model from the specified path.

        Parameters
        ----------
        path : str
            The path to load the model from.
        """
        # Implement loading logic here.
        pass

    def detail(self) -> Any:
        """
        Output detailed information about the model.
        """
        # This can be used to inspect the model's internal state,
        # parameters, or performance metrics.
        return 


class OfflineModel(metaclass=ABCMeta):
    """Base class for machine learning models"""

    @abstractmethod
    def learn(self, dataset: pd.DataFrame) -> None:
        """
        Fit the model with dataset
        """
        pass

    @abstractmethod
    def predict(self, dataset: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the model
        """
        pass

    def save(self, path: str):
        """
        Save the model to the specified path.

        Parameters
        ----------
        path : str
            The path to save the model.
        """
        # Implement saving logic here
        # Can be called in the strategy to save the model after training with the backtest engine.
        pass
    
    def load(self, path: str):
        """
        Load the model from the specified path.

        Parameters
        ----------
        path : str
            The path to load the model from.
        """
        # Implement loading logic here
        # Can be called in the strategy to load the model before training with the backtest engine.
        pass
    
    def detail(self) -> Any:
        """
        Output detailed information about the model
        """
        return  
    
    
class RLModel(metaclass=ABCMeta):
    """Base class for reinforcement learning models"""

    @abstractmethod
    def learn(self, state: Any, action: Any, reward: float, next_state: Any) -> None:
        """
        Update the model based on the state, action, reward, and next state.
        """
        pass

    @abstractmethod
    def predict(self, state: Any) -> Any:
        """
        Predict the best action to take in the given state.
        """
        pass

    def save(self, path: str):
        """
        Save the model to the specified path.
        """
        pass
    
    def load(self, path: str):
        """
        Load the model from the specified path.
        """
        pass

