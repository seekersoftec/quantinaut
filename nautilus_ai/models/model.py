# nautilus_ai/models/base.py
# Non-RL models should follow river's API
from pathlib import Path
import pickle
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
    
    @property
    def metric(self):
        pass

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


def load_model(config, log=None):
    """
    Loads a model and a scaler (if configured) from file.

    Args:
        config: Configuration object with model_path, scale_data, scaler_path attributes.
        log: Optional logger for info/warning/error messages.

    Returns:
        model, scaler (tuple)
    """
    model = None
    scaler = None
    # Load the model
    if getattr(config, 'model_path', None):
        path = Path(config.model_path)
        if not path.exists():
            if log:
                log.warning(f"Model file not found: {config.model_path}. Skipping model load.")
            return None, None
        ext = str(path).lower()
        if ext.endswith(".joblib"):
            try:
                import joblib
                model = joblib.load(path)
            except ImportError:
                if log:
                    log.error("joblib is not installed. Please install with `pip install joblib`.")
        elif ext.endswith((".h5", ".keras")):
            try:
                from keras.models import load_model
                model = load_model(path)
            except ImportError:
                if log:
                    log.error("Keras is not installed. Please install with `pip install keras`.")
        elif ext.endswith(".pkl"):
            with open(path, "rb") as f:
                model = pickle.load(f)
        else:
            if log:
                log.error(f"Unsupported model format: {ext}")
        if model is not None and log:
            log.info(f"Model loaded from {config.model_path}", color=getattr(log, 'GREEN', None))
    # Load the scaler if configured
    if getattr(config, 'scale_data', False):
        if not getattr(config, 'scaler_path', None):
            if log:
                log.warning("`scale_data` is True but `scaler_path` is not provided. Cannot load scaler.")
            return model, None
        scaler_path = Path(config.scaler_path)
        if not scaler_path.exists():
            if log:
                log.warning(f"Scaler file not found: {config.scaler_path}. Skipping scaler load.")
            return model, None
        try:
            import joblib
            scaler = joblib.load(scaler_path)
            if log:
                log.info(f"Scaler loaded from {config.scaler_path}", color=getattr(log, 'GREEN', None))
        except ImportError:
            if log:
                log.error("joblib is not installed. Please install with `pip install joblib`.")
        except Exception as e:
            if log:
                log.error(f"Failed to load scaler from {scaler_path}: {e}")
    return model, scaler



def save_model(model, scaler, config, log=None):
    """
    Saves the model and scaler (if they exist) to their respective paths.

    Args:
        model: The model object to save.
        scaler: The scaler object to save.
        config: Configuration object with model_path, scaler_path attributes.
        log: Optional logger for info/warning/error messages.
    """
    # Save the model
    if model and getattr(config, 'model_path', None):
        path = Path(config.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = str(path).lower()
        try:
            if ext.endswith(".joblib"):
                import joblib
                joblib.dump(model, path)
            elif ext.endswith((".h5", ".keras")):
                model.save(path)
            elif ext.endswith(".pkl"):
                with open(path, "wb") as f:
                    pickle.dump(model, f)
            else:
                if log:
                    log.error(f"Unsupported model format for saving: {ext}")
                return
            if log:
                log.info(f"Model saved to {config.model_path}", color=getattr(log, 'GREEN', None))
        except Exception as e:
            if log:
                log.error(f"Failed to save model to {path}: {e}")
    # Save the scaler
    if scaler and getattr(config, 'scaler_path', None):
        scaler_path = Path(config.scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
            joblib.dump(scaler, scaler_path)
            if log:
                log.info(f"Scaler saved to {config.scaler_path}", color=getattr(log, 'GREEN', None))
        except ImportError:
            if log:
                log.error("joblib is not installed. Please install with `pip install joblib`.")
        except Exception as e:
            if log:
                log.error(f"Failed to save scaler to {scaler_path}: {e}")
