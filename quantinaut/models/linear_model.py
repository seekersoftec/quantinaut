from pathlib import Path
from typing import Dict, Optional, Union
from pyparsing import Any
from river import compose, linear_model, preprocessing, metrics
from river import optim
from quantinaut.models.model import OnlineModel


class LinearRegressionModel(OnlineModel):
    """
    Online Linear Regression Model using River.

    This model uses River's online learning API for incremental training and prediction.
    Supports multiclass classification via OneVsRest, and includes L1/L2 regularization.

    Args:
        l2 (float):
            Amount of L2 regularization (default: 1e-4).
        l1 (float):
            Amount of L1 regularization (default: 0.0).
    """
    def __init__(self, l2: float = 1e-4, l1: float = 0.0):
        super().__init__()
        self._model: Optional[compose.Pipeline] = None
        self._batch_count = 0  # Counter for the number of bars processed in the current batch
        self._metric = metrics.MSE()
        self.l2 = l2
        self.l1 = l1
        self._build_model()

    @property
    def metric(self):
        """
        Returns the current evaluation metric (Macro F1).
        """
        return self._metric.get()

    def learn_one(self, X: Dict[str, Any], y: Union[float, int]) -> None:
        """
        Incrementally train the model on a single sample.

        Args:
            X (Dict[str, Any]):
                Input features for one sample.
            y (Union[float, int]):
                Target label for one sample.
        """
        self._model.learn_one(X, y)
        y_pred = self._model.predict_one(X)
        self._metric.update(y, y_pred)

    def predict_one(self, X: Dict[str, Any]) -> Any:
        """
        Predict the label for a single sample.

        Args:
            X (Dict[str, Any]):
                Input features for one sample.

        Returns:
            Any:
                Predicted label (int for classification).
        """
        return self._model.predict_one(X)

    def _build_model(self):
        """
        Build the River pipeline for online learning.

        Uses a standard scaler and linear regression wrapped in OneVsRest for multiclass support.
        """
        base = preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(0.1), l2=self.l2, l1=self.l1
        )
        self._model = base

    def save(self, path: Union[str, Path]):
        """
        Save the model to the specified path using pickle.

        Args:
            path (str): Path to save the model.
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Union[str, Path]):
        """
        Load the model from the specified path using pickle.

        Args:
            path (str): Path to load the model from.
        """
        import pickle
        with open(path, "rb") as f:
            self._model = pickle.load(f)
            
    def detail(self) -> Dict[str, Any]:
        """
        Output detailed information about the model, including parameters and metrics.

        Returns:
            dict: Dictionary containing model parameters and current metric value.
        """
        details = {
            "l2": self.l2,
            "l1": self.l1,
            "metric": self._metric.get(),
            "model_type": type(self._model).__name__,
            "model_repr": repr(self._model)
        }
        return details