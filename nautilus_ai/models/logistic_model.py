from collections import deque
from typing import Dict, Union
from pyparsing import Any
from river import compose, linear_model, preprocessing, metrics
from river import multiclass, optim
from nautilus_ai.models.model import OnlineModel


class LogisticRegressionModel(OnlineModel):
    """
   Logistic Regression Model using River

    l2
    Amount of L2 regularization used to push weights towards 0. For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.

    l1
    Amount of L1 regularization used to push weights towards 0. For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.

    """
    def __init__(self, l2, l1):
        
        super().__init__()
        
        self._model: linear_model.LogisticRegression = None
        self._batch_count = 0  # Counter for the number of bars processed in the current batch
        self._metric = metrics.MacroF1()
        
        self._build_model()

    @property
    def metric(self):
        return self._metric
    
    def learn_one(self, X: Dict[str, Any], y: Union[float, int]) -> None:
        """
        Update the indicator with the given raw value.

        Parameters
        ----------
        value : double
            The update value.

        """

        self._metric.update()
    
    def predict_one(self, X: Dict[str, Any]) -> Any:
        pass
            
    def _build_model(self):
        """
        Set the model for online learning.

        Parameters
        ----------
        model : Optional[Any]
            The online learning model to be used.
        """
        # River pipeline: scaler + logistic regression (multiclass OneVsRest)
        base = preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(0.05), l2=1e-4
        )
        self._model = multiclass.OneVsRestClassifier(base)
        # External model should be compatible with River's online learning paradigm.
