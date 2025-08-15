from collections import deque
from river import compose, linear_model, preprocessing, metrics
from river import multiclass, optim
from nautilus_ai.models.model import OnlineModel


class LogisticRegressionModel(OnlineModel):
    """
   Logistic Regression Model using River
   
    Parameters:
        period: int
            The rolling window period for the indicator (> 0).

    """
    def __init__(self, period: int):
        
        super().__init__()
        self._model: linear_model.LogisticRegression = None
        self._train_model = False  # Flag to indicate if the model is being trained
        self._prices = deque(maxlen=period)
        self._batch_count = 0  # Counter for the number of bars processed in the current batch
        self._metric = metrics.MacroF1()
        
        self.value = 0.0  
    
    def update(self, value: float):
        """
        Update the indicator with the given raw value.

        Parameters
        ----------
        value : double
            The update value.

        """
        # Keep count of the prices stored, once it is >= period 
        # send it to be processed and clear the count.
        
        self._prices.append(value)
        
        # Check if this is the initial input
        if not self.has_inputs:
            self.value = value

        self.value = self.alpha * value + ((1.0 - self.alpha) * self.value)
        self.count += 1
    
    def get_label(self, prices: list) -> int:
        """
        Generate label based on the triple barrier method.

        Parameters
        ----------
        prices : list
            The input price history (float values).

        Returns
        -------
        int
            The generated label value, where -1 is a loss, 0 is no change, and 1 is a gain.
        """
        if None in prices:
            raise ValueError("Price data cannot contain None values.")
            
        if not prices or len(prices) < self.period:
            return 0
        
        for i, price in enumerate(prices):
            for j in range(len(prices) - 1, i, -1):
                diff = (price - prices[j]) / prices[j] # is this in percentages?
                if diff >= self.upper:
                    return 1
                elif diff <= self.lower:
                    return -1
        return 0

    def _build_model(self):
        """
        Set the model for online learning.

        Parameters
        ----------
        model : Optional[Any]
            The online learning model to be used.
        """
        # PyCondition.not_none(model, "model")
        
        if model is not None:
            self._model = model
        
        # River pipeline: scaler + logistic regression (multiclass OneVsRest)
        base = preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(0.05), l2=1e-4
        )
        self._model = multiclass.OneVsRestClassifier(base)
        # External model should be compatible with River's online learning paradigm.
        
    def reset(self):
        self.value = 0.0
        self.count = 0
        self._model = None
    