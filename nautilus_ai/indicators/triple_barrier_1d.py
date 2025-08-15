# nautilus_ai/indicators/triple_barrier_1d.py
from collections import deque
from typing import Any
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import PriceType
from river import compose, linear_model, preprocessing, metrics
from river import multiclass, optim


class TripleBarrier1D(Indicator):
    """
    Logic regarding labeling from Advances in Financial Machine Learning, chapter 3. 
    In particular the Triple Barrier Method and Meta-Labeling.

    It would be treated as a binary classification problem.

    Use Triple barrier labeling to generate targets for a simple logistic regression model.

    Parameters:
    period: int
        The rolling window period for the indicator (> 0).
    upper: float
        The upper barrier for the triple barrier labeling(in pct).
    lower: float
        The lower barrier for the triple barrier labeling(in pct).
        
    targets: 
    [-1, 0, 1] where -1 is a loss, 0 is no change, and 1 is a gain. 

    Libraries:
    - River (for online machine learning)

    Aim:
    - Predict the next barrier price might touch first.

    The processing of the barriers would be done on the fly, i.e when the model is about to learn from the data, convert the data to the targets.

    Since there is time in it, make sure that the data is done in batches, i.e after a certain number of bars, the model is trained on the data, then it predicts and waits for the next batch of data to be processed.

    Sources:
    - https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/labeling/labeling.py
    - https://github.com/nkonts/barrier-method
    
    """
    def __init__(self, period: int, upper: float = 0.03, lower: float = -0.03, price_type: PriceType = PriceType.LAST):
        PyCondition.positive_int(period, "period")
        PyCondition.positive(upper, "upper")
        PyCondition.negative(lower, "lower")
        
        if upper <= lower:
            raise ValueError("Upper barrier must be greater than lower barrier.")
        if upper >= 1.0 or lower <= -1.0:
            raise ValueError("Barriers must be in the range (-1, 1).")
        
        super().__init__(params=[period, upper, lower])
        self.period = period
        self.upper = upper
        self.lower = lower
        self.price_type = price_type
        
        self._model = None
        self._train_model = False  # Flag to indicate if the model is being trained
        self._prices = deque(maxlen=period)
        self._batch_count = 0  # Counter for the number of bars processed in the current batch
        self._metric = metrics.MacroF1()
        
        self.value = 0.0  

    def handle_quote_tick(self, tick: QuoteTick):
        """
        Update the indicator with the given quote tick.

        Parameters
        ----------
        tick : QuoteTick
            The update tick to handle.

        """
        PyCondition.not_none(tick, "tick")

        self.update_raw(tick.extract_price(self.price_type).as_double())

    def handle_trade_tick(self, tick: TradeTick):
        """
        Update the indicator with the given trade tick.

        Parameters
        ----------
        tick : TradeTick
            The update tick to handle.

        """
        PyCondition.not_none(tick, "tick")

        self.update_raw(tick.price.as_double())
        
    def handle_bar(self, bar: Bar):
        """
        Update the indicator with the given bar.

        Parameters
        ----------
        bar : Bar
            The update bar to handle.

        """
        PyCondition.not_none(bar, "bar")

        self.update_raw(bar.close.as_double())
    
    def update_raw(self, value: float):
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

        # Initialization logic
        if not self.initialized:
            self._set_has_inputs(True)
            if len(self._prices) >= self.period:
                self._set_initialized(True)
        pass
    
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
    
    def get_model(self):
        """
        Get the model for online learning.

        Returns
        -------
        object
            The online learning model.
        """
        return self._model
    
    def set_model(self, model: Any = None):
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
        
    def save_model(self, path: str):
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
    
    def load_model(self, path: str):
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
    
    def _reset(self):
        self.value = 0.0
        self.count = 0
        self._model = None
    
# TODO:
# Define a model interface for the online learning model.
# Implement the logic to handle the triple barrier method.
