# nautilus_ai/indicators/triple_barrier.py
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import PriceType


class TripleBarrier(Indicator):
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
        
        self.value = 0.0  # <-- stateful value
        self.count = 0  # <-- stateful value

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
        pass
    
    def get_barrier_labels(self, data):
        """
        Generate labels based on the triple barrier method.

        Parameters
        ----------
        data : list
            The input data to generate labels from.

        Returns
        -------
        list
            The generated labels.
        """
        # Implement the logic to generate labels based on the triple barrier method.
        # This is a placeholder for the actual implementation.
        # return [0] * len(data)
        # if the current price - the price at the time of the barrier is greater than the upper barrier, return 1
        # if the current price - the price at the time of the barrier is less than the lower barrier, return -1
        # else return 0
        pass
    
    def get_model(self):
        """
        Get the model for online learning.

        Returns
        -------
        object
            The online learning model.
        """
        return self._model
    
    def set_model(self, model):
        """
        Set the model for online learning.

        Parameters
        ----------
        model : object
            The online learning model to be used.
        """
        PyCondition.not_none(model, "model")
        self._model = model
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
