# nautilus_ai/indicators/triple_barrier.py
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import PriceType

# TODO: In-progress
class MLIndicator(Indicator):
    """
   Base ML Indicator class for online learning models.
   
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
    
    def _reset(self):
        self.value = 0.0
        self.count = 0
        self._model = None
