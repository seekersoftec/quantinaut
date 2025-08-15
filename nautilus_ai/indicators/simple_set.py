# nautilus_ai/indicators/simple_set.py
from collections import deque
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import PriceType
from nautilus_ai.models import OnlineModel


class SimpleSet(Indicator):
    """
    A generic machine learning-based indicator.

    It collects a rolling window of prices and is designed to feed this
    data to an externally set machine learning model for online learning,
    inference, or feature engineering. The indicator itself does not perform
    any calculations other than storing the most recent prices.

    Parameters:
        period: int
            The rolling window period for the indicator (> 0).
        price_type: PriceType
            The type of price to use for the indicator (default is PriceType.LAST).
    """
    def __init__(self, period: int, price_type: PriceType = PriceType.LAST):
        PyCondition.positive_int(period, "period")

        super().__init__(params=[period])
        self.period = period
        self.price_type = price_type
        
        self._prices = deque(maxlen=period)
        self._count = 0  # Counter for the number of bars processed in the current batch
        self.model: OnlineModel = None
                
        self.value = 0.0  

    def handle_quote_tick(self, tick: QuoteTick):
        """
        Update the indicator with the given quote tick.

        It extracts the price from the tick and adds it to the internal
        rolling window.

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

        It extracts the price from the tick and adds it to the internal
        rolling window.

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

        It extracts the close price from the bar and adds it to the internal
        rolling window.

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
        self._prices.append(value)
        self.count += 1
        
        # Initialization logic
        if not self.initialized:
            self._set_has_inputs(True)
            if len(self._prices) >= self.period:
                self._set_initialized(True)

        self.value = self.model.predict_one()
        
        if self.model is not None:
            # If a model is set, it can process the rolling window
            self.model.learn_one()

    def set_model(self, model: OnlineModel):
        """
        Set the machine learning model for this indicator to use.

        This method is used to attach a pre-trained or new machine learning model
        to the indicator. The model can then process the data collected in the
        rolling window.

        Parameters
        ----------
        model : OnlineModel
            The machine learning model to be used.
        """
        PyCondition.not_none(model, "model")
        
        if not issubclass(model, OnlineModel):
            raise TypeError("Model must be a subclass of OnlineModel.")
        self.model = model
        
    def _reset(self):
        """
        Resets the state of the indicator, clearing its rolling window and model.
        """
        self._prices.clear()
        self._count = 0
        self.model = None
        self.value = 0.0
        