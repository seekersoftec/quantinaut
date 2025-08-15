# nautilus_ai/indicators/simple_set.py
import pandas as pd
from datetime import datetime
from collections import deque
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import PriceType
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.vwap import VolumeWeightedAveragePrice
from nautilus_ai.models import OnlineModel
from nautilus_ai.features import Feature
from nautilus_ai.labels import Label


class SimpleSet(Indicator):
    """
    A generic classification-based indicator.

    It collects a rolling window of prices and is designed to feed this
    data to an externally set machine learning model for online learning and inference. 

    Parameters:
        period: int
            The rolling window period for the indicator (> 0).
        price_type: PriceType
            The type of price to use for the indicator (default is PriceType.LAST).
        process_batch: bool
            For rolling approaches that require one output per window. for the number of bars processed in the current batch
    """
    def __init__(self, period: int, price_type: PriceType = PriceType.LAST, process_batch: bool = False):
        PyCondition.positive_int(period, "period")

        super().__init__(params=[period])
        self.period = period
        self.price_type = price_type
        self.process_batch = process_batch # use to clear the prices after it has been used 
        
        self._prices = deque(maxlen=period)
        self._atr = AverageTrueRange(period)
        self._vwap = VolumeWeightedAveragePrice()
        
        self.features: Feature = None    
        self.label: Label = None    
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
        hlc3 = (bar.high.as_double() + bar.low.as_double() + bar.close.as_double()) / 3.0
        self.update_raw(bar.high.as_double(), bar.low.as_double(), 
                        bar.close.as_double(), hlc3, bar.volume.as_double(), 
                        pd.Timestamp(bar.ts_init, tz="UTC"))
    
    def update_raw(self, high: float, low: float, close: float, price: float, volume: float = 0.0, ts: datetime = None):
        """
        Update the indicator with the given raw value.

        Parameters
        ----------
        price : double
            The update value.
        """
        self._prices.append(price)
        self._atr.update_raw(high, low, close)
        self._vwap.update_raw(price, volume, ts)
        
        # Initialization logic
        if not self.initialized:
            self._set_has_inputs(True)
            if len(self._prices) < self.period:
                return
            if not self._atr.initialized or not self._vwap.initialized:
                return
            self._set_initialized(True)
        
        if self.model is not None:
            # Assemble the features from the rolling window
            features = self.features.generate({"atr": self._atr.value, "vwap": self._vwap.value, "prices": list(self._prices)})
            target = self.label.transform()
                
            # Make a prediction with the current features
            self.value = self.model.predict_one(features)

            # If a target is provided, train the model with the new data point
            if target is not None:
                self.model.learn_one(features, target)
        
        # For rolling approaches that require one output per fixed window.
        if self.process_batch:
            self._prices.clear() 

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
        self.model = None
        self.value = 0.0
        