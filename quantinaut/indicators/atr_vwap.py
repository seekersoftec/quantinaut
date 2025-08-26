# nautilus_ai/indicators/simple_set.py
import pandas as pd
from datetime import datetime
from collections import deque
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import PriceType
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.vwap import VolumeWeightedAveragePrice
from quantinaut.models import OnlineModel
from quantinaut.features import Feature
from quantinaut.labels import Label


class AverageTrueRangeWithVwap(Indicator):
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
    def __init__(self, period: int, price_type: PriceType = PriceType.LAST, batch_bars: bool = False):
        PyCondition.positive_int(period, "period")

        super().__init__(params=[period])
        self.period = period
        self.price_type = price_type
        self.batch_bars = batch_bars # use to clear the prices after it has been used 
        
        self._count = 0
        self._prices = deque(maxlen=period)
        self._atr = AverageTrueRange(period)
        self._vwap = VolumeWeightedAveragePrice()
        
        self.features: Feature = None    
        self.label: Label = None    
        self.model: OnlineModel = None 
        self.metric = 0.0 
        
        self.atr = 0.0
        self.vwap = 0.0
        self.value = 0.0 
        
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
        self._count += 1
        
        # Initialization logic
        if not self.initialized:
            self._set_has_inputs(True)
            if len(self._prices) < self.period:
                return
            if not self._atr.initialized or not self._vwap.initialized:
                return
            self._set_initialized(True)
        
        self.atr = self._atr.value
        self.vwap = self._vwap.value
        if self.model is not None and self._count >= self.period: # 
            # Assemble the features from the rolling window
            features = self.features.generate({"atr": self._atr.value, "vwap": self._vwap.value, "prices": list(self._prices)})
            target = self.label.transform_one({"prices": list(self._prices)})
                
            # Make a prediction with the current features
            self.value = self.model.predict_one(features)

            # If a target is provided, train the model with the new data point
            if target is not None:
                self.model.learn_one(features, target)

            self.metric = self.model.metric
            
        # For rolling approaches that require one output per fixed window.
        if self.batch_bars:
            self._count = 0

    def set_model(self, features: Feature, label: Label, model: OnlineModel):
        """
        Attach feature, label, and online model objects to the indicator for learning and inference.

        This method sets the feature extractor, label transformer, and online learning model
        that will be used to process the rolling window of data. All three must be subclasses
        of their respective base classes.

        Args:
            features (Feature):
                The feature extractor (must be a subclass of Feature).
            label (Label):
                The label transformer (must be a subclass of Label).
            model (OnlineModel):
                The online machine learning model (must be a subclass of OnlineModel).
        """
        PyCondition.not_none(features, "features")
        PyCondition.not_none(label, "label")
        PyCondition.not_none(model, "model")
        
        if not isinstance(features, Feature) or not issubclass(features.__class__, Feature):
            raise TypeError("Feature Model must be a subclass of Feature.")
        
        if not isinstance(label, Label) or not issubclass(label.__class__, Label):
            raise TypeError("Label Model must be a subclass of Label.")
        
        if not isinstance(model, OnlineModel) or not issubclass(model.__class__, OnlineModel):
            raise TypeError("Model must be a subclass of OnlineModel.")
        
        self.features = features
        self.label = label
        self.model = model
    
    def _reset(self):
        """
        Resets the state of the indicator, clearing its rolling window and model.
        """
        self._count = 0
        self._prices.clear()
        self._atr.reset()
        self._vwap.reset()
        
        self.features = None
        self.label = None
        self.model = None
        self.metric = 0.0 
        
        self.atr = 0.0
        self.vwap = 0.0
        self.value = 0.0
        