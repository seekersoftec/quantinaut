from typing import Any
from nautilus_trader.core.data import Data
from sklearn.linear_model import LinearRegression

class Model(Data):
    def __init__(
        self,
        model: Any,
        ts_init: int,
    ):
        super().__init__(ts_init=ts_init, ts_event=ts_init)
        self.model = model
        
class ModelUpdate(Data):
    def __init__(
        self,
        model: Any,
        hedge_ratio: float,
        std_prediction: float,
        ts_init: int,
    ):
        super().__init__(ts_init=ts_init, ts_event=ts_init)
        self.model = model
        self.hedge_ratio = hedge_ratio
        self.std_prediction = std_prediction


class Prediction(Data):
    def __init__(
        self,
        instrument_id: str,
        prediction: float,
        ts_init: int,
    ):
        super().__init__(ts_init=ts_init, ts_event=ts_init)
        self.instrument_id = instrument_id
        self.prediction = prediction