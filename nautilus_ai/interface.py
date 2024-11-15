import pandas as pd
from typing import Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from nautilus_ai.data.drawer import Model
from nautilus_trader.config import ActorConfig
from nautilus_trader.common.actor import Actor, ActorConfig
from nautilus_trader.model.data import DataType
from nautilus_trader.model.data import Bar, BarSpecification
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.datetime import secs_to_nanos, unix_nanos_to_dt
from nautilus_ai.common.utils import bars_to_dataframe, make_bar_type, ModelUpdate, Prediction


class INautilusAIModelConfig(ActorConfig):
    symbol: str
    bar_spec: str = "10-DAY-LAST" # "10-SECOND-LAST"
    min_model_timedelta: str = "14D"

class INautilusAIModel(Actor):
    def __init__(self, config: INautilusAIModelConfig):
        super().__init__(config=config)
        
        self.symbol_id = InstrumentId.from_str(config.symbol)
        self.bar_spec = BarSpecification.from_str(self.config.bar_spec)
        self.model: Optional[Model] = None
        self.hedge_ratio: Optional[float] = None
        self._min_model_timedelta = secs_to_nanos(
            pd.Timedelta(self.config.min_model_timedelta).total_seconds()
        )
        self._last_model = pd.Timestamp(0)