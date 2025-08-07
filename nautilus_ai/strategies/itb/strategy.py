from collections import deque
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Union, Literal
import pandas as pd
import numpy as np

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveFloat, PositiveInt, NonNegativeFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.identifiers import InstrumentId, ClientId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce, TrailingOffsetType, OrderType, TriggerType

from nautilus_ai.strategies.itb.config import AlgorithmConfig, ITBConfig, OutputSetConfig, RollingPredictConfig, SignalSetConfig

np.random.seed(100)
    

class ITB(Strategy):
    """
        Intelligent Trading Strategy
    """
    def __init__(self, config: ITBConfig) -> None:
        PyCondition.type(config.data_folder, Path, "data_folder")
        PyCondition.type(config.data_sources, list, "data_sources")
        PyCondition.type(config.feature_sets, list, "feature_sets")
        PyCondition.type(config.label_sets, list, "label_sets")
        PyCondition.type(config.train_feature_sets, list, "train_feature_sets")
        PyCondition.type(config.train_features, list[str], "train_features")
        PyCondition.type(config.labels, list[str], "labels")
        PyCondition.type(config.algorithms, list[AlgorithmConfig], "algorithms")
        PyCondition.type(config.signal_sets, list[SignalSetConfig], "algorithms")
        PyCondition.type(config.output_sets, list[OutputSetConfig], "algorithms")
        PyCondition.type(config.rolling_predict, RollingPredictConfig, "rolling_predict")
        
        super().__init__(config)
        
        self.model = None
        self.scaler = None
        self._instruments: dict[Union[InstrumentId, str], BarType] = {}
        self._bar_history: dict[Union[InstrumentId, str], deque] = {}
        self._expanding_history: dict[Union[InstrumentId, str], list] = {}  # For expanding window
        self._bars_since_retrain: int = 0
        self._is_warmed_up: bool = False
        self._enable_context: bool = False

    def on_start(self) -> None:
        """
        Handles strategy startup logic.
        """
        pass 
    
    def on_stop(self) -> None:
        """
        Handles strategy shutdown logic.
        """
        pass
    
    def on_bar(self, bar: Bar) -> None:
        """
        
        """
        pass
    
    def on_reset(self) -> None:
        """
        
        """
        pass
    
