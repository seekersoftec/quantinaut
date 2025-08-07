from collections import deque
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Union, Literal
import pandas as pd
import numpy as np

from nautilus_trader.core.data import Data
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveFloat, PositiveInt, NonNegativeFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, DataType
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.identifiers import InstrumentId, ClientId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce, TrailingOffsetType, OrderType, TriggerType

from nautilus_ai.common import save_logs, TradingDecision, TradeSignal, Volatility
from nautilus_ai.notifications.channel import ChannelData
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
        self._bars_since_retrain: int = 0
        self._is_warmed_up: bool = False
        self._enable_context: bool = False
        
        self._notification_channel_id = None
        self._trade_signal = TradingDecision.NEUTRAL
        self._volatility = Volatility.NEUTRAL

    def on_start(self) -> None:
        """
        Handles strategy startup logic.
        """
        self.subscribe_data(data_type=DataType(ChannelData))
        pass 
    
    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)

        # Unsubscribe from data
        self.unsubscribe_quote_ticks(self.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
    
    def on_reset(self) -> None:
        """
        
        """
        pass
    
    def on_bar(self, bar: Bar) -> None:
        """
        
        """
        pass
    
    def on_data(self, data: Data) -> None:
        """
        Actions to perform when the channel stops.
        """
        PyCondition.not_none(data, "data")
        
        if isinstance(data, ChannelData) and len(data.channel_id) != 0 and self._notification_channel_id is None:
            self._notification_channel_id = data.channel_id
    
    def _get_model_decision(self, bar: Bar):
        """
        
        """
        self._trade_signal = TradingDecision.ENTER_LONG if side == OrderSide.BUY else TradingDecision.ENTER_SHORT 
        pass
    
    def _trade(self, bar: Bar, confidence: float = 0.50):
        side = OrderSide.BUY if "BUY" in self._trade_signal else OrderSide.SELL

        self.log.info(f"Trade signal: {self._trade_signal}. Placing {side} order.")

        # --- Determine if RVI filter passes ---
        rvi_ok = True if (side == OrderSide.BUY and self._volatility == Volatility.BULLISH) or \
                        (side == OrderSide.SELL and self._volatility == Volatility.BEARISH) else False

        if not rvi_ok:
            self.log.info(
                f"Skipping {self._trade_signal} due to insufficient {Volatility.BULLISH if side == OrderSide.BUY else Volatility.BEARISH} volatility: {self.rvi.value:.2f}",
                color=LogColor.YELLOW,
            )
            return

        # --- Entry Logic by Signal Type ---
        if self._trade_signal != TradingDecision.NEUTRAL:
            self._send_trade(self._trade_signal, entry=bar, order_side=side, confidence=confidence, 
                             entry_order_type=OrderType.MARKET, use_trailing_stop=True, atr=self.atr.value)
        else:
            # Catch-all log for unexpected or unmapped signals
            self.log.info(
                f"Skipping entry due to unmatched signal or unsupported condition: {self._trade_signal}",
                color=LogColor.YELLOW,
            )

    def _send_trade(self, action: TradingDecision, entry: Bar, **kwargs):
        data = TradeSignal(
                entry=entry,
                action=action,
                time_in_force=TimeInForce.GTC,
                reason=f"ITB={action.name}",
                **kwargs
            )
        self.publish_data(
                data_type=DataType(TradeSignal, metadata={"instrument_id": entry.bar_type.instrument_id.value}),
                data=data
            )
        self.log.info(f"Sent Trade Signal {data!r}", color=LogColor.CYAN)
            
    def _send_notification(self, data: ChannelData):
        """Sends notifications to the configured channel."""
        # Ensure the data has the correct ID before publishing
        PyCondition.is_true(
            data.channel_id == self._notification_channel_id,
            "Cannot send data from a channel instance that doesn't match the data's ID."
        )
        self.publish_data(
            data_type=DataType(ChannelData, metadata={"channel_id": self._notification_channel_id}),
            data=data
        )
        self.log.info(f"Sent Notification to {self._notification_channel_id}", color=LogColor.CYAN)
    
    def _save_logs(self, bar: Bar, **kwargs) -> None:
        """
        Save logs for analysis.

        Parameters
        ----------
        bar : Bar
            The current market bar.
        """
        # Ensure all arrays in the data dictionary have the same length
        data = {
            "timestamp": [pd.Timestamp(bar.ts_init, tz="UTC")],
            "instrument_id": [f"{bar.bar_type.instrument_id}"],
            "high": [float(bar.high)],
            "low": [float(bar.low)],
            "close": [float(bar.close)],
            **{key: [value] for key, value in kwargs.items()},
        }

        # Save logs
        save_logs(data, "itb_logs.csv")
