from decimal import Decimal
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

from nautilus_trader.core.data import Data
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveInt, PositiveFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, DataType
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.enums import OrderSide, TimeInForce, OrderType
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators.rvi import RelativeVolatilityIndex
from nautilus_trader.indicators.atr import AverageTrueRange

from quantinaut.common import save_logs, trading_decision_to_order_side
from quantinaut.common import TradingDecision, TradeSignal, Volatility
from quantinaut.channels import ChannelData
from quantinaut.models import load_model

np.random.seed(100)
    
class RulePolicyConfig(StrategyConfig, frozen=True):
    """
    Configuration for RulePolicy instances, tailored for rule-based ML models.

    This configuration provides a robust set of parameters to define the behavior
    of a machine learning-based trading strategy. It covers model loading,
    feature engineering, training, and inference.

    Parameters
    ----------
    bar_types : BarType
        BarType object representing the instrument and it's timeframe.
    client_id : ClientId
        The client ID for the strategy, used for logging and identification.
    rvi_period : PositiveInt, default=9
        Period for RVI indicator.
    rvi_threshold : PositiveFloat, default=55.0
        Threshold for RVI filter
    model_path : Union[Path, str, None], default=None
        Path to load the pre-trained model from.
    scale_data : bool, default=False
        Whether to scale the data before training.
    scaler_path : Union[Path, str, None], default=None
        Path to load the scaler from.
    """
    bar_type: BarType
    client_id: ClientId = ClientId("RP-001")
    
    atr_period: PositiveInt = 21
    atr_multiple: PositiveFloat = 3.0
    trade_size: Decimal = Decimal("0.010")
    rvi_period: PositiveInt = 9
    rvi_threshold: PositiveFloat = 50.0
    model_path: Path = Path("./data/models/model.pkl")
    scale_data: bool = False
    scaler_path: Union[Path, str, None] = None
    save_logs: bool = False


class RulePolicy(Strategy):
    """
        Intelligent Trading Strategy
        
        Uses offline approach, train a model offline and deploy it using this strategy.
    """
    def __init__(self, config: RulePolicyConfig) -> None:
        PyCondition.not_none(config, "config")
        PyCondition.type(config, RulePolicyConfig, "config")
        PyCondition.type(config.bar_type, BarType, "bar_type")
        PyCondition.type(config.client_id, ClientId, "client_id")
        PyCondition.positive_int(config.rvi_period, "rvi_period")
        PyCondition.positive(config.rvi_threshold, "rvi_threshold")
    
        super().__init__(config)

        self._notification_channel_id = None
        self._trade_signal = TradingDecision.NEUTRAL
        self._volatility = Volatility.NEUTRAL

        self.instrument_id = self.config.bar_type.instrument_id

        # Initialized in on_start
        self.instrument: Optional[Instrument] = None
        self.tick_size = None

        self.rvi = RelativeVolatilityIndex(config.rvi_period)
        self.atr = AverageTrueRange(period=14)
        self.model, self.scaler = load_model({
            "model_path": config.model_path,
            "scale_data": config.scale_data,
            "scaler_path": config.scaler_path
        }, self.log)
        

    def on_start(self) -> None:
        """
        Handles strategy startup logic: initializes instruments, indicators, and attaches ML model.
        """        
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        self.tick_size = self.instrument.price_increment

        self.register_indicator_for_bars(self.config.bar_type, self.rvi)
        self.register_indicator_for_bars(self.config.bar_type, self.atr)

        # Get historical data
        self.request_bars(self.config.bar_type)

        # Subscribe to live data
        self.subscribe_quote_ticks(self.instrument_id)
        self.subscribe_bars(self.config.bar_type)
        self.subscribe_data(data_type=DataType(ChannelData))
        
    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)

        # Unsubscribe from data
        self.unsubscribe_quote_ticks(self.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
        self.unsubscribe_data(data_type=DataType(ChannelData))
    
    def on_reset(self) -> None:
        """
        Actions to be performed when the strategy is reset.
        """
        self.rvi.reset()
        self.atr.reset()
    
    def on_data(self, data: Data) -> None:
        """
        Actions to perform when the channel stops.
        """
        PyCondition.not_none(data, "data")
        
        if isinstance(data, ChannelData) and len(data.channel_id) != 0 and self._notification_channel_id is None:
            self._notification_channel_id = data.channel_id
    
    def on_bar(self, bar: Bar) -> None:
        """
        Actions to be performed when the strategy is running and receives a bar.
        Parameters
        ----------
        bar : Bar
            The bar received.
        """
        if not self.indicators_initialized():
            self.log.info(
                f"Waiting for indicators to warm up [{self.cache.bar_count(self.config.bar_type)}]",
                color=LogColor.BLUE,
            )
            return

        if bar.is_single_price():
            return
        
        self._decide(bar)
        
    def _decide(self, bar: Bar):
        """
        Make a trading decision using the ML model and current indicator values.
        Sets the trade signal and volatility, then triggers trade and logging actions.
        """
        # Map prediction to trade signal (example logic)
        if self.model is None:
            self.log.warning("Model is None, cannot proceed with trading decision.")
            return
        
        self.value = self.model.predict([])

        if self.value > 0:
            self._trade_signal = TradingDecision.ENTER_LONG
        elif self.value < 0:
            self._trade_signal = TradingDecision.ENTER_SHORT
        else:
            self._trade_signal = TradingDecision.NEUTRAL

        confidence = self.atr_vwap.metric 
        if confidence is None:
            raise ValueError("Confidence is None, cannot proceed with trading decision.")

        # Example volatility logic (can be customized)
        # TODO: try consistent increase or decrease as another approach AND check for divergence too 
        self._volatility = Volatility.NEUTRAL 
        if self.rvi.value > self.config.rvi_threshold:
            self._volatility = Volatility.BULLISH   
        elif self.rvi.value < (100 - self.config.rvi_threshold):
            self._volatility = Volatility.BEARISH

        # Execute trade and log
        self._trade(bar, confidence)
        if self.config.save_logs:
            self._save_logs(bar, prediction=self.value, confidence=confidence, trade_signal=self._trade_signal.name)
    
    def _trade(self, bar: Bar, confidence: float = 0.50):
        side = trading_decision_to_order_side(self._trade_signal)
        self.log.info(f"Trade signal: {self._trade_signal.name}.")

        # No action for NEUTRAL signals
        if side == OrderSide.NO_ORDER_SIDE:
            self.log.info("No trade action taken for NEUTRAL signal.", color=LogColor.YELLOW)
            return

        # --- Determine if RVI filter passes ---
        rvi_ok = True if (side == OrderSide.BUY and self._volatility == Volatility.BULLISH) or \
                        (side == OrderSide.SELL and self._volatility == Volatility.BEARISH) else False

        if not rvi_ok:
            self.log.info(
                f"Skipping {self._trade_signal.name} due to insufficient {Volatility.BULLISH if side == OrderSide.BUY else Volatility.BEARISH} volatility: {self.rvi.value:.2f}",
                color=LogColor.YELLOW,
            )
            return

        self.log.info(f"Placing {side.name} order.")
        # --- Entry Logic by Signal Type ---
        if self._trade_signal != TradingDecision.NEUTRAL:
            self._send_trade(self._trade_signal, entry=bar, order_side=side, confidence=confidence, 
                             entry_order_type=OrderType.MARKET, use_trailing_stop=True, atr=self.atr_vwap.atr)
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
                reason=f"RP={action.name}",
                **kwargs
            )
        self.publish_data(
                data_type=DataType(TradeSignal, metadata={"instrument_id": entry.bar_type.instrument_id.value}),
                data=data
            )
        self.log.info(f"Sent Trade Signal {data!r}", color=LogColor.CYAN)
            
    def _send_notifications(self, data: ChannelData):
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
        save_logs(data, "rule_policy_logs.csv")
