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
from nautilus_trader.model.enums import OrderSide, TimeInForce, TrailingOffsetType, OrderType, TriggerType

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators.rvi import RelativeVolatilityIndex

from nautilus_ai.common import save_logs, TradingDecision, TradeSignal, Volatility
from nautilus_ai.channels import ChannelData
from nautilus_ai.indicators.atr_vwap import AverageTrueRangeWithVwap
from nautilus_ai.models import LogisticRegressionModel

np.random.seed(100)
    
class SimpleRulePolicyConfig(StrategyConfig, frozen=True):
    """
    Configuration for SimpleRulePolicy instances, tailored for rule-based ML models.

    This configuration provides a robust set of parameters to define the behavior
    of a machine learning-based trading strategy. It covers model loading,
    feature engineering, training, and inference.

    Parameters
    ----------
    bar_types : BarType
        BarType object representing the instrument and it's timeframe.
    client_id : ClientId
        The client ID for the strategy, used for logging and identification.
 
    """
    bar_type: BarType
    client_id: ClientId = ClientId("ITB-001")
    data_folder: Path = Path("./DATA_ITB")
    model_path: Union[Path, str, None] = None
    scale_data: bool = False
    scaler_path: Union[Path, str, None] = None


class SimpleRulePolicyStrategy(Strategy):
    """
        Intelligent Trading Strategy
        
        Uses offline approach.
    """
    def __init__(self, config: SimpleRulePolicyConfig) -> None:
        PyCondition.type(config.data_folder, Path, "data_folder")
        PyCondition.type(config.data_sources, list, "data_sources")
        PyCondition.type(config.feature_sets, list, "feature_sets")
        PyCondition.type(config.label_sets, list, "label_sets")
        PyCondition.type(config.train_feature_sets, list, "train_feature_sets")
        PyCondition.type(config.train_features, list[str], "train_features")
        PyCondition.type(config.labels, list[str], "labels")

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
        
        self.instrument_id = self.config.bar_type.instrument_id

        # Initialized in on_start
        self.instrument: Instrument | None = None
        self.tick_size = None
        
        self.rvi = RelativeVolatilityIndex(config.rvi_period)
        self.atr_vwap = AverageTrueRangeWithVwap(
            period=config.atr_vwap_period,
            price_type=config.bar_type.price_type,
            process_batch=config.process_batch
        )

    def on_start(self) -> None:
        """
        Handles strategy startup logic.
        """
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        self.tick_size = self.instrument.price_increment

        self.register_indicator_for_bars(self.config.bar_type, self.rvi)
        self.register_indicator_for_bars(self.config.bar_type, self.atr_vwap)

        # Get historical data
        self.request_bars(self.config.bar_type)

        # Subscribe to live data
        self.subscribe_quote_ticks(self.instrument_id)
        self.subscribe_bars(self.config.bar_type) 
        self.subscribe_data(data_type=DataType(ChannelData))
        
        self.atr_vwap.set_model(
            features=,
            label=,
            model=LogisticRegressionModel(
            l2=self.config.l2,
            l1=self.config.l1
        ))
        
        if self.config.model_path:
            self.log.info("Loading pre-trained model.")
            self._load_model()
            self.simple_set.model.load(self.config.model_path)
        
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
    
    def on_load(self) -> None:
        """
        Loads the model and scaler when the strategy state is loaded.
        """
        self._load_model()
    
    def on_save(self) -> None:
        """
        Saves the model and scaler when the strategy state is saved.
        """
        self._save_model()
    
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
        Model Decision
        """
        
        # self._trade_signal = TradingDecision.ENTER_LONG if side == OrderSide.BUY else TradingDecision.ENTER_SHORT 

        # self._trade()
        # self._send_notifications()
        # self._save_logs()
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
        save_logs(data, "itb_logs.csv")

    def _load_model(self):
        """
        Loads a model and a scaler (if configured) from file.
        """
        # Load the model
        if self.config.model_path:
            path = Path(self.config.model_path)
            if not path.exists():
                self.log.warning(f"Model file not found: {self.config.model_path}. Skipping model load.")
                return # Exit if model path is given but file doesn't exist

            ext = str(path).lower()
            if ext.endswith(".joblib"):
                try:
                    import joblib
                    self.model = joblib.load(path)
                except ImportError:
                    self.log.error("joblib is not installed. Please install with `pip install joblib`.")
            elif ext.endswith((".h5", ".keras")):
                try:
                    from keras.models import load_model
                    self.model = load_model(path)
                except ImportError:
                    self.log.error("Keras is not installed. Please install with `pip install keras`.")
            elif ext.endswith(".pkl"):
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
            else:
                self.log.error(f"Unsupported model format: {ext}")

            if self.model is not None:
                self.log.info(f"Model loaded from {self.config.model_path}", color=LogColor.GREEN)

        # Load the scaler if configured
        if self.config.scale_data:
            if not self.config.scaler_path:
                self.log.warning("`scale_data` is True but `scaler_path` is not provided. Cannot load scaler.")
                return

            scaler_path = Path(self.config.scaler_path)
            if not scaler_path.exists():
                self.log.warning(f"Scaler file not found: {self.config.scaler_path}. Skipping scaler load.")
                return
            
            try:
                import joblib
                self.scaler = joblib.load(scaler_path)
                self.log.info(f"Scaler loaded from {self.config.scaler_path}", color=LogColor.GREEN)
            except ImportError:
                self.log.error("joblib is not installed. Please install with `pip install joblib`.")
            except Exception as e:
                self.log.error(f"Failed to load scaler from {scaler_path}: {e}")


    def _save_model(self):
        """
        Saves the model and scaler (if it exists) to their respective paths.
        """
        # Save the model
        if self.model and self.config.model_path:
            path = Path(self.config.model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            ext = str(path).lower()

            try:
                if ext.endswith(".joblib"):
                    import joblib
                    joblib.dump(self.model, path)
                elif ext.endswith((".h5", ".keras")):
                    self.model.save(path)
                elif ext.endswith(".pkl"):
                    with open(path, "wb") as f:
                        pickle.dump(self.model, f)
                else:
                    self.log.error(f"Unsupported model format for saving: {ext}")
                    return
                self.log.info(f"Model saved to {self.config.model_path}", color=LogColor.GREEN)
            except Exception as e:
                self.log.error(f"Failed to save model to {path}: {e}")

        # Save the scaler
        if self.scaler and self.config.scaler_path:
            scaler_path = Path(self.config.scaler_path)
            scaler_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                import joblib
                joblib.dump(self.scaler, scaler_path)
                self.log.info(f"Scaler saved to {self.config.scaler_path}", color=LogColor.GREEN)
            except ImportError:
                self.log.error("joblib is not installed. Please install with `pip install joblib`.")
            except Exception as e:
                self.log.error(f"Failed to save scaler to {scaler_path}: {e}")
