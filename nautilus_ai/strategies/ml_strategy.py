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
from nautilus_trader.indicators.atr import AverageTrueRange

from nautilus_ai.strategies.config import RiskModelConfig
from nautilus_ai.strategies.execution.risk_models import RiskModelFactory
from nautilus_ai.common.enums import MLFramework, MLModelType, MLLearningType, MLTaskType

np.random.seed(100)

class MLStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for MLStrategy instances, tailored for ML/DL/RL models.

    This configuration provides a robust set of parameters to define the behavior
    of a machine learning-based trading strategy. It covers model loading,
    feature engineering, training, and inference.

    Parameters
    ----------
    bar_types : List[BarType]
        List of BarType objects representing the instruments and their timeframes
    client_id : ClientId
        The client ID for the strategy, used for logging and identification.
    train_test_split : float
        Fraction of data to use for training in walk-forward analysis.
    model_path : Union[Path, str, None]
        Path to the pre-trained machine learning model file.
        (e.g., .joblib, .pkl for scikit-learn, .h5, .keras for TensorFlow/Keras).
    model_type : MLModelType
        The type of machine learning model.
        >>> MLModelType(
        ...     learning_type=MLLearningType.SUPERVISED,
        ...     task_type=MLTaskType.CLASSIFICATION
        ... )
        (task_type e.g., CLASSIFICATION, REGRESSION, REINFORCEMENT_LEARNING).
    framework : MLFramework
        The machine learning framework used to build the model.
        (e.g., SCIKIT_LEARN, TENSORFLOW, PYTORCH).
    scale_data : bool
        Whether to scale input features before prediction. Defaults to False.
    scaler_path : Union[Path, str, None]
        Path to a saved scaler object (e.g., from scikit-learn).
        Required if `scale_data` is True.
    lookback_window : PositiveInt
        The number of past data points to use for generating features.
    prediction_threshold : float
        Threshold for converting model outputs (e.g., probabilities) into
        trading signals. For example, a value of 0.5 for a binary
        classification model.
    feature_config : dict
        A dictionary defining parameters for feature generation. This can be
        used to pass custom settings to the `prices_to_features` method.
    train_on_start : bool
        If True, the strategy will perform an initial training run on historical
        data when it starts. Defaults to False, assuming a pre-trained model is loaded.
    initial_training_period_days : PositiveInt
        The number of days of historical data to use for the initial training.
        Only used if `train_on_start` is True.
    retrain_every : PositiveInt | None
        The interval (in number of bars) at which to retrain the model.
        If None, the model is not retrained. Defaults to None.
    learning_rate : PositiveFloat | None
        The learning rate for the model's optimizer, for online training.
    rl_algorithm : str | None
        The reinforcement learning algorithm to use (e.g., 'PPO', 'A2C').
        Applicable only if `model_type` is REINFORCEMENT_LEARNING.
    environment_id : str | None
        The ID of the trading environment for RL agents.
    window_type : Literal["sliding", "expanding"] = "sliding"
        The type of walk-forward window to use: 'sliding' for a fixed-size window
        that moves forward in time, or 'expanding' for a window that grows
        progressively larger.
    volatility_period : PositiveInt, default=14
        Period for Average True Range (ATR), used to scale SL/TP distances and measure Volatility.
    max_trade_size : Decimal, default="0.01"
        Maximum allowable trade size in units of the asset.
    position_sizer : RiskModelConfig, default=dict(risk_pct=0.1)
        Configuration for the position sizing algorithm.
    emulation_trigger : TriggerType, default=TriggerType.NO_TRIGGER
        Determines how/when orders are triggered in emulation/live modes.
    """
    bar_types: List[BarType] 
    client_id: ClientId = ClientId("ML-001")
    train_test_split: float = 0.5 
    model_path: Union[Path, str, None] = None
    model_type: MLModelType = MLModelType(learning_type=MLLearningType.SUPERVISED, task_type=MLTaskType.CLASSIFICATION)
    framework: MLFramework = MLFramework.SCIKIT_LEARN
    scale_data: bool = False
    scaler_path: Union[Path, str, None] = None
    lookback_window: PositiveInt = 60
    prediction_threshold: NonNegativeFloat = 0.6
    feature_config: dict = {}
    train_on_start: bool = False
    initial_training_period_days: PositiveInt = 365
    retrain_every: Union[PositiveInt, None] = None
    learning_rate: Union[PositiveFloat, None] = None
    rl_algorithm: Union[str, None] = None
    environment_id: Union[str, None] = None
    window_type: Literal["sliding", "expanding"] = "sliding"  # 'sliding' or 'expanding'
    
    # Risk management
    volatility_period: PositiveInt = 14
    max_trade_size: Decimal = Decimal("0.01")
    position_sizer: RiskModelConfig = RiskModelConfig(init_args=dict(risk_pct=0.1))
    emulation_trigger: TriggerType = TriggerType.NO_TRIGGER



class MLStrategy(Strategy):
    """
    Base class for Sage machine learning strategies.

    This class provides a framework for developing and backtesting ML-driven
    strategies. It includes logic for model loading, saving, inference, and
    periodic retraining (walk-forward analysis).

    **Workflow:**

    1.  **Initial Training:** It is recommended to train your model initially on a
        large historical dataset *before* running the strategy. The strategy will
        then load this pre-trained model on start.

    2.  **Inference:** On each new bar (`on_bar`), the strategy converts the recent
        price history into features, predicts using the loaded model, and generates
        trading signals.

    3.  **Retraining:** The strategy can be configured to automatically retrain the
        model on a periodic basis (`retrain_every`) using the most recent data,
        enabling a walk-forward optimization approach.

    To create a strategy, subclass this and implement the required abstract methods
    like `prices_to_features` and `predictions_to_signals`.
    
    TODO: 
        - Create a utility for merging Multiple timeframes for analysis
        - Tune the order placment utility 
    """
    
    def __init__(self, config: MLStrategyConfig) -> None:
        PyCondition.type(config.model_type, MLModelType, "model_type")
        super().__init__(config)
        self.model = None
        self.scaler = None
        self._instruments: dict[Union[InstrumentId, str], BarType] = {}
        self._bar_history: dict[Union[InstrumentId, str], deque] = {}
        self._expanding_history: dict[Union[InstrumentId, str], list] = {}  # For expanding window
        self._bars_since_retrain: int = 0
        self._is_warmed_up: bool = False
        self._enable_context: bool = False
        self.position_sizer_factory = RiskModelFactory.create(**config.position_sizer.__dict__)
        self.atr_volatility = AverageTrueRange(period=config.volatility_period)

    def on_start(self) -> None:
        """
        Handles strategy startup logic.

        - Initializes instrument subscriptions.
        - Loads a pre-trained model or initiates training if configured.
        """
        self._subscribe_to_instruments()
        
        # register indicators | Supports only one indicator for now 
        self.register_indicator_for_bars(self.config.bar_types[0], self.atr_volatility)
        
        if self.config.train_on_start:
            self._initiate_initial_training()
        elif self.config.model_path:
            self.log.info("Loading pre-trained model.")
            self._load_model()

    def _subscribe_to_instruments(self) -> None:
        """Subscribes to instruments based on the strategy configuration."""
        for bar_type in self.config.bar_types:
            instrument = self.cache.instrument(bar_type.instrument_id)
            if instrument:
                self._instruments[bar_type.instrument_id] = bar_type
                self.subscribe_quote_ticks(bar_type.instrument_id)
                self.subscribe_trade_ticks(bar_type.instrument_id)
                self.subscribe_bars(bar_type, self.config.client_id)
            else:
                self.log.error(f"Could not find instrument with ID {bar_type.instrument_id}", color=LogColor.RED)
                self.stop()
                return

    def _initiate_initial_training(self) -> None:
        """Requests historical data for all configured instruments to start the initial training process."""
        self.log.info("Initial training on start is enabled.")

        if not self.config.bar_types:
            self.log.error("Cannot initiate training: `bar_types` is not configured.", color=LogColor.RED)
            self.stop()
            return

        self.log.info(f"Requesting {self.config.initial_training_period_days} days of historical data for all instruments...")

        end_date = datetime.fromtimestamp(self.clock.timestamp())
        start_date = end_date - timedelta(days=self.config.initial_training_period_days)

        for bar_type in self.config.bar_types:
            self.request_bars(
                bar_type=bar_type,
                start=start_date,
                end=end_date,
            )
        
        # The actual training is deferred to `_perform_initial_training`,
        # which should be called after the historical data has been received.
        # Depending on the framework, this might be handled in a callback or
        # by checking data availability in a subsequent event.
        self._perform_initial_training()

    def on_stop(self) -> None:
        """
        Actions to be performed on strategy stop.
        """
        for bar_type in self.config.bar_types:
            self.unsubscribe_quote_ticks(bar_type.instrument_id)
            self.unsubscribe_trade_ticks(bar_type.instrument_id)
            self.unsubscribe_bars(bar_type) 
        
        # self.log.info("Strategy is stopping. Saving model and scaler state...")
        # self._save_model()

    def on_reset(self) -> None:
        """
        Actions to be performed on strategy reset.
        """
        self._bars_since_retrain = 0
        self._is_warmed_up = False
        self._instruments.clear()
        self.atr_volatility.reset()

        # for RL
        if self.config.model_type.learning_type == MLLearningType.REINFORCEMENT:
            self.log.info("Resetting RL environment...")
            self.reset_env()

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

    def on_bar(self, bar: Bar) -> None:
        """
        Main event handler for incoming bars.
        Implements walk-forward training: supports both sliding and expanding windows.
        """
        if self._enable_context:
            self.update_context(bar)
        instrument_id = bar.bar_type.instrument_id if hasattr(bar.bar_type, 'instrument_id') else bar.id
        # Sliding window: use deque
        if self.config.window_type == "sliding":
            if instrument_id not in self._bar_history:
                self._bar_history[instrument_id] = deque(maxlen=self.config.lookback_window)
            self._bar_history[instrument_id].append(bar)
            # Only proceed if all instruments have enough bars
            if not all(len(bars) >= self.config.lookback_window for bars in self._bar_history.values()):
                return
            history_df = self._get_combined_bar_history_df()
        # Expanding window: accumulate all bars
        else:
            if instrument_id not in self._expanding_history:
                self._expanding_history[instrument_id] = []
            self._expanding_history[instrument_id].append(bar)
            # Only proceed if all instruments have at least lookback_window bars
            if not all(len(bars) >= self.config.lookback_window for bars in self._expanding_history.values()):
                return
            history_df = self._get_combined_expanding_history_df()
        if history_df.empty:
            return
        n = len(history_df)
        split_idx = int(n * self.config.train_test_split)
        train_df = history_df.iloc[:split_idx]
        test_df = history_df.iloc[split_idx:]
        if len(train_df) < 2 or len(test_df) < 1:
            return
        features_train, targets_train = self.prices_to_features(train_df)
        features_test, targets_test = self.prices_to_features(test_df)
        self.train(features_train, targets_train)
        try:
            preds = self.predict(features_test)
            preds = self._reshape_predictions(preds)
            if self.config.model_type.task_type == MLTaskType.CLASSIFICATION:
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(targets_test, np.round(preds))
                self.log.info(f"Walk-forward test accuracy: {acc:.4f}")
            elif self.config.model_type.task_type == MLTaskType.REGRESSION:
                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(targets_test, preds)
                self.log.info(f"Walk-forward test MSE: {mse:.4f}")
        except Exception as e:
            self.log.warning(f"Walk-forward test failed: {e}")
        # Inference on the latest bar(s)
        if self.config.window_type == "sliding":
            latest_df = history_df.iloc[-self.config.lookback_window:]
        else:
            latest_df = history_df.iloc[-self.config.lookback_window:]  # For expanding, still use last window for inference
        signals = self._prices_to_signals(latest_df)
        self.act(instrument_id, signals)

    def _get_combined_bar_history_df(self) -> pd.DataFrame:
        """
        Combines the _bar_history of all instruments into a single DataFrame (like _get_combined_history_df but from _bar_history).
        """
        all_dfs = []
        for instrument_id, bars in self._bar_history.items():
            if not bars:
                continue
            df = pd.DataFrame([
                {
                    "timestamp": bar.ts_event,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                }
                for bar in bars
            ])
            df.set_index("timestamp", inplace=True)
            df["instrument_id"] = instrument_id if isinstance(instrument_id, str) else instrument_id.value
            all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        combined_df = pd.concat(all_dfs)
        combined_df = combined_df.set_index(["instrument_id", combined_df.index])
        combined_df = combined_df.unstack(level="instrument_id").swaplevel(0, 1, axis=1).sort_index(axis=1)
        combined_df.index.names = ["timestamp"]
        combined_df.columns.names = ["instrument_id", "Field"]
        return combined_df

    def _get_combined_expanding_history_df(self) -> pd.DataFrame:
        """
        Combines the _expanding_history of all instruments into a single DataFrame.
        """
        all_dfs = []
        for instrument_id, bars in self._expanding_history.items():
            if not bars:
                continue
            df = pd.DataFrame([
                {
                    "timestamp": bar.ts_event,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                }
                for bar in bars
            ])
            df.set_index("timestamp", inplace=True)
            df["instrument_id"] = instrument_id if isinstance(instrument_id, str) else instrument_id.value
            all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        combined_df = pd.concat(all_dfs)
        combined_df = combined_df.set_index(["instrument_id", combined_df.index])
        combined_df = combined_df.unstack(level="instrument_id").swaplevel(0, 1, axis=1).sort_index(axis=1)
        combined_df.index.names = ["timestamp"]
        combined_df.columns.names = ["instrument_id", "Field"]
        return combined_df

    def _perform_initial_training(self):
        """
        Requests and processes historical data for initial model training.
        
        NOTE: This is a conceptual implementation. You will need to adapt this
        to your specific data provider and framework to fetch historical data.
        """
        bar_type = self._get_bar_type()

        # In a real scenario, you would request data for each instrument.
        # This is a simplified example for one instrument.
        instrument = self.instruments[0]
        
        # ======================= CONCEPTUAL CODE ========================
        # Replace this block with your framework's method for fetching data.
        # Example:
        # historical_bars = self.request_bars(
        #     instrument_id=instrument.id,
        #     start_date=self.clock.timestamp() - timedelta(days=self.config.initial_training_period_days),
        #     end_date=self.clock.timestamp(),
        # )
        # historical_df = self._bar_list_to_dataframe(historical_bars)
        # ==============================================================
        
        # For demonstration, we'll assume `historical_df` is available.
        # In a real run, you would need to populate it with actual data.
        # If you run this as is, it will fail unless you provide data
        bar_count = self.cache.bar_count(bar_type)
        if bar_count < self.config.initial_training_period_days:
            self.log.info(f"Courent Bar count not enough: {self.config.initial_training_period_days}")
            return

        bars = self.cache.bars(self.bar_type)

        # Convert list of Bar objects to dicts
        bar_dicts = [bar.to_dict() for bar in bars]

        # Create DataFrame
        historical_df = pd.DataFrame(bar_dicts)

        # Optional: convert timestamps to datetime
        historical_df['ts_event'] = pd.to_datetime(historical_df['ts_event'], utc=True)
        historical_df['ts_init'] = pd.to_datetime(historical_df['ts_init'], utc=True)

        # Optional: set index
        historical_df.set_index('ts_event', inplace=True)

        # historical_df = pd.DataFrame() # Replace with actual data

        if historical_df.empty:
            self.log.error("Historical data is empty. Cannot perform initial training. Please provide data.")
            return

        self.log.info(f"Received {len(historical_df)} data points. Starting initial training...")
        features, targets = self.prices_to_features(historical_df)
        self.train(features, targets)
        self.log.info("Initial training complete.")
        self._save_model()
        
    def _retrain_model(self):
        """
        Coordinates the retraining of the model using the latest data.
        """
        # This is a simplified example assuming single-instrument data.
        # For multi-instrument strategies, you would need to combine data appropriately.
        instrument_id = next(iter(self.instruments)).id
        training_df = self._bar_history_to_dataframe(instrument_id)

        # Generate features and targets from the latest data
        features, targets = self.prices_to_features(training_df)

        # The `train` method must be implemented by the subclass
        self.log.info("Calling train() method on new data...")
        self.train(features, targets)
        self.log.info("Training complete.")

        # Save the newly trained model
        self._save_model()

    def _bar_history_to_dataframe(self, instrument_id: str) -> pd.DataFrame:
        """
        Converts the list of Bar objects into a pandas DataFrame.
        """
        bars = self.cache.bars(instrument_id)
        df = pd.DataFrame([
            {
                "timestamp": bar.ts_event,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            }
            for bar in bars
        ])
        df.set_index("timestamp", inplace=True)
        # Create a multi-index similar to what `prices_to_features` might expect
        df = df.unstack().to_frame().swaplevel(0, 1).sort_index()
        df.index.names = ["Field", "timestamp"]
        return df

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

    def _make_identifier(self, df: pd.DataFrame, prefix: Optional[str] = None) -> str:
        """
        Generates a unique identifier string based on the DataFrame's index and columns,
        and an optional prefix.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to generate an identifier for.

        prefix : Optional[str]
            An optional prefix to include in the identifier (e.g., 'features').

        Returns
        -------
        str
            A unique identifier string.
        """
        base = str(df.index.tolist()) + str(df.columns.tolist())
        _hash = hashlib.md5(base.encode()).hexdigest()
        if prefix is not None:
            return f"{self.id}_{prefix}_{_hash}"
        return f"{self.id}_{_hash}"
    
    def _calculate_position_size(
        self,
        price: Decimal,
        instrument: Instrument
        ) -> Decimal:
            """
            Calculates the total trade size using the configured position sizing algorithm,
            stop-loss distance, and current market price, while enforcing instrument limits.

            Parameters:
            -----------
                price (Decimal): Current market price of the instrument.
                instrument (Instrument): The instrument being traded.
                
            Returns:
            --------
                Decimal: Total position size.

            Raises:
            -------
                ValueError: If stop-loss distance or price is zero or negative.
            """
            if price <= 0:
                raise ValueError("Price must be > 0")

            # Get account and free capital
            account = self.cache.account_for_venue(instrument.id.venue)
            free_balance = Decimal(account.balance_free(instrument.quote_currency).as_decimal())

            # Compute capital to risk using position sizer
            risk_amt = self.position_sizer_factory.compute_size(capital=free_balance)
            self.log.info(f"Calculated risk amount: {risk_amt}", color=LogColor.CYAN)

            # Convert to raw position size based on SL distance
            raw_size = risk_amt / Decimal(price)

            # Ensure notional value meets minimum
            notional = raw_size * Decimal(price)
            if instrument.min_notional is not None:
                min_notional = Decimal(instrument.min_notional.as_decimal())
                if notional < min_notional:
                    raw_size = Decimal(min_notional) / Decimal(price)
                    self.log.info(
                        f"Adjusted position size to meet minimum notional value: {raw_size}",
                        color=LogColor.YELLOW
                    )

            # Check min_quantity
            if instrument.min_quantity is not None:
                if raw_size < instrument.min_quantity.as_decimal():
                    raw_size = instrument.min_quantity.as_decimal()
                    self.log.info(
                        f"Adjusted position size to meet min_quantity: {raw_size}",
                        color=LogColor.YELLOW
                    )

            # Check max_quantity
            if instrument.max_quantity is not None:
                if raw_size > instrument.max_quantity.as_decimal():
                    raw_size = instrument.max_quantity.as_decimal()
                    self.log.info(
                        f"Adjusted position size to meet max_quantity: {raw_size}",
                        color=LogColor.YELLOW
                    )

            # Check max_notional
            if instrument.max_notional is not None:
                max_notional = Decimal(instrument.max_notional.as_decimal())
                notional = raw_size * Decimal(price)  # recalculate after adjustments
                if notional > max_notional:
                    raw_size = Decimal(max_notional) / Decimal(price)
                    self.log.info(
                        f"Adjusted position size to meet max_notional: {raw_size}",
                        color=LogColor.YELLOW
                    )

            self.log.info(f"raw_size: {raw_size}")
            # Quantize to instrument's size precision
            size_precision = Decimal(str(instrument.size_increment))
            total_size = float(raw_size.quantize(size_precision, rounding=ROUND_DOWN))

            self.log.info(f"Final calculated size: {total_size}", color=LogColor.GREEN)

            return Decimal(total_size)

    def _submit_order(self, instrument: Instrument, side: OrderSide, price: Decimal, exit_price: Optional[Decimal] = None) -> None:
        """
        Submit a bracket order using predefined risk-reward parameters and trailing stop logic.

        Parameters
        ----------
        instrument : Instrument
            The financial instrument to trade.
        
        side : OrderSide
            The side of the order, either BUY or SELL.
            
        price : Decimal
            The intended entry price for the order.
        
        exit_price : Decimal | None
            The intended exit(profit) price for the order.
        

        Notes
        -----
        - The stop-loss (SL) is calculated using 1.5× ATR.
        - The take-profit (TP) is calculated using a 2.5× SL distance.
        - TP is implemented as a TRAILING_STOP_MARKET with a 300 basis points trailing offset.
        - All orders use GTC (Good Till Cancelled) time-in-force.

        This method builds a bracket order using Nautilus Trader's `order_factory.bracket()`, including:
        - Entry order
        - Stop-loss order
        - Trailing take-profit order
        
        TODO: Would still need to tune this
        
        Add leverage to this: position_size * leverage e.g: 0.1 * 3
        
        """
        if not isinstance(price, Decimal):
            price = Decimal(price)
            
        # Risk/Reward Parameters
        # sl_multiplier = Decimal("1.5")
        # tp_multiplier = Decimal("2.5")
        atr_value = Decimal(self.atr_volatility.value)
        atr_pct = atr_value / Decimal(price)
                
        # min_val = np.percentile(self.secondary_ctx.atr_history, 30) # tighter trail in calm conditions
        # max_val = np.percentile(self.secondary_ctx.atr_history, 80) # wider trail in high volatility
        # vol_rank = np.clip((float(atr_value) - min_val) / (max_val - min_val), 0, 1)
        # trailing_offset_bps = Decimal("30") + Decimal("40") * Decimal(vol_rank)
        # trailing_offset_bps = Decimal("0.01") * Decimal(price)
    
        # Use ATR as a dynamic trailing stop offset
        trailing_multiplier = Decimal(1.5)
        trailing_offset = trailing_multiplier * atr_pct * Decimal(price)
        trailing_offset_bps = (trailing_offset / Decimal(price)) * Decimal("10000")
        # if confidence > 0.9:
        #     trailing_offset = Decimal("0.008") * Decimal(price)  # tighter trail
        # elif confidence < 0.6:
        #     trailing_offset = Decimal("0.015") * Decimal(price)  # looser


        # sl_offset = atr_value * sl_multiplier
        # tp_offset = sl_offset * tp_multiplier
        sl_offset = Decimal("0.01") * Decimal(price)
        tp_offset = Decimal("0.03") * Decimal(price)

        if side == OrderSide.BUY:
            sl_price = price - sl_offset
            tp_price = price + tp_offset
        else:
            sl_price = price + sl_offset
            tp_price = price - tp_offset

        if exit_price is not None:
            tp_price = Decimal(exit_price)
            
        # Position Sizing 
        position_size = self._calculate_position_size(price, instrument)
        trade_size = min(position_size, self.config.max_trade_size) 
    
        # Construct bracket order
        orders = self.order_factory.bracket(
            instrument_id=instrument.id,
            order_side=side,
            quantity=instrument.make_qty(trade_size),
            entry_price=instrument.make_price(price),
            sl_trigger_price=instrument.make_price(sl_price),
            emulation_trigger=self.config.emulation_trigger,

            # Trailing take-profit setup
            tp_order_type=OrderType.TRAILING_STOP_MARKET, # LIMIT | TRAILING_STOP_MARKET
            tp_activation_price=instrument.make_price(tp_price),
            tp_price=instrument.make_price(tp_price),  # used only for reference by some brokers
            tp_trigger_price=instrument.make_price(tp_price),
            tp_trailing_offset=trailing_offset_bps,
            tp_trailing_offset_type=TrailingOffsetType.BASIS_POINTS,

            # TIF settings
            time_in_force=TimeInForce.GTC,
            sl_time_in_force=TimeInForce.GTC,
            tp_time_in_force=TimeInForce.GTC,
        )

        # Submit to the exchange
        self.submit_order_list(orders)
        self.log.info(
            f"Submitted {side.name} bracket order on {instrument.id} | "
            f"Entry: {price:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}"
            f"Quantity: {trade_size:.8f}",
            color=LogColor.GREEN,
        )
    
    def _prices_to_signals(self, prices: pd.DataFrame, no_cache: bool = False):
        """
        Converts a prices DataFrame to signals using an ML model.

        Steps:
        - Converts prices to features (and optionally targets)
        - Predicts using model on features
        - Converts predictions to signals
        
        serve features from cache in backtests if possible. The features are cached
        based on the index and columns of prices. If this file has been
        edited more recently than the features were cached, the cache is
        not used.
        """
        cache_key = self._make_identifier(prices, prefix="features")
        features = None

        # Use cache
        if not no_cache:
            features_bytes = self.cache.get(cache_key)
            if features_bytes is not None:
                features = pickle.loads(features_bytes)

        # If not cached, compute features & cache them
        if features is None:
            features = self.prices_to_features(prices)
            features_bytes = pickle.dumps(features)
            self.cache.add(cache_key, features_bytes)

        # Validate output format
        if not isinstance(features, tuple) or len(features) != 2:
            raise ValueError("prices_to_features should return a tuple of (features, targets)")

        features, _ = features  # discard targets

        features, index, unstack = self._prepare_features(features)

        # Predict
        predictions = self.predict(features)
        predictions = self._reshape_predictions(predictions)

        predictions = pd.Series(predictions, index=index)
        if unstack:
            # The name of the level to unstack is the name of the columns index of the feature DataFrames
            level_name = features[0].columns.name if isinstance(features, list) else next(iter(features.values())).columns.name
            predictions = predictions.unstack(level=level_name or 0)

        return self.predictions_to_signals(predictions, prices)

    def _prepare_features(self, features: Union[pd.DataFrame, dict, list]) -> tuple[np.ndarray, pd.Index, bool]:
        """
        Prepares input features for the ML model.

        Converts input features (DataFrame, Series, or collection of these)
        into a NumPy array, applies scaling if a scaler is provided, and returns:
            - np.ndarray of processed features
            - index for reconstructing predictions
            - bool indicating whether to unstack the prediction Series later
        """
        if isinstance(features, pd.DataFrame):
            index = features.index
            X = features.fillna(0).values
            if self.scaler:
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    self.log.warning(f"Scaler not fitted or transform failed: {e}. Skipping scaling.")
                    X_scaled = X
            else:
                X_scaled = X
            return X_scaled, index, False

        if isinstance(features, dict):
            features = list(features.values())

        if not isinstance(features, list):
            raise TypeError(f"Unsupported feature type: {type(features)}")

        feature_list = []
        index = None
        unstack = False
        has_df = has_series = False

        for i, f in enumerate(features):
            if isinstance(f, pd.DataFrame):
                if has_series:
                    raise ValueError("Cannot mix Series and DataFrames in features.")
                has_df = True
                unstack = True
                f = f.stack(dropna=False)
            elif isinstance(f, pd.Series):
                if has_df:
                    raise ValueError("Cannot mix Series and DataFrames in features.")
                has_series = True
            else:
                raise ValueError("Each feature must be a DataFrame or Series.")

            f = f.fillna(0)

            if i == 0:
                index = f.index

            feature_list.append(f.values)

        X = np.stack(feature_list, axis=-1)
        if self.scaler:
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                self.log.warning(f"Scaler not fitted or transform failed: {e}. Skipping scaling.")
                X_scaled = X
        else:
            X_scaled = X

        return X_scaled, index, unstack

    def _reshape_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Handles reshaping predictions to 1D array as needed.
        
        Keras output has (n_samples,1) shape and needs to be squeezed.
        
        predict_proba has (n_samples,2) shape where first col is probablity of
        0 (False) and second col is probability of 1 (True); we just want the
        second col (https://datascience.stackexchange.com/a/22821)
        """
        if predictions.ndim == 2:
            if predictions.shape[-1] == 1:
                return predictions.squeeze(axis=-1)
            elif hasattr(self.model, "classes_") and list(self.model.classes_) == [0, 1]:
                return predictions[:, -1]
            else:
                raise NotImplementedError(
                    f"Unhandled 2D prediction shape: {predictions.shape}"
                )
        return predictions

    def update_context(self, bar: Bar) -> None:
        raise NotImplementedError("strategies must implement update_context if enabled")
        
    def prices_to_features(
        self,
        prices: pd.DataFrame
        ) -> tuple[
            Union[
                list[Union[pd.DataFrame, 'pd.Series[float]']],
                dict[str, Union[pd.DataFrame, 'pd.Series[float]']]
            ],
            Union[pd.DataFrame, 'pd.Series[float]']]:
        """
        From a DataFrame of prices, return a tuple of features and targets to be
        provided to the machine learning model.

        The returned features can be a list or dict of DataFrames, where each
        DataFrame is a feature and should have the same shape, with a Date or
        (Date, Time) index and sids as columns. (the strategy will convert the
        DataFrames to the format expected by the machine learning model).

        Alternatively, a list or dict of Series can be provided, which is
        suitable if using multiple securities to make predictions for a
        single security (for example, an index).

        The returned targets should be a DataFrame or Series with an index
        matching the index of the features DataFrames or Series. Targets are
        used in training and are ignored for prediction. (Model training is
        not handled by the base ml strategy class.) Alternatively return None if
        using an already trained model.

        Must be implemented by strategy subclasses.

        Parameters
        ----------
        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        tuple of (dict or list of DataFrames or Series, and DataFrame or Series)
            features and targets

        Examples
        --------
        Predict next-day returns based on 1-day and 2-day returns::

            def prices_to_features(self, prices: pd.DataFrame):
                closes = prices.loc["Close"]
                features = {}
                features["returns_1d"]= closes.pct_change()
                features["returns_2d"] = (closes - closes.shift(2)) / closes.shift(2)
                targets = closes.pct_change().shift(-1)
                return features, targets

        Predict next-day returns for a single security in the prices
        DataFrame using another security's returns::

            def prices_to_features(self, prices: pd.DataFrame):
                closes = prices.loc["Close"]
                closes_to_predict = closes[12345]
                closes_to_predict_with = closes[23456]
                features = {}
                features["returns_1d"]= closes_to_predict_with.pct_change()
                features["returns_2d"] = (closes_to_predict_with - closes_to_predict_with.shift(2)) / closes_to_predict_with.shift(2)
                targets = closes_to_predict.pct_change().shift(-1)
                return features, targets
        """
        raise NotImplementedError("strategies must implement prices_to_features")
    
    def predictions_to_signals(
        self,
        predictions: Union[pd.DataFrame, pd.Series],
        prices: pd.DataFrame
        ) -> pd.DataFrame:
        """
        From a DataFrame of predictions produced by a machine learning model,
        return a DataFrame of signals. By convention, signals should be
        1=long, 0=cash, -1=short.

        The index of predictions will match the index of the features
        DataFrames or Series returned in prices_to_features.

        Must be implemented by strategy subclasses.

        Parameters
        ----------
        predictions : DataFrame or Series, required
            DataFrame of machine learning predictions

        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        DataFrame
            signals

        Examples
        --------
        Buy when prediction (a DataFrame) is above zero::

            def predictions_to_signals(self, predictions: pd.DataFrame, prices: pd.DataFrame):
                signals = predictions > 0
                return signals.astype(int)

        Buy a single security when the predictions (a Series) is above zero::

            def predictions_to_signals(self, predictions: pd.Series, prices: pd.DataFrame):
                closes = prices.loc["Close"]
                signals = pd.DataFrame(False, index=closes.index, columns=closes.columns)
                signals[12345] = predictions > 0
                return signals.astype(int)
        """
        raise NotImplementedError("strategies must implement predictions_to_signals")

    def check_is_fitted(self):
        """
        Checks if the model is fitted before making predictions.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded or initialized.")

        # Use scikit-learn's utility if available
        if self.config.framework == MLFramework.SCIKIT_LEARN:
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.model)
            except ImportError:
                self.log.warning("Could not import `check_is_fitted` from scikit-learn. Falling back to attribute check.")
                # Fallback for older versions or different environments
                if not hasattr(self.model, "classes_") and not hasattr(self.model, "coef_"):
                    raise RuntimeError("Model is not fitted. Call `train` before `predict`.")
            except Exception as e:
                 raise RuntimeError(f"Model is not fitted: {e}")
        # TODO: Add checks for other frameworks (e.g., Keras/TF)

    def train(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Train the machine learning model.

        This method serves as a universal trainer for the loaded model. It calls
        the appropriate fitting method based on the model type:
        - For supervised models (e.g., classification, regression), it calls `model.fit(X, y)`.
        - For unsupervised models, it calls `model.fit(X)`.

        For Reinforcement Learning models, this method should be overridden in the
        subclass to implement the specific training loop required by the agent and
        its environment.

        Parameters
        ----------
        X : pd.DataFrame
            The feature data for training. The shape should be (n_samples, n_features).
        y : pd.Series, optional
            The target labels for supervised learning. This should be omitted for
            unsupervised models. Default is None.
        """
        if self.config.model_type.learning_type == MLLearningType.REINFORCEMENT:
            raise NotImplementedError("RL training loop must be implemented in the `train` method of the subclass.")
        
        if self.model is None:
            self.log.error("Cannot train because no model is loaded or initialized.")
            return

        try:
            # Fit the scaler and transform X
            if self.scaler is not None:
                self.log.info("Fitting scaler...")
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X.values  # fallback to raw NumPy array if no scaler

            # Fit model depending on learning type
            if y is not None:
                if y is not None and not y.index.equals(X.index):
                    raise ValueError("Feature and target indices do not match.")
                self.log.info("Fitting supervised model...")
                self.model.fit(X_scaled, y)
            else:
                self.log.info("Fitting unsupervised model...")
                self.model.fit(X_scaled)
        except (ValueError, TypeError) as e:
            self.log.error(f"Model training failed due to bad input: {e}")
        except Exception as e:
            self.log.error(f"Unexpected error during training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions or transformations from feature data.

        This method acts as a smart dispatcher for inference. It automatically
        calls the most appropriate method on the underlying model:
        - If the model has a `predict_proba` method (common in classifiers),
          it will be used to get probability scores.
        - If the model is unsupervised and has a `transform` method, it will be
          called to get the transformed data representation.
        - Otherwise, it falls back to the standard `predict` method.

        Parameters
        ----------
        X : pd.DataFrame
            The feature data to generate predictions from.

        Returns
        -------
        pd.Series or np.ndarray
            An array of predictions, probabilities, or transformations from the model.
        """
        if self.model is None:
            raise ValueError("Cannot predict because no model is loaded.")

        # For unsupervised models, `predict` often means `transform`
        if self.config.model_type.learning_type == MLLearningType.UNSUPERVISED and hasattr(self.model, 'transform'):
            return self.model.transform(X)
        
        # For classification, it's often better to get probabilities
        if self.config.model_type.task_type == MLTaskType.CLASSIFICATION and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)

        return self.model.predict(X)

    def act(self, instrument_id: InstrumentId, signals: pd.DataFrame) -> None:
        """
        Execute trading logic based on the generated signals.

        This is an **abstract method** that must be implemented by the strategy
        subclass. It is the final step in the inference pipeline, receiving the
        output from `predictions_to_signals`.

        Your implementation should contain the logic for interpreting the signals
        and translating them into trading actions, such as submitting, modifying,
        or canceling orders.

        Parameters
        ----------
        instrument_id : InstrumentId
        
        signals : pd.DataFrame
            A DataFrame of trading signals, where rows are timestamps and columns
            are instruments. The values typically represent desired positions
            (e.g., 1 for long, -1 for short, 0 for flat).

        Example
        -------
        ```python
        def act(self, signals: pd.DataFrame) -> None:
            latest_signals = signals.iloc[-1]
            for instrument_id, signal in latest_signals.items():
                if signal > 0:
                    self.log.info(f"Placing LONG order for {instrument_id}")
                    # self.place_order(...)
                elif signal < 0:
                    self.log.info(f"Placing SHORT order for {instrument_id}")
                    # self.place_order(...)
        ```
        """
        raise NotImplementedError("strategies must implement the `act` method.")

    def reset_env(self, *args, **kwargs) -> None:
        """
        Reset the environment state, primarily for Reinforcement Learning.

        This method is called automatically by `on_reset`. Its purpose is to
        reset the state of an RL agent or its trading environment at the end of
        an episode.

        For non-RL strategies, this method does nothing. For RL strategies,
        you can override it to include custom reset logic.
        """
        # Add RL-specific environment reset logic here, e.g., self.environment.reset()
        pass

