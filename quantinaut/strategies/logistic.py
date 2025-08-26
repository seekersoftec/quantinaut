from decimal import Decimal
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from nautilus_trader.common.enums import LogColor
from nautilus_trader.model.enums import OrderSide, TimeInForce, TrailingOffsetType, OrderType
from nautilus_trader.model.identifiers import ClientId, InstrumentId

from quantinaut.common.enums import MLTaskType
from quantinaut.strategies.ml_strategy import MLStrategy, MLStrategyConfig



class LogisticStrategyConfig(MLStrategyConfig, frozen=True):
    """
    Configuration for the LogisticStrategy.
    """
    client_id: ClientId = ClientId("LR-001")
    model_params: dict = {}
    trade_size: Decimal = Decimal("0.01")

class LogisticStrategy(MLStrategy):
    """
    A simple momentum-based strategy using a Logistic Regression model.

    This strategy generates features from historical returns and uses a
    logistic regression model to predict whether the next bar's return will
    be positive.
    """
    def __init__(self, config: LogisticStrategyConfig):
        super().__init__(config)
        self.is_regression = config.model_type.task_type == MLTaskType.REGRESSION
        self.model = LogisticRegression(**config.model_params)
        if self.config.scale_data:
            self.scaler = StandardScaler()
    
    def prices_to_features(self, prices: pd.DataFrame) -> tuple:
        """
        Generates momentum-based features from historical price data.

        Features:
        - 1-bar return
        - 5-bar return
        - 10-bar return

        Target:
        - 1 if the next bar's close is higher than the current close, 0 otherwise.
        """
        highs = prices.xs('high', level='Field', axis=1)
        lows = prices.xs('low', level='Field', axis=1)
        closes = prices.xs('close', level='Field', axis=1)
        prices = (highs + lows + closes) / 3.0
        self.closes = closes  # Store closes for later use in act method
   
        returns_1d = prices.pct_change(periods=1)
        returns_5d = prices.pct_change(periods=5)
        returns_10d = prices.pct_change(periods=10)

        sma_5 = closes.rolling(window=5).mean()
        sma_10 = closes.rolling(window=10).mean()
        sma_cross = (sma_5 > sma_10).astype(int)
        
        volatility_5d = closes.rolling(window=5).std()
        momentum = closes - closes.shift(5)
        rsi = 100 - 100 / (1 + closes.pct_change().rolling(14).mean() / closes.pct_change().rolling(14).std())
        
        features_df = pd.concat({
            "returns_1d": returns_1d,
            "returns_5d": returns_5d,
            "returns_10d": returns_10d,
            "sma_5": sma_5,
            "sma_10": sma_10,
            "sma_cross_signal": sma_cross,
            "volatility_5d": volatility_5d,
            "momentum": momentum,
            "rsi_14": rsi,
        }, axis=1)

        # Create target series
        targets = (closes.pct_change().shift(-1) > 0).astype(int)
        # Ensure targets is a Series with a name
        if isinstance(targets, pd.DataFrame):
            targets = targets.iloc[:, 0]
        targets.name = 'target'

        # Combine all data into a single DataFrame to handle NaNs and alignment
        combined = pd.concat([features_df, targets], axis=1).dropna()
        return combined.drop(columns=['target']), combined['target']

    def predictions_to_signals(self, predictions: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Converts model predictions (probabilities) into trading signals.

        A signal of 1 (long) is generated if the predicted probability of a
        positive return is greater than the configured threshold.
        A signal of -1 (short) is generated if the probability is less than (1 - threshold).
        Otherwise, the signal is 0 (flat).

        Returns:
            pd.DataFrame with columns: ['timestamp', 'signal', 'instrument_id']
        """
        threshold = self.config.prediction_threshold
        signals = pd.Series(0, index=predictions.index, name="signal")
        signals[predictions > threshold] = 1
        signals[predictions < (1 - threshold)] = -1

        # Convert to DataFrame
        signals_df = signals.to_frame()
        signals_df["close"] = prices.xs('close', level='Field', axis=1) # .iloc[-1]  # Get the latest close price
                    
        # Determine instrument_id and ensure it's included
        if isinstance(predictions.index, pd.MultiIndex) and "instrument_id" in predictions.index.names:
            # Multi-instrument: reset index to expose instrument_id
            signals_df = signals_df.reset_index()
        else:
            # Single-instrument: extract from prices
            if hasattr(prices, 'columns') and hasattr(prices.columns, 'names') and 'instrument_id' in prices.columns.names:
                instrument_id = prices.columns.get_level_values('instrument_id')[0]
            else:
                instrument_id = prices.columns[0] if isinstance(prices.columns[0], str) else str(prices.columns[0])
            signals_df["instrument_id"] = instrument_id
            if isinstance(predictions.index, pd.DatetimeIndex):
                signals_df["timestamp"] = predictions.index
            else:
                signals_df = signals_df.reset_index()

        return signals_df

    def act(self, instrument_id, signals) -> None:
        """
        Executes trading logic based on the generated signals DataFrame.
        Expects columns: ['timestamp', 'signal', 'instrument_id']
        """
        # volatility = self.closes.pct_change().rolling(5).std().iloc[-1]
        # if volatility.mean() < 0.01:
        #     self.log.info("Market too quiet, skipping.")
        #     return

        if signals.empty:
            self.log.info("No signals to process.")
            return

        # Get the latest signal row (by timestamp if present)
        latest_row = signals.sort_values('timestamp').iloc[-1] if 'timestamp' in signals.columns else signals.iloc[-1]

        signal = latest_row['signal']
        timestamp = latest_row.get('timestamp', None)
        latest_close = latest_row.get('close', None)

        # Convert to InstrumentId if necessary
        self.log.info(f"Processing signal | Time: {timestamp}, Instrument: {instrument_id}, Signal: {signal}")
        instrument = self.cache.instrument(instrument_id)
  
        if latest_close is None:
            self.log.warning(f"No market data for {instrument_id}, skipping order.")
            return

        # if self.cache.has_position(instrument_id):
        #     self.log.info(f"Already in position on {instrument_id}, skipping signal.")
        #     return

        if signal > 0:
            self._submit_order(instrument, latest_close, OrderSide.BUY)
        elif signal < 0:
            self._submit_order(instrument, latest_close, OrderSide.SELL)
        else:
            self.log.info(f"Generated FLAT signal for {instrument_id}. No action taken.")

    def _submit_order(self, instrument, price, side):
        atr = self.closes.rolling(14).apply(lambda x: np.max(x) - np.min(x)).iloc[-1].values[0]
        sl_offset = Decimal(atr) * Decimal("1.5")
        tp_offset = sl_offset * Decimal("2.5")

        if side == OrderSide.BUY:
            sl_price = Decimal(price) - sl_offset
            tp_price = Decimal(price) + tp_offset
        else:
            sl_price = Decimal(price) + sl_offset
            tp_price = Decimal(price) - tp_offset

        orders = self.order_factory.bracket(
            instrument_id=instrument.id,
            order_side=side,
            quantity=instrument.make_qty(self.config.trade_size),
            entry_price=instrument.make_price(price),
            sl_trigger_price=instrument.make_price(sl_price),
            tp_order_type=OrderType.TRAILING_STOP_MARKET,
            tp_price=instrument.make_price(tp_price),
            tp_time_in_force=TimeInForce.GTC,
            tp_activation_price=instrument.make_price(tp_price),
            tp_trailing_offset=Decimal("300"),
            tp_trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
            time_in_force=TimeInForce.GTC,
            sl_time_in_force=TimeInForce.GTC,
        )
        self.submit_order_list(orders)
        self.log.info(
            f"Submitted {side.name} order on {instrument.id} | Entry: {price}",
            color=LogColor.GREEN,
        )
