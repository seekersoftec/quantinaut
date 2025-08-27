from decimal import Decimal
import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.preprocessing import StandardScaler
from nautilus_trader.common.enums import LogColor
from nautilus_trader.model.enums import OrderSide, TimeInForce, OrderType, TrailingOffsetType
from nautilus_trader.model.identifiers import ClientId, InstrumentId
from quantinaut.common.enums import MLTaskType
from data.toolkit.old_ml_strategy import MLStrategy, MLStrategyConfig


class LightGBMStrategyConfig(MLStrategyConfig, frozen=True):
    client_id: ClientId = ClientId("LGB-001")
    model_params: dict = {}
    train_params: dict = {}
    trade_size: Decimal = Decimal("0.001")


class LightGBMStrategy(MLStrategy):
    def __init__(self, config: LightGBMStrategyConfig):
        super().__init__(config)
        self.is_regression = config.model_type.task_type == MLTaskType.REGRESSION
        self.model = None
        self.scaler = StandardScaler() if self.config.scale_data else None
        self.model_params = config.model_params
        self.train_params = config.train_params

    def prices_to_features(self, prices: pd.DataFrame) -> tuple:
        closes = prices.xs('close', level='Field', axis=1)
        features_df = pd.concat({
            "returns_1d": closes.pct_change(periods=1),
            "returns_5d": closes.pct_change(periods=5),
            "returns_10d": closes.pct_change(periods=10),
        }, axis=1)
        targets = closes.pct_change().shift(-1)
        if not self.is_regression:
            targets = (targets > 0).astype(int)
        targets.name = 'target'
        combined_data = pd.concat([features_df, targets], axis=1).dropna()
        aligned_targets = combined_data['target']
        aligned_features_df = combined_data.drop(columns=['target'])
        return aligned_features_df, aligned_targets

    def train(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X)
        else:
            X_train = X.values
        y_train = y.values
        params = self.model_params.copy()
        if not params:
            params = {
                "objective": "regression" if self.is_regression else "binary",
                "learning_rate": 0.05,
                "max_depth": 5,
                "num_leaves": 32,
                "min_data_in_leaf": int(0.01 * len(X)),
                "lambda_l1": 0.0,
                "lambda_l2": 0.0,
                "boosting_type": "gbdt",
                "metric": "mse" if self.is_regression else "binary_logloss",
                "is_unbalance": True,
                "verbose": -1
            }
        self.model = lgb.train(
            params,
            train_set=lgb.Dataset(X_train, label=y_train),
            num_boost_round=self.train_params.get("num_boost_round", 100)
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        original_index = X.index
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        X_nonan = pd.DataFrame(X_scaled, index=original_index).dropna()
        y_pred = self.model.predict(X_nonan.values)
        return self._postprocess(y_pred, X_nonan.index, original_index)

    def _postprocess(self, y_pred: np.ndarray, test_index: pd.Index, original_index: pd.Index) -> pd.Series:
        y_series = pd.Series(data=y_pred, index=test_index)
        df_ret = pd.DataFrame(index=original_index)
        df_ret["y_hat"] = y_series
        return df_ret["y_hat"]

    def predictions_to_signals(self, predictions: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
        if self.is_regression:
            signals = pd.Series(0, index=predictions.index)
            signals[predictions > 0] = 1
            signals[predictions < 0] = -1
        else:
            threshold = self.config.prediction_threshold
            signals = pd.Series(0, index=predictions.index)
            signals[predictions > threshold] = 1
            signals[predictions < (1 - threshold)] = -1
        if hasattr(prices, 'columns') and hasattr(prices.columns, 'names') and 'instrument_id' in prices.columns.names:
            instrument_id = prices.columns.get_level_values('instrument_id')[0]
        else:
            instrument_id = prices.columns[0] if isinstance(prices.columns[0], str) else str(prices.columns[0])
        signals = signals.to_frame(name='signal')
        signals['instrument_id'] = instrument_id
        if isinstance(predictions.index, pd.DatetimeIndex):
            signals['timestamp'] = predictions.index
        else:
            signals = signals.reset_index()
        return signals

    def act(self, signals) -> None:
        if signals.empty:
            self.log.info("No signals to process.")
            return
        latest_row = signals.sort_values('timestamp').iloc[-1] if 'timestamp' in signals.columns else signals.iloc[-1]
        signal = latest_row['signal']
        instrument_id_str = latest_row['instrument_id']
        timestamp = latest_row.get('timestamp', None)
        instrument_id = instrument_id_str if isinstance(instrument_id_str, InstrumentId) else InstrumentId.from_str(instrument_id_str)
        self.log.info(f"Processing signal | Time: {timestamp}, Instrument: {instrument_id}, Signal: {signal}")
        instrument = self.cache.instrument(instrument_id)
        latest_close = getattr(getattr(instrument, 'market_data', None), 'close', None)
        if latest_close is None:
            self.log.warning(f"No market data for {instrument_id}, skipping order.")
            return
        if signal > 0:
            self._submit_long_bracket_order(instrument, latest_close)
        elif signal < 0:
            self._submit_short_bracket_order(instrument, latest_close)
        else:
            self.log.info(f"Generated FLAT signal for {instrument_id}. No action taken.")

    def _submit_long_bracket_order(self, instrument, entry_price):
        sl_offset = Decimal("0.01") * Decimal(entry_price)
        tp_offset = Decimal("0.02") * Decimal(entry_price)
        sl_price = Decimal(entry_price) - sl_offset
        tp_price = Decimal(entry_price) + tp_offset
        self.log.info(f"Generated LONG signal for {instrument.id}.")
        orders = self.order_factory.bracket(
            instrument_id=instrument.id,
            order_side=OrderSide.BUY,
            quantity=instrument.make_qty(self.config.trade_size),
            entry_price=instrument.make_price(entry_price),
            sl_trigger_price=instrument.make_price(sl_price),
            tp_order_type=OrderType.LIMIT,
            tp_price=instrument.make_price(tp_price),
            tp_time_in_force=TimeInForce.GTC,
            tp_activation_price=instrument.make_price(tp_price),
            tp_trailing_offset=Decimal("0.01") * Decimal(entry_price),
            tp_trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
            time_in_force=TimeInForce.GTC,
            sl_time_in_force=TimeInForce.GTC,
        )
        self.submit_order_list(orders)
        self.log.info(
            f"Submitted LONG bracket order for {instrument.id} | Quantity: {self.config.trade_size:.8f}",
            color=LogColor.GREEN,
        )

    def _submit_short_bracket_order(self, instrument, entry_price):
        sl_offset = Decimal("0.01") * Decimal(entry_price)
        tp_offset = Decimal("0.02") * Decimal(entry_price)
        sl_price = Decimal(entry_price) + sl_offset
        tp_price = Decimal(entry_price) - tp_offset
        self.log.info(f"Generated SHORT signal for {instrument.id}.")
        orders = self.order_factory.bracket(
            instrument_id=instrument.id,
            order_side=OrderSide.SELL,
            quantity=instrument.make_qty(self.config.trade_size),
            entry_price=instrument.make_price(entry_price),
            sl_trigger_price=instrument.make_price(sl_price),
            tp_order_type=OrderType.LIMIT,
            tp_price=instrument.make_price(tp_price),
            tp_time_in_force=TimeInForce.GTC,
            tp_activation_price=instrument.make_price(tp_price),
            tp_trailing_offset=Decimal("0.01") * Decimal(entry_price),
            tp_trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
            time_in_force=TimeInForce.GTC,
            sl_time_in_force=TimeInForce.GTC,
        )
        self.submit_order_list(orders)
        self.log.info(
            f"Submitted SHORT bracket order for {instrument.id} | Quantity: {self.config.trade_size:.8f}",
            color=LogColor.GREEN,
        )

    def _save_model(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save_model(str(path / "model.txt"))
        if self.scaler:
            joblib.dump(self.scaler, path / "scaler.pkl")

    def _load_model(self, path: Union[str, Path]) -> None:
        path = Path(path)
        self.model = lgb.Booster(model_file=str(path / "model.txt"))
        scaler_file = path / "scaler.pkl"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)