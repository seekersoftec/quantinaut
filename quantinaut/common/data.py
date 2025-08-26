from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import logging
import re
import polars as pl
from typing import Any, Dict, List, Literal, Optional, Union
from nautilus_trader.core.data import Data
from nautilus_trader.model import (
    Bar,
    BarType,
    BarSpecification,
)
from nautilus_trader.model.identifiers import ClientId, InstrumentId
from nautilus_trader.model.enums import ( 
    OrderSide, 
    TimeInForce, 
    BarAggregation,
    AggregationSource,
    PriceType,
)
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderType
from sklearn.base import ClassifierMixin, RegressorMixin

from nautilus_ai.common.enums import TradingDecision, MLLearningType, MLTaskType

class TimeframeSubscription(Data):
    """
    Represents a subscription to specific timeframes for analysis and entry.

    Parameters:
    ----------
    kwargs : dict
    timestamp : int
        The Unix timestamp indicating when the subscription was created or last updated.
    
    Attributes:
    ----------
    entry : Optional[Union[BarType, str]]
        The primary timeframe for trade entry decisions. Can be a BarType object or a string
        representing the timeframe (e.g., "5m" for 5 minutes).
    analysis : Dict[str, List[Union[BarType, str]]]
        A dictionary where keys are descriptive names (e.g., "trend", "momentum") and values
        are lists of BarType objects or timeframe strings used for analysis.        
 
    """

    def __init__(
        self,
        entry: Optional[Union[BarType, str]] = None,
        timestamp: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.log = logging.getLogger(self.__class__.__name__)
        self.entry: Optional[Union[BarType, str]] = None
        self.analysis: Dict[str, List[Union[BarType, str]]] = {}
        self.timestamp: int = timestamp if timestamp is not None else int(datetime.now().timestamp())

        # Handle entry
        if entry is not None:
            self._validate_and_set_timeframe("entry", entry, set_as="entry")

        # Handle analysis kwargs
        for key, value in kwargs.items():
            if key.endswith("_analysis"):
                if isinstance(value, list):
                    raise TypeError(f"{key} must be a single BarType or str, not a list")
                base_key = key.removesuffix("_analysis")
                self._validate_and_set_timeframe(key, value, set_as="analysis", analysis_key=base_key)
            else:
                setattr(self, key, value)

    @property
    def spans(self) -> Dict[str, BarAggregation]:
        return {
            "ms": BarAggregation.MILLISECOND,
            "s": BarAggregation.SECOND,
            "m": BarAggregation.MINUTE,
            "h": BarAggregation.HOUR,
            "D": BarAggregation.DAY,
            "W": BarAggregation.WEEK,
            "M": BarAggregation.MONTH,
            "tick": BarAggregation.TICK,
            "tick_imbalance": BarAggregation.TICK_IMBALANCE,
            "tick_runs": BarAggregation.TICK_RUNS,
            "volume": BarAggregation.VOLUME,
            "volume_imbalance": BarAggregation.VOLUME_IMBALANCE,
            "volume_runs": BarAggregation.VOLUME_RUNS,
            "value": BarAggregation.VALUE,
            "value_imbalance": BarAggregation.VALUE_IMBALANCE,
            "value_runs": BarAggregation.VALUE_RUNS,
        }

    def _validate_and_set_timeframe(
        self,
        source: str,
        value: Union[str, BarType],
        set_as: str,
        analysis_key: Optional[str] = None
    ) -> None:
        if isinstance(value, str):
            # value = value.lower()
            self._validate_timeframe_unit(value, source)
        elif not isinstance(value, BarType):
            raise TypeError(f"{source} must be a BarType or str, got {type(value)}")

        if set_as == "entry":
            self.entry = value
        elif set_as == "analysis" and analysis_key:
            self.analysis[analysis_key] = [value]
        else:
            raise ValueError(f"Invalid assignment target: {set_as}")

    def _validate_timeframe_unit(self, tf: str, source: str) -> None:
        unit = ''.join(filter(str.isalpha, tf))
        if unit not in self.spans:
            raise ValueError(f"Invalid timeframe unit '{unit}' in {source}: '{tf}'")

    def convert_timeframe(self, timeframe: str = "5m") -> tuple[int, BarAggregation]:
        match = re.match(r"(\d+)([a-zA-Z_]+)", timeframe)
        step, unit = (1, timeframe) if not match else (int(match.group(1)), match.group(2))

        if unit not in self.spans:
            raise ValueError(f"Unsupported timeframe unit '{unit}' in '{timeframe}'")
        return step, self.spans[unit]

    def validate_timeframes(self) -> bool:
        def extract_unit(tf: str) -> str:
            return ''.join(filter(str.isalpha, tf))

        if isinstance(self.entry, str):
            if extract_unit(self.entry) not in self.spans:
                self.log.error(f"Invalid entry timeframe: {self.entry}")
                return False

        for key, values in self.analysis.items():
            for tf in values:
                if isinstance(tf, str) and extract_unit(tf) not in self.spans:
                    self.log.error(f"Invalid analysis timeframe '{tf}' under key '{key}'")
                    return False
        return True

    def get_all_timeframes(
        self,
        as_type: type[Union[BarType, str]],
        instrument_id: Optional[InstrumentId] = None,
        price_type: PriceType = PriceType.LAST,
        aggregation_source: AggregationSource = AggregationSource.EXTERNAL,
    ) -> list[Union[BarType, str]]:
        """
        Assumptions:
        - Price type is Last 
        - A
        """
        if not self.validate_timeframes():
            raise ValueError("Some timeframes are invalid.")

        raw_timeframes = [self.entry] if self.entry else []
        for lst in self.analysis.values():
            raw_timeframes.extend(lst)

        if as_type not in {BarType, str}:
            raise ValueError(f"Unsupported as_type: {as_type}")
        if instrument_id is None:
            raise ValueError("instrument_id is required to convert to the specified type")

        return [self._convert_to_type(tf, as_type, instrument_id, price_type, aggregation_source) for tf in raw_timeframes]

    def _convert_to_type(
        self,
        tf: Union[BarType, str],
        as_type: type[Union[BarType, str]],
        instrument_id: InstrumentId,
        price_type: PriceType = PriceType.LAST,
        aggregation_source: AggregationSource = AggregationSource.EXTERNAL,
    ) -> Union[BarType, str]:
        if isinstance(tf, as_type):
            return tf

        step, agg = self.convert_timeframe(tf)
        bar_type = self.get_bar_type(instrument_id, step, agg, price_type, aggregation_source)

        return bar_type if as_type is BarType else str(bar_type)

    def set_entry(self, entry: Union[BarType, str]) -> None:
        self._validate_and_set_timeframe("entry", entry, set_as="entry")

    def add(self, key: str, bar_type: Union[BarType, str]) -> None:
        self._validate_and_set_timeframe(f"{key} (add)", bar_type, set_as="analysis", analysis_key=key)

    def remove(
        self,
        key: str,
        bar_type: Optional[Union[BarType, str]] = None,
    ) -> None:
        if key not in self.analysis:
            self.log.warning(f"Key '{key}' not found in analysis subscriptions.")
            return
        if bar_type is None:
            del self.analysis[key]
        else:
            try:
                self.analysis[key].remove(bar_type)
                if not self.analysis[key]:
                    del self.analysis[key]
            except ValueError:
                self.log.warning(f"Bar type '{bar_type}' not found under key '{key}'.")

    def sub_timeframes(self, func, indicator):
        inst = getattr(indicator, 'instrument_id', None)
        for bt in self.get_all_timeframes(BarType, inst):
            func(bt, indicator)

    @staticmethod
    def get_bar_type(
        instrument_id: InstrumentId,
        step: int = 1,
        aggregation: BarAggregation = BarAggregation.MINUTE,
        price_type: PriceType = PriceType.LAST,
        aggregation_source: AggregationSource = AggregationSource.EXTERNAL,
    ) -> BarType:
        return BarType(
            instrument_id=instrument_id,
            bar_spec=BarSpecification(
                step=max(step, 1),
                aggregation=aggregation,
                price_type=price_type,
            ),
            aggregation_source=aggregation_source,
        )


class TradeSignal(Data):
    """
    Represents a trading signal with metadata for order generation and execution.

    Attributes:
    ----------
    entry : Bar
        The entry bar used in entry price calculation.
    stop_loss : Optional[Bar]
        The stop loss bar used in stop loss calculation.
    take_profit : Optional[Bar]
        The take profit bar used in take profit calculation.
    order_side : OrderSide
        The direction of the trade (BUY, SELL, or NO_ORDER_SIDE).
    confidence : float
        Signal confidence level between 0.0 and 1.0.
    order_type : OrderType
        Type of entry order (e.g., MARKET_IF_TOUCHED).
    time_in_force : TimeInForce
        Time-in-force policy (e.g., GTC, GTD, IOC).
    timestamp : int
        Timestamp when the signal was generated (UNIX).
    Additional attributes:
        Any keyword arguments passed to the constructor will be set as attributes.
    """

    def __init__(
        self,
        entry: Bar,
        action: TradingDecision = TradingDecision.NEUTRAL,
        confidence: float = 0.0,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        order_side: OrderSide = OrderSide.NO_ORDER_SIDE,
        entry_order_type: OrderType = OrderType.MARKET_IF_TOUCHED,
        tp_order_type: OrderType = OrderType.TRAILING_STOP_MARKET,
        time_in_force: TimeInForce = TimeInForce.GTC,
        use_auto_sl: bool = False,
        use_trailing_stop: bool = False,
        use_bracket_order: bool = False,
        timestamp: Optional[int] = None,
        client_id: ClientId = ClientId("SIG-001"),
        **kwargs,
    ) -> None:
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())

        super().__init__()

        self.client_id = client_id
        self.instrument_id = entry.bar_type.instrument_id
        self.entry = entry
        self.action = action
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_side = order_side
        self.confidence = confidence
        self.entry_order_type = entry_order_type
        self.tp_order_type = tp_order_type
        self.time_in_force = time_in_force
        self.use_auto_sl = use_auto_sl
        self.use_trailing_stop = use_trailing_stop
        self.use_bracket_order = use_bracket_order
        self.timestamp = timestamp

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            f"<TradeSignal client_id={self.client_id} instrument_id={self.instrument_id} "
            f"entry={self.entry} stop_loss={self.stop_loss} take_profit={self.take_profit} "
            f"order_side={self.order_side} confidence={self.confidence:.2f} "
            f"entry_order_type={self.entry_order_type} tp_order_type={self.tp_order_type} use_auto_sl={self.use_auto_sl} "
            f"time_in_force={self.time_in_force} timestamp={self.timestamp}>"
        )

    def is_valid(self, min_confidence: float = 0.65) -> bool:
        """
        Check whether the trade signal meets a minimum confidence threshold.

        Parameters:
            min_confidence (float): Minimum required confidence level.

        Returns:
            bool: True if signal confidence is sufficient, else False.
        """
        return self.confidence >= min_confidence
    

class GeneratorData(Data):
    """
    A data class for communication to/from generator-based components.

    This class provides a standardized container for data, including a Polars DataFrame,
    to be passed between different stages of a data processing pipeline. It ensures a
    consistent and type-safe interface for all generator components.
    """
    def __init__(
        self,
        generator_id: str,
        df: Optional[pl.DataFrame] = None,
        new_columns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Any = None,
        timestamp: Optional[int] = None,
        kwargs: Dict[str, Any] = {},
    ):
        if df is None:
            df = pl.DataFrame()
        if new_columns is None:
            new_columns = []
        if metadata is None:
            metadata = {}
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())

        super().__init__()
        self.generator_id = generator_id
        self.df = df
        self.new_columns = new_columns
        self.metadata = metadata
        self.model = model
        self.timestamp = timestamp
        self.kwargs = kwargs



class ModelPrediction(Data):
    """
    Represents a prediction generated by a model.

    Attributes:
    ----------
    instrument_id : str
        The identifier of the instrument for which the prediction is made.
    prediction : float
        The predicted value.
    confidence : float
        The confidence level of the prediction.
    timestamp : int
        The Unix timestamp when the prediction was generated.
    """
    def __init__(
        self,
        instrument_id: str,
        prediction: float,
        confidence: float,
        timestamp: Optional[int] = None,
    ):
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
            
        super().__init__(ts_init=timestamp, ts_event=timestamp)
        self.instrument_id = instrument_id
        self.prediction = prediction
        self.confidence = confidence


class MLModelType(Data):
    """
    Represents the characteristics of a machine learning model in terms of its
    learning paradigm and task type.

    Attributes
    ----------
    learning_type : MLLearningType
        The learning paradigm of the model. For example, supervised, unsupervised,
        or reinforcement learning.

    task_type : Optional[MLTaskType]
        The specific task the model is designed for, such as classification,
        regression, clustering, or dimensionality reduction. This may be None for
        models like reinforcement learning where a traditional task type may not apply.

    Example
    -------
    >>> MLModelType(
    ...     learning_type=MLLearningType.SUPERVISED,
    ...     task_type=MLTaskType.CLASSIFICATION
    ... )

    Notes
    -----
    - Separating learning type from task type enables more precise model introspection
      and control in ML pipelines.
    - Task type is optional to accommodate RL models or general-purpose learners.
    """
    def __init__(
        self,
        learning_type: MLLearningType = MLLearningType.SUPERVISED,
        task_type: Optional[MLTaskType] = MLTaskType.CLASSIFICATION  # Task might be None for RL
    ) -> None:
        super().__init__()

        if learning_type != MLLearningType.REINFORCEMENT:
            if task_type is None:
                raise ValueError("task_type must be specified for non-reinforcement learning models.")
        else:
            if task_type is not None:
                raise ValueError("task_type should be None for reinforcement learning models.")

        self.learning_type = learning_type
        self.task_type = task_type

    def __repr__(self) -> str:
        return (
            f"MLModelType(learning_type={self.learning_type.name}, "
            f"task_type={self.task_type.name if self.task_type else 'None'})"
        )


    
# class ModelUpdate(Data):
#     """
#     Represents an update to a predictive model.

#     Attributes:
#     ----------
#     model : ClassifierMixin, RegressorMixin
#         The predictive model being updated.
#     hedge_ratio : float
#         The hedge ratio calculated by the model.
#     std_prediction : float
#         The standard deviation of the model's predictions.
#     timestamp : int
#         The Unix timestamp when the model update occurred.
#     """
#     def __init__(
#         self,
#         model: Any,
#         hedge_ratio: float,
#         std_prediction: float,
#         timestamp: Optional[int] = None,
#     ):
#         if timestamp is None:
#             timestamp = int(datetime.now().timestamp())

#         super().__init__()
#         self.model = model
#         self.hedge_ratio = hedge_ratio
#         self.std_prediction = std_prediction
#         self.timestamp = timestamp
