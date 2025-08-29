from dataclasses import field
from datetime import timedelta
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Any, List, Optional, Tuple, TypedDict

from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.config import StrategyConfig
from nautilus_trader.config import PositiveInt, PositiveFloat
from nautilus_trader.common.enums import LogColor
from nautilus_trader.model.data import Bar, DataType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.enums import OrderSide, OrderType, PositionSide, TriggerType, TimeInForce, TrailingOffsetType, ContingencyType
from nautilus_trader.model.events import OrderFilled, PositionChanged, PositionOpened, PositionClosed
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders.base import Order
from nautilus_trader.model.orders.list import OrderList
from nautilus_trader.model.orders import MarketOrder, TrailingStopMarketOrder
from nautilus_trader.trading.strategy import Strategy

from quantinaut.common import TradeSignal, TradingDecision
from quantinaut.portfolio.risk_models import BaseRiskModel, RiskModelFactory


# Set the precision high enough to handle the quantization
getcontext().prec = 16  # or higher if needed

class ModelInitArgs(TypedDict, total=False):
    risk_pct: float = 0.02 # risk 1% of equity per trade

class AdaptiveRiskEngineConfig(StrategyConfig, frozen=True):
    """
    Configuration for the Adaptive Risk Engine.

    This configuration class specifies the parameters used by the risk engine
    to adapt trading strategies based on market conditions, including signal confidence,
    trailing stop behavior, risk management, and market regime filtering.

    Parameters
    ----------
        model_name (str): Name of the risk model (e.g., 'fixed_fractional').
        model_init_args (Dict): Initialization arguments for the risk model.
            For example, {'risk_pct': 0.01} to risk 1% of equity per trade.
        
        Configuration for trailing stop settings:

        offset_bps_scale (Decimal): Scaling factor for trailing stop offset in basis points.
        trigger_type (TriggerType): Type of price trigger used to activate trailing stops (e.g., LAST_PRICE).
        
        Configuration for risk management parameters:
        
        min_rr (float): Minimum acceptable risk-to-reward ratio for trade execution.
        top_k_quantiles (int): Number of top quantiles to consider in risk management analysis.
        rvol_threshold (float): Threshold for relative volume to filter trades.
            For example, 0.5 means only consider trades with RVOL > 0.5.
        
        min_confidence_threshold : PositiveFloat
        Minimum confidence level required for trade signals to be considered.

    """
    model_name: str = "fixed_fractional"
    model_init_args: ModelInitArgs = field(default_factory=lambda: {"risk_pct": 0.02}) # field(default_factory=lambda: dict(risk_pct=0.02))  # risk 1% of equity per trade
    min_rr: PositiveFloat = 1            # Minimum risk-to-reward ratio, e.g., 1:3 R:R
    top_k_quantiles: PositiveInt = 4
    rvol_threshold: float = 0.5 # 1.5
    # Confidence threshold for trade signals
    min_confidence_threshold: PositiveFloat = 0.55
    
    bracket_distance_atr: PositiveFloat = 3.0
    trailing_atr_multiple: PositiveFloat = 3.0
    trailing_offset_type: str = "BASIS_POINTS"
    trailing_offset_bps_scale: Decimal = Decimal(100)
    trigger_type: str = "NO_TRIGGER"
    max_trade_sizes: Decimal = Decimal("0.01")
    emulation_trigger: str = "NO_TRIGGER"


class AdaptiveRiskEngine(Strategy):
    def __init__(self, config: AdaptiveRiskEngineConfig) -> None:
        super().__init__(config)
        
        # Users order management variables
        self.instrument = None
        self.entry = None
        self.use_trailing_stop = False
        self.trailing_stop = None
        self.position_id = None
        # self.trade_signal = None
        
        # risk
        self.tp_levels = [(0.5, 1.5), (0.5, 3.0)]   # e.g., [(0.5, 1.5), (0.5, 3.0)]  meaning: 50% at 1.5R, 50% at 3.0R
        self.max_trade_sizes: dict[str, Decimal] = {
            "Any": Decimal("0.01")
        }
        self.max_trade_sizes.update(config.max_trade_sizes)
        self.risk_models: dict[InstrumentId, BaseRiskModel] = {}
        self.atr_values: dict[InstrumentId, float] = {}

    def on_start(self) -> None:
        """
        Initializes the trading engine on strategy start.

        This method:
        - Registers all indicators (ATR, VVR filter, etc.).
        - Subscribes to trade signal data.
        - Logs the engine's readiness.

        This method is automatically invoked by the framework during startup.
        """
        self.subscribe_data(data_type=DataType(TradeSignal))
        self.log.info("Risk engine started and subscribed to trade signals.")
    
    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        self.unsubscribe_data(data_type=DataType(TradeSignal))
        self.log.info("Risk engine state has been stopped.")
            
    def on_reset(self) -> None:
        """
        Resets the trading engine state and its components.

        This method:
        - Calls the base `on_reset` method.
        - Resets the ATR indicator, VVR filter, and nodal displacement sequence.
        - Logs that the engine has been reset.

        Useful during reinitialization or when restarting a backtest session.
        """
        self.log.info("Risk engine state has been reset.")
        
    def on_data(self, data: Data) -> None:
        """
        Handles incoming data objects for the trading engine.

        Parameters
        ----------
        data : Data
            A data object, expected to be a `TradeSignal` for trading decisions.

        Behavior
        --------
        - Validates that the data is not `None`.
        - Waits for indicators to warm up before processing.
        - If the data is a `TradeSignal`, handles it accordingly.
        """
        PyCondition.not_none(data, "data")

        if isinstance(data, TradeSignal):
            self._handle_trade_signal(data)
            
    def on_event(self, event: Event) -> None:
        """
        Handles events related to order and position lifecycle.

        Parameters
        ----------
        event : Event
            The incoming event. Expected types:
            - `OrderFilled`: A trade order was filled.
            - `PositionOpened`, `PositionChanged`, `PositionClosed`: Position state changes.

        Behavior
        --------
        - Delegates to appropriate handler based on the event type.
        - Logs position updates when positions are opened, changed, or closed.
        """
        if self.use_trailing_stop:        
            if isinstance(event, OrderFilled):
                self._handle_order_filled(event)
                if self.trailing_stop and event.client_order_id == self.trailing_stop.client_order_id:
                    self.trailing_stop = None
            elif isinstance(event, PositionOpened | PositionChanged):
                if self.trailing_stop:
                    return  # Already a trailing stop
                if self.entry and event.opening_order_id == self.entry.client_order_id:
                    if event.entry == OrderSide.BUY:
                        self.position_id = event.position_id
                        self.trailing_stop_sell(event.instrument_id)
                    elif event.entry == OrderSide.SELL:
                        self.position_id = event.position_id
                        self.trailing_stop_buy(event.instrument_id)
            elif isinstance(event, PositionClosed):
                self.position_id = None
        
        if isinstance(event, (PositionOpened, PositionChanged, PositionClosed)):
            pos = self.cache.position(event.position_id)
            self.log.info(f"Position update: {pos}", color=LogColor.MAGENTA)
            
            # Get position counts and show statistics
            total = self.cache.positions_total_count()
            open_count = self.cache.positions_open_count()
            closed_count = self.cache.positions_closed_count()
            self.log.info(
                f"Position counts - Total: {total}, Open: {open_count}, Closed: {closed_count}",
            )
        
    # ——————————————————————————————————————————————————————————————
    # Event Handlers
    # ——————————————————————————————————————————————————————————————
    
    def _handle_trade_signal(self, signal: TradeSignal) -> None:
        """
        Handles an incoming `TradeSignal`.

        If the portfolio is flat for the instrument:
        - Cancels all pending standalone and bracket orders.
        - Logs the cleanup.

        Then delegates signal execution to the trade engine.

        Parameters:
        -----------
            signal: TradeSignal containing the instrument, order side, and metadata.
        """
        instr_id = signal.instrument_id
        if "XRP" in str(instr_id.symbol):
            max_trade_size = self.max_trade_sizes["XRP"]
        else:
            max_trade_size = self.max_trade_sizes["Any"]
            
        if str(instr_id) not in self.risk_models:
            self.risk_models[str(instr_id)] = RiskModelFactory.create(name=self.config.model_name, max_size=max_trade_size, **self.config.model_init_args) # e.g., an instance of a Risk model
        
        self.instrument = self.cache.instrument(instr_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {instr_id}")
            self.stop()
            return
        
        is_flat = self.portfolio.is_flat(instr_id)
        is_long = self.portfolio.is_net_long(instr_id)
        is_short = self.portfolio.is_net_short(instr_id)

        if is_flat:
            standalone = self.cache.orders(instrument_id=instr_id)
            order_lists = self.cache.order_lists(instrument_id=instr_id)
            self.log.info(
                f"Portfolio is flat for {instr_id}, {len(order_lists)} stale bracket order(s) "
                f"and {len(standalone)} stale order(s).",
                color=LogColor.YELLOW
            )
            # self.cancel_all_orders(instr_id)
            
        else:  # Not flat, manage existing position
            if is_long and signal.action == TradingDecision.EXIT_LONG or signal.action == TradingDecision.ENTER_SHORT:
                self.cancel_all_orders(instr_id, order_side=OrderSide.BUY)
                self.close_all_positions(instr_id, position_side=PositionSide.LONG)
                
            elif is_short and signal.action == TradingDecision.EXIT_SHORT or signal.action == TradingDecision.ENTER_LONG:
                self.cancel_all_orders(instr_id, order_side=OrderSide.SELL)
                self.close_all_positions(instr_id, position_side=PositionSide.SHORT)
        
        # Signal Validation
        if not signal.is_valid(self.config.min_confidence_threshold):
            self.log.info(
                f"{signal.client_id}: {signal.confidence} < {self.config.min_confidence_threshold}, signal validation failed, skipping trade.",
                color=LogColor.RED
            )
            return

        if signal.order_side not in (OrderSide.BUY, OrderSide.SELL) or signal.action == TradingDecision.NEUTRAL:
            self.log.info(
                f"{signal.client_id}: Invalid order side: {signal.order_side}, skipping trade.",
                color=LogColor.RED
            )
            return
        
        # TODO: Handle trade types here: bracket, trailing stop, etc
        # 
        # Enable or disable trailing stop based on side
        self.use_trailing_stop = signal.use_trailing_stop # True for others except bracket orders || (side == OrderSide.BUY)
        if signal.use_bracket_order:
            self.use_trailing_stop = False # Bracket orders 
            self.place_bracket_order(instr_id, signal.order_side, signal.entry, signal.stop_loss, signal.take_profit, atr=signal.atr)
        elif signal.entry_order_type == OrderType.MARKET and not signal.use_bracket_order: 
            self.atr_values[str(instr_id)] = signal.atr
            self.place_simple_order(instr_id, signal.order_side, signal.entry)
            
    def _handle_order_filled(self, event: Event) -> None:
        """
        Handles an `OrderFilled` event.

        - Activates a trailing stop if configured and conditions match.
        - Removes filled entry orders from the internal order list.

        Parameters:
        ----------
            event: OrderFilled, OrderList, or Order event containing order fill details.
        """
        self.log.info(f"Order filled: {event}", color=LogColor.GREEN)
        
        if isinstance(event, OrderFilled):
            position = self.cache.position(event.position_id)
            if position and position.is_open:
                self.log.info(f"Active Position={position}", color=LogColor.MAGENTA)

        # Get order counts directly (more efficient than len() when only count is needed)
        total = self.cache.orders_total_count()
        open_count = self.cache.orders_open_count()
        closed_count = self.cache.orders_closed_count()
        self.log.info(f"Order counts - Total: {total}, Open: {open_count}, Closed: {closed_count}")
    
    # ——————————————————————————————————————————————————————————————
    # Helpers
    # ——————————————————————————————————————————————————————————————
    def calculate_position_size(
        self,
        instrument: Instrument,
        entry_price: Decimal,
    ) -> float:
        """
        Calculates the total trade size using the configured position sizing algorithm,
        stop-loss distance, and current market price, while enforcing instrument limits.

        Parameters:
        -----------
            instrument (Instrument): The instrument being traded.
            entry_price (float): Current market price of the instrument.
            
        Returns:
        --------
            float: Total position size.

        Raises:
        -------
            ValueError: If stop-loss distance or price is zero or negative.
        """
        if entry_price <= 0:
            raise ValueError("Price must be > 0")

        # Get account and free capital
        account = self.cache.account_for_venue(instrument.id.venue)
        free_balance = Decimal(account.balance_free(instrument.quote_currency).as_decimal())

        # Compute capital to risk using position sizer
        risk_amt = self.risk_models[str(instrument.id)].get_size(capital=free_balance)
        self.log.info(f"Calculated risk amount: {risk_amt}", color=LogColor.CYAN)

        # Convert to raw position size based on SL distance
        raw_size = risk_amt / Decimal(entry_price)

        # Ensure notional value meets minimum
        notional = raw_size * Decimal(entry_price)
        if instrument.min_notional is not None:
            min_notional = Decimal(instrument.min_notional.as_decimal())
            if notional < min_notional:
                raw_size = Decimal(min_notional) / Decimal(entry_price)
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
            notional = raw_size * Decimal(entry_price)  # recalculate after adjustments
            if notional > max_notional:
                raw_size = Decimal(max_notional) / Decimal(entry_price)
                self.log.info(
                    f"Adjusted position size to meet max_notional: {raw_size}",
                    color=LogColor.YELLOW
                )

        # Quantize to instrument's size precision
        size_precision = Decimal(str(instrument.size_increment))
        total_size = float(raw_size.quantize(size_precision, rounding=ROUND_DOWN))

        self.log.info(f"Final calculated size: {total_size}", color=LogColor.GREEN)

        return total_size
    
    def place_simple_order(self, instrument_id, side: OrderSide, entry_bar: Bar) -> None:
        """
        Place a simple market order (buy or sell) with optional trailing stop setup.

        :param side: OrderSide.BUY or OrderSide.SELL
        :param entry_bar: latest Bar object to derive price
        """
        if not self.instrument:
            self.log.error("No instrument loaded")
            return

        # Retrieve the latest bar
        last_price = Decimal(entry_bar.close.as_double())

        # Calculate position sizing
        trade_size = self.calculate_position_size(self.instrument, last_price)

        # Create and submit market order
        order: MarketOrder = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(trade_size),
        )
        self.entry = order
        self.submit_order(order)

    def trailing_stop_buy(self, instrument_id) -> None:
        """
        Users simple trailing stop BUY for (``SHORT`` positions).
        """
        if not self.instrument:
            self.log.error("No instrument loaded")
            return

        atr_value = self.atr_values[str(instrument_id)]
        
        # last_quote = self.cache.quote_tick(self.config.instrument_id)
        # if not last_quote:
        #     self.log.warning("Cannot submit order: no quotes yet")
        #     return

        current_position = self.cache.position(self.position_id)
        if not current_position:
            self.log.warning("Cannot place trailing stop: no open long position.", color=LogColor.YELLOW)
            return

        trade_size = Decimal(current_position.quantity.as_double())
        offset = atr_value * self.config.trailing_atr_multiple
        order: TrailingStopMarketOrder = self.order_factory.trailing_stop_market(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(trade_size),
            # sl_price=self.instrument.make_price(sl_price),
            # limit_offset=Decimal(f"{offset / 2:.{self.instrument.price_precision}f}"),
            # price=self.instrument.make_price(last_quote.ask_price.as_double() + offset),
            trailing_offset=Decimal(f"{offset:.{self.instrument.price_precision}f}"),
            trailing_offset_type=TrailingOffsetType[self.config.trailing_offset_type],
            trigger_type=TriggerType[self.config.trigger_type],
            reduce_only=True,
            emulation_trigger=TriggerType[self.config.emulation_trigger],
        )

        self.trailing_stop = order
        self.submit_order(order, position_id=self.position_id)

    def trailing_stop_sell(self, instrument_id) -> None:
        """
        Users simple trailing stop SELL for (LONG positions).
        """
        if not self.instrument:
            self.log.error("No instrument loaded")
            return

        atr_value = self.atr_values[str(instrument_id)]
        
        # last_quote = self.cache.quote_tick(self.config.instrument_id)
        # if not last_quote:
        #     self.log.warning("Cannot submit order: no quotes yet")
        #     return
        
        current_position = self.cache.position(self.position_id)
        if not current_position:
            self.log.warning("Cannot place trailing stop: no open long position.", color=LogColor.YELLOW)
            return

        trade_size = Decimal(current_position.quantity.as_double())
        offset = atr_value * self.config.trailing_atr_multiple
        order: TrailingStopMarketOrder = self.order_factory.trailing_stop_market(
            instrument_id=instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(trade_size),
            # sl_price=self.instrument.make_price(sl_price),
            # limit_offset=Decimal(f"{offset / 2:.{self.instrument.price_precision}f}"),
            # price=self.instrument.make_price(last_quote.bid_price.as_double() - offset),
            trailing_offset=Decimal(f"{offset:.{self.instrument.price_precision}f}"),
            trailing_offset_type=TrailingOffsetType[self.config.trailing_offset_type],
            trigger_type=TriggerType[self.config.trigger_type],
            reduce_only=True,
            emulation_trigger=TriggerType[self.config.emulation_trigger],
        )

        self.trailing_stop = order
        self.submit_order(order, position_id=self.position_id)

    def place_bracket_order(self, instrument_id, side: OrderSide, entry_bar: Bar, sl_px: Optional[Decimal] = None, tp_px: Optional[Decimal] = None, **kwargs) -> None:
        """
        Place a bracket order (buy or sell) using ATR-based distance.

        :param side: OrderSide.BUY or OrderSide.SELL
        :param entry_bar: latest Bar object to derive price 
        """
        if not self.instrument:
            self.log.error("No instrument loaded")
            return

        # Fetch latest closing price and swing prices
        entry_px = Decimal(entry_bar.close.as_double())
        
        atr = kwargs.get("atr", None)
        if atr is None:
            self.log.info()
            return
        
        # Calculate trade size
        trade_size = self.calculate_position_size(self.instrument, entry_px)
        
        if sl_px is None or tp_px is None:
            # ATR-based distance for SL/TP levels
            dist = self.config.bracket_distance_atr * atr

            # Determine SL and TP relative adjustments
            if side == OrderSide.BUY:
                sl_px = entry_px - Decimal(dist)
                tp_px = entry_px + Decimal(dist)
            else:  # SELL
                sl_px = entry_px + Decimal(dist)
                tp_px = entry_px - Decimal(dist)

        ol: OrderList = self.order_factory.bracket(
            instrument_id=instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(trade_size),
            time_in_force=TimeInForce.GTD,
            expire_time=self.clock.utc_now() + timedelta(seconds=60),
            entry_price=self.instrument.make_price(entry_bar.close), 
            entry_trigger_price=self.instrument.make_price(entry_bar.close),
            sl_trigger_price=self.instrument.make_price(sl_px),
            tp_price=self.instrument.make_price(tp_px),
            entry_order_type=OrderType.LIMIT_IF_TOUCHED,
            emulation_trigger=TriggerType[self.config.emulation_trigger],
        )

        self.submit_order_list(ol)
