from abc import ABC, abstractmethod
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Any, Dict

from nautilus_trader.common.enums import LogColor
from nautilus_trader.model.instruments import Instrument

# Set global decimal precision
getcontext().prec = 28



class BaseRiskModel(ABC):
    def __init__(self, strategy, max_size=None, **kwargs):
        self.strategy = strategy
        self.max_size = Decimal(str(max_size)) if max_size else None
        self.params = {k: Decimal(str(v)) for k, v in kwargs.items()}
        self.trailing_mode = kwargs.get("trailing_mode", None)
        self.trailing_offset = kwargs.get("trailing_offset", None)

    @abstractmethod
    def get_size(self, capital: Decimal, **kwargs: Any) -> Decimal:
        """
        Compute the position size.

        Parameters:
            capital (Decimal): Total available capital.
            **kwargs: Additional parameters specific to the strategy.

        Returns:
            Decimal: Size of the position to take.
        """
        pass

    

# === Plugin registry ===
RISK_MODEL_REGISTRY: Dict[str, BaseRiskModel] = {}


def register_model(name: str):
    """
    Decorator to register a position sizing strategy.

    Parameters:
        name (str): Name of the sizing strategy.

    Returns:
        Callable: Class decorator.
    """
    def decorator(cls):
        RISK_MODEL_REGISTRY[name] = cls
        return cls
    return decorator


@register_model("fixed_fractional")
class FixedFractionModel(BaseRiskModel):
    """
    Fixed Fractional Sizing.

    Parameters:
        risk_pct (float or Decimal): Fraction of capital to risk per trade (default: 0.02).
    """
    def __init__(self, risk_fraction=0.01, min_size=1.0, max_size=None, **kwargs):
        """
        Parameters:
        - risk_fraction: e.g., 0.01 means risk 1% of equity per trade
        - min_size: minimum order size allowed
        """
        self.risk_fraction = risk_fraction
        self.min_size = min_size
        super().__init__(max_size, **kwargs)
        self.risk_pct = self.params.get("risk_pct", Decimal("0.02"))

    def get_size(self, capital: Decimal, **kwargs):
        # entry_price, stop_price, account_state, = kwargs
        
        # # Calculate risk per unit
        # risk_per_unit = abs(entry_price - stop_price)

        # # Handle edge case
        # if risk_per_unit == 0.0:
        #     return self.min_size

        # # Get equity and maximum allowed loss
        # equity = account_state.total_equity.as_double()
        # max_loss = equity * self.risk_fraction

        # # Raw size
        # size = max_loss / risk_per_unit

        # return max(size, self.min_size)
        return capital * self.risk_pct


@register_model("dynamic_fractional")
class DynamicFractionalModel(BaseRiskModel):
    """
    Dynamic Fractional Sizing(Linear Position Scaling). 
    Add dynamic position size (risk %) based on win probability.

    Parameters:
        base_risk_pct (float): Minimum risk size (e.g. 0.01 = 1%)
        scale_factor (float): Risk scale per 1.0 prob delta
        threshold (float): Neutral probability point (e.g. 0.5)
        min_risk (float): Minimum allowed risk size
        max_risk (float): Max allowed risk size
    """

    def get_size(self, capital: Decimal, **kwargs) -> Decimal:
        base_risk_pct = self.params.get("base_risk_pct", Decimal("0.01"))
        win_prob = kwargs.get("win_prob", Decimal("0.5"))
        threshold = self.params.get("threshold", Decimal("0.5"))
        scale_factor = self.params.get("scale_factor", Decimal("0.05"))
        min_risk = self.params.get("min_risk", Decimal("0.005"))
        max_risk = self.params.get("max_risk", Decimal("0.03"))
        
        risk_pct = base_risk_pct + (win_prob - threshold) * scale_factor
        risk_pct = max(min_risk, min(risk_pct, max_risk))
        return capital * risk_pct

@register_model("kelly")
class KellyModel(BaseRiskModel):
    """
    Kelly Criterion Sizing.

    Parameters:
        risk_factor (float or Decimal): Fraction of full Kelly to use (default: 1.0).

    Expects:
        win_rate (float): Historical win probability.
        reward_risk (float): Average reward-to-risk ratio.
    """

    def get_size(self, capital: Decimal, **kwargs) -> Decimal:
        win_rate = Decimal(str(kwargs.get("win_rate", 0)))
        reward_risk = Decimal(str(kwargs.get("reward_risk", 0)))

        if win_rate == 0 or reward_risk == 0:
            return Decimal("0")

        kelly_fraction = win_rate - ((Decimal("1") - win_rate) / reward_risk)
        kelly_fraction = max(Decimal("0"), min(kelly_fraction, Decimal("1")))

        risk_factor = self.params.get("risk_factor", Decimal("1"))
        return capital * kelly_fraction * risk_factor


@register_model("volatility_adjusted")
class VolatilityAdjustedModel(BaseRiskModel):
    """
    Volatility-Adjusted Sizing.

    Parameters:
        target_volatility (float or Decimal): Desired portfolio volatility (default: 0.02).

    Expects:
        volatility (float): Estimated asset volatility.
    """

    def get_size(self, capital: Decimal, **kwargs) -> Decimal:
        volatility = Decimal(str(kwargs.get("volatility", 0)))
        if volatility == 0:
            return Decimal("0")

        target_volatility = self.params.get("target_volatility", Decimal("0.02"))
        return capital * (target_volatility / volatility)


@register_model("risk_based")
class RiskBasedModel(BaseRiskModel):
    """
    Risk-Based Sizing by Stop Loss Distance.

    Parameters:
        risk_pct (float or Decimal): Max risk per trade as a fraction of capital (default: 0.01).

    Expects:
        stop_distance (float): Price distance to stop loss.
    """

    def get_size(self, capital: Decimal, **kwargs) -> Decimal:
        stop_distance = Decimal(str(kwargs.get("stop_distance", 0)))
        if stop_distance == 0:
            return Decimal("0")

        risk_pct = self.params.get("risk_pct", Decimal("0.01"))
        risk_amount = capital * risk_pct
        return risk_amount / stop_distance


@register_model("signal_confidence")
class SignalConfidenceModel(BaseRiskModel):
    """
    Signal-Confidence-Based Sizing.

    Parameters:
        base_risk_pct (float or Decimal): Base risk percentage (default: 0.01).

    Expects:
        confidence (float): Confidence level from model (e.g., 0.0–1.0).
    """

    def get_size(self, capital: Decimal, **kwargs) -> Decimal:
        confidence = Decimal(str(kwargs.get("confidence", 0.5)))
        base_risk_pct = self.params.get("base_risk_pct", Decimal("0.01"))
        return capital * base_risk_pct * confidence


def dynamic_size(confidence, vol, base_risk=0.01):
    # Riskier environments, smaller size
    vol_adj = min(1.0, 0.02 / vol)  # Cap risk at 2% volatility
    return base_risk * confidence * vol_adj

class RiskModelFactory:
    """
    Factory to create risk model instances.

    Methods:
        create(name, **kwargs): Instantiates a model by name.
    """

    @staticmethod
    def create(name: str, **kwargs) -> BaseRiskModel:
        """
        Create a risk model instance by name.

        Parameters:
            name (str): Name of the registered model.
            **kwargs: Parameters passed to the model.

        Returns:
            BaseRiskModel: model instance.

        Raises:
            ValueError: If the model name is not registered.
        """
        if name not in RISK_MODEL_REGISTRY:
            raise ValueError(f"Unknown model strategy: {name}")
        
        # Remove 'name' from kwargs if present
        kwargs.pop('name', None)
        init_args = kwargs.get("init_args", {})
        return RISK_MODEL_REGISTRY[name](**init_args)


def calculate_position_size(
    price: Decimal,
    instrument: Instrument,
    account_state: Any,  
    risk_model: BaseRiskModel,
    logger: Any,  
) -> Decimal:
    """
    Calculates the total trade size using the configured position sizing algorithm,
    stop-loss distance, and current market price, while enforcing instrument limits.

    Parameters:
    -----------
    price (Decimal): Current market price of the instrument.
    instrument (Instrument): The instrument being traded.
    account_state (Any): Current state of the trading account (e.g., free balance).
    risk_model (BaseRiskModel): The risk model instance to determine capital to risk.
    logger (Any): Logger instance for logging messages.

    Returns:
    --------
    Decimal: Total position size.

    Raises:
    -------
    ValueError: If price is zero or negative.
    """
    if price <= 0:
        raise ValueError("Price must be > 0")

    # Get account and free capital
    # The commented line `cache.account_for_venue` suggests `account_state`
    # might be an object that directly provides account details or a cache.
    # Assuming `account_state` already holds the necessary account object.
    account = account_state
    free_balance = Decimal(account.balance_free(instrument.quote_currency).as_decimal())

    logger.info(f"Free balance for {instrument.quote_currency}: {free_balance}", color=LogColor.CYAN)

    # Compute capital to risk using position model
    risk_amt = risk_model.get_size(capital=free_balance)
    logger.info(f"Calculated capital to risk: {risk_amt}", color=LogColor.CYAN)

    # Convert to raw position size based on risk amount and price
    # Note: The original docstring mentions 'stop-loss distance', but the calculation
    # `risk_amt / Decimal(price)` implies `risk_amt` is already the desired notional
    # size or a percentage of capital to be directly converted. If `risk_amt`
    # represents the 'risk capital' and the `stop-loss distance` is used in
    # `risk_model.get_size`, then this calculation is correct for `raw_size`.
    # If `risk_amt` is the maximum loss amount in currency, then `raw_size = risk_amt / stop_loss_distance_per_unit`.
    # Based on the code, `risk_amt` seems to be the total notional value to trade.
    raw_size = risk_amt / price

    logger.info(f"Initial raw position size (before instrument limits): {raw_size}", color=LogColor.CYAN)

    # Enforce instrument limits
    # Ensure notional value meets minimum
    if instrument.min_notional is not None:
        min_notional = Decimal(instrument.min_notional.as_decimal())
        current_notional = raw_size * price
        if current_notional < min_notional:
            # Adjust raw_size to meet the minimum notional
            raw_size = min_notional / price
            logger.info(
                f"Adjusted position size to meet minimum notional value ({min_notional}): {raw_size}",
                color=LogColor.YELLOW
            )

    # Check min_quantity
    if instrument.min_quantity is not None:
        min_quantity_decimal = Decimal(instrument.min_quantity.as_decimal())
        if raw_size < min_quantity_decimal:
            raw_size = min_quantity_decimal
            logger.info(
                f"Adjusted position size to meet minimum quantity ({min_quantity_decimal}): {raw_size}",
                color=LogColor.YELLOW
            )

    # Check max_quantity
    if instrument.max_quantity is not None:
        max_quantity_decimal = Decimal(instrument.max_quantity.as_decimal())
        if raw_size > max_quantity_decimal:
            raw_size = max_quantity_decimal
            logger.info(
                f"Adjusted position size to meet maximum quantity ({max_quantity_decimal}): {raw_size}",
                color=LogColor.YELLOW
            )

    # Check max_notional (recalculate notional after potential quantity adjustments)
    if instrument.max_notional is not None:
        max_notional = Decimal(instrument.max_notional.as_decimal())
        current_notional = raw_size * price
        if current_notional > max_notional:
            # Adjust raw_size to meet the maximum notional
            raw_size = max_notional / price
            logger.info(
                f"Adjusted position size to meet maximum notional value ({max_notional}): {raw_size}",
                color=LogColor.YELLOW
            )

    logger.info(f"Raw size after all limit checks: {raw_size}", color=LogColor.CYAN)

    # Quantize to instrument's size precision
    size_precision = Decimal(str(instrument.size_increment))
    # Use quantize with ROUND_DOWN to ensure the size is never above the maximum allowed by precision
    total_size = raw_size.quantize(size_precision, rounding=ROUND_DOWN)

    # Final check: ensure the total_size is not zero if a valid trade was intended
    if total_size <= 0:
        logger.warning(
            "Calculated position size is zero or negative after quantization. "
            "This might indicate insufficient capital or very restrictive instrument limits.",
            color=LogColor.YELLOW
        )
        return Decimal("0") # Explicitly return 0 if size becomes non-positive
    
    total_size = min(total_size, risk_model.max_size) if risk_model.max_size is not None else total_size
    
    logger.info(f"Final calculated position size: {total_size}", color=LogColor.GREEN)

    return total_size
