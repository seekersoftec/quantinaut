#!/usr/bin/env python3

from decimal import Decimal

from nautilus_trader.model.data import BarType
from quantinaut.portfolio import AdaptiveRiskEngine, AdaptiveRiskEngineConfig
from quantinaut.strategies.rule_policy import RulePolicyConfig, RulePolicy

def risk_engine() -> AdaptiveRiskEngine:
    """Configure and return an AdaptiveRiskEngine instance.

    trailing_atr_multiple=>3, 2.5, 1.5, 0.5, 0.3 | The smaller the timeframe the lesser this value should be.
    max_trade_size=> XRP:150
    """
    config = AdaptiveRiskEngineConfig(
        model_name="fixed_fractional",
        model_init_args={"risk_pct":0.2}, # 0.05, 0.2
        # risk_model=RiskModelConfig(init_args=dict(risk_pct=0.2)), # 0.05, 0.2
        bracket_distance_atr=2.5, # 3
        trailing_atr_multiple=0.3, # 0.3
        trigger_type="MARK_PRICE",
        max_trade_sizes={
            "XRP": Decimal("150"),
            "Any": Decimal("0.1")
            },
    )
    return AdaptiveRiskEngine(config=config)

def rule_policy_strategy(bar_type: BarType) -> RulePolicy:
    """Configure and return a SwingCross strategy instance.
    
    A combination of period(s):
    - 2-3 days(2.5 days) / 1 week - seems to work for BTC/USDT, XRP/USDT
    - 5 hrs / 24 hrs(1 day) - seems to work for BTC/USDT
    - 6 hrs / 24 hrs(1 day) - seems to work for XRP/USDT
    
    5mins: 240/500 
    15mins: 200/500, 250/500, 252/500, 240/500, 245/500
    1hr: 50/200, 
    4hr: 17/35, 21/50
    
    15mins: 
        - 10 / 20 (e.g., 2.5 hours vs. 5 hours of data) lower pnl when compared with 21/42 
        - 14 / 28 (e.g., 3.5 hours vs. 7 hours of data) lower pnl when compared with 21/42 
        - 21 / 42 (e.g., ~5 hours vs. ~10.5 hours of data - Fibonacci-based) seems to work but the sharpe and sortino is quite bad.
    
    rvi_period=>9, 10, 21
    """
    # Configure your strategies
    config = RulePolicyConfig(
        bar_type=bar_type,
        rvi_period=9, # 9, 10, 21
        rvi_threshold=50, # 60
        # atr_vwap_batch_bars=True,
    )
    return RulePolicy(config=config)
