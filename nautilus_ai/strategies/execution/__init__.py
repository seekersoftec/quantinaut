"""
Execution Engine Module

This module implements an automated execution algorithm. It subscribes to
incoming trade signals, executes orders based on signal parameters, and
manages positions and trailing stops.

The engine supports both bracket orders (with entry, stop-loss, and take-profit)
and trailing stop orders. It is designed to be easily extended or integrated into
a larger system.


Notes
------
Markets often transition between different regimes characterized by varying levels of volatility and trading volume. 
Identifying these regimes can provide insights into market sentiment and potential price movements.

Common Regimes:
    1. Low Volume, Low Volatility: Indicates consolidation phases; potential for breakout.
    2. High Volume, High Volatility: Suggests heightened market activity; often seen during news events.
    3. High Volume, Low Volatility: May precede significant price moves; accumulation or distribution phases.
    4. Low Volume, High Volatility: Can signal uncertainty or lack of consensus among traders.
    
    | Regime | Strategy Type                | Position Sizing                                     |
    | ------ | ---------------------------- | --------------------------------------------------- |
    | LV\\_LV | Mean‑Reversion / Range‑Bound | Smaller sizes, tight stops                          |
    | HV\\_HV | Momentum / Breakout          | Larger sizes, trailing stops                        |
    | HV\\_LV | Accumulation/Distribution    | Scaled entries, multi‑leg scaling (e.g. pyramiding) |
    | LV\\_HV | Volatility Fade / Options    | Gamma‑scalping or volatility selling                |


TODO
----
- Account for the fact the NDS can give a different signal from the strategy e.g if NDS says -0.618 which might indicate a bearish move and the strategy gives a bullish siganl, how do we resolve it?
- Consider converting to cython if more speed is needed.

"""
from nautilus_ai.execution.risk_engine import AdaptiveRiskEngine, AdaptiveRiskEngineConfig
from nautilus_ai.execution.risk_models import *
