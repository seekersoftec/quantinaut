# Nautilus AI

Nautilus AI is a project that aims to create machine learning-based trading bots, leveraging the high-performance capabilities of the [**Nautilus Trader**](https://github.com/nautechsystems/nautilus_trader/) framework. The core objective is to use cutting-edge machine learning algorithms and feature engineering to build a robust and autonomous trading system.

## Approaches

Nautilus AI will explore **two complementary design paths**:

### 1. Indicator-Based ML

In this approach, ML models are developed as **predictive indicators** — akin to a leading technical indicator, but trained on historical and live market data.

* **Purpose:** Generate signals (buy/sell/hold, directional probabilities, volatility forecasts, etc.).
* **Integration:** Output feeds into any strategy (ML-based or traditional rule-based).
* **Best for:** **Online learning**, where the model updates continually as new data arrives, adapting to evolving market conditions.

### 2. Strategy-Based ML

Here, ML is embedded directly into the **strategy logic**, allowing the model to decide **when to enter, exit, or hold** based on a holistic market view.

* **Purpose:** Full trade decision-making pipeline, from signal generation to execution.
* **Integration:** May incorporate multiple data sources and indicators (including ML-based ones).
* **Best for:** **Offline learning**, where models are trained on large historical datasets, validated, and then deployed in a live environment.


## Sources

- https://github.com/nautechsystems/nautilus_trader/
- https://github.com/asavinov/intelligent-trading-bot.git 
- https://github.com/microsoft/qlib.git
- https://github.com/vnpy/vnpy.git
- https://github.com/online-ml/river/
- https://github.com/Martingale42/pml-trader
- https://github.com/limx0/nautilus_talks
- https://github.com/hudson-and-thames/mlfinlab/