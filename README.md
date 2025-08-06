# Nautilus AI

Nautilus AI is a project that aims to create machine learning-based trading bots, leveraging the high-performance capabilities of the [**Nautilus Trader**](https://github.com/nautechsystems/nautilus_trader/) framework. The core objective is to replicate and expand upon the concepts of the [**intelligent-trading-bot**](https://github.com/asavinov/intelligent-trading-bot) project, focusing on using cutting-edge machine learning algorithms and sophisticated feature engineering to build a robust and autonomous trading system.


### Approaches

The project will explore two primary approaches for building the ML-based trading bots:

* **Indicator-Based Approach**: Create new machine learning-based indicators. These indicators can be trained and optimized independently to generate signals (e.g., buy, sell, hold) that can then be used by a traditional trading strategy. This approach is well-suited for **online learning**, where the model can continuously adapt to new market data in real-time.

* **Strategy-Based Approach**: Develop complete machine learning-based trading strategies. This involves training a model to directly output trading decisions (e.g., entering or exiting a position) based on a wide range of market and historical data. This method is often more suitable for **offline learning**, where the model is trained on a large dataset and then deployed to make predictions without further training.

