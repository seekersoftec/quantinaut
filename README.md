<p align="center">
  <img src="docs/assets/logo.png" alt="Nautilus AI Logo" width="100"/>
</p>

<h1 align="center">Nautilus AI</h1>
<p align="center"><em>Breaking Alpha</em></p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/seekersoftec/nautilus_ai/actions"><img src="https://img.shields.io/github/actions/workflow/status/seekersoftec/nautilus_ai/build.yml?branch=main" alt="Build Status"></a>
  <a href="https://github.com/seekersoftec/nautilus_ai/actions"><img src="https://img.shields.io/badge/Tests-Passing-brightgreen" alt="Tests"></a>
  <a href="../../CONTRIBUTING.md"><img src="https://img.shields.io/badge/Contributions-Welcome-orange.svg" alt="Contributions"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style"></a>
  <a href="https://nautilustrader.io"><img src="https://img.shields.io/badge/Made%20with-Nautilus%20Trader-orange" alt="Nautilus Trader"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Machine%20Learning-Active-green" alt="Machine Learning"></a>
  <a href="https://github.com/seekersoftec/nautilus_ai/stargazers"><img src="https://img.shields.io/github/stars/seekersoftec/nautilus_ai?style=social" alt="Stars"></a>
</p>


<!-- --- -->

**Nautilus AI** — *An open-source, fully autonomous algorithmic trading platform integrating machine learning, quantitative analysis, and feature engineering to deliver adaptive, high-performance strategies in real-time markets.*

<!-- --- -->

## 🚀 Introduction

Welcome to **Nautilus AI** — your gateway to the future of algorithmic trading! This project brings together the power of machine learning and the speed of the [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader/) framework to build intelligent, adaptive trading bots. Whether you're a quant, a developer, or just curious about AI in finance, Nautilus AI is designed to help you experiment, learn, and deploy cutting-edge trading strategies.

---

## 🌟 Project Vision

Nautilus AI aims to:
- Empower traders and researchers to build robust, autonomous trading systems.
- Leverage state-of-the-art ML algorithms for both predictive indicators and strategy optimization.
- Enable rapid experimentation and validation in realistic market environments.

---

## 🧠 How Nautilus AI Works

Nautilus AI separates **where** different types of machine learning shine in a trading stack:

- **Reinforcement Learning (RL):** Powers trading strategies, making smart decisions with full market and portfolio context.
- **Supervised & Unsupervised Learning:** Fuels predictive indicators, forecasting market variables like price direction, volatility, and order flow.

This approach matches the right ML paradigm to the right problem, making your trading system both precise and context-aware.

### 🔍 Online learning
ML models act as predictive indicators — like supercharged technical signals, trained on historical and live data.
- **Purpose:** Generate buy/sell probabilities, price targets, volatility forecasts, and more.
- **Integration:** Plug into strategies (RL or rule-based) for smarter decisions.
- **Best for:** Indicators, adapting to new data as markets evolve.

### 🕹️ Offline learning
Here, ML is the brain of your trading strategy, deciding when to enter, exit, or hold positions with a holistic view.
- **Purpose:** Optimize trade execution and portfolio management.
- **Integration:** Combine multiple inputs — indicators, order book data, market regimes, and more.
- **Best for:** Strategies (batch training) or deep RL, learning optimal policies from simulations or historical data.

---

## ✨ Key Features
- Modular design: Mix and match ML models, indicators, and strategies.
- Fast retraining: Keep your models up-to-date with changing markets.
- Realistic backtesting: Validate ideas in the Nautilus Trader environment.
- Open-source: Build on top of a rich ecosystem of trading and ML libraries.

---

## 🚦 Getting Started
Ready to dive in? Check out the [docs](docs/README.md) or explore the `examples/` folder for sample scripts. To install dependencies, see `pyproject.toml` or use the provided install scripts in `scripts/`.

---

## 📚 Sources & Inspiration

- [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader/)
- [Nautilus Talks](https://github.com/limx0/nautilus_talks)
- [Microsoft Qlib](https://github.com/microsoft/qlib.git)
- [FreqTrade](https://github.com/freqtrade/freqtrade/)
- [VNpy](https://github.com/vnpy/vnpy.git)
- [River (Online ML)](https://github.com/online-ml/river/)
- [Mlfin.py](https://github.com/baobach/mlfinpy/)
- [MLFinLab](https://github.com/hudson-and-thames/mlfinlab/)
- [HyperTrade](https://github.com/karanpratapsingh/HyperTrade/)
- [PML Trader](https://github.com/Martingale42/pml-trader)
- [Intelligent Trading Bot](https://github.com/asavinov/intelligent-trading-bot.git)

---

Happy trading! 🐚🤖
