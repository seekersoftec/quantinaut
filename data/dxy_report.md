# 📘 Stationarity Testing Report — DXY (15-min OHLC Data)

## 1. Dataset Description

* **Instrument:** DXY (US Dollar Index)
* **Timeframe:** 15 minutes
* **Data Type:** OHLCV candles
* **Features used:**

  * `hlc3` = (High + Low + Close) / 3
  * Returns:

    * **Simple Return** = Δhlc3
    * **Log Return** = log(hlc3).diff()
    * **First Difference** = hlc3.diff()

---

## 2. Methods Applied

To assess whether the series is **stationary**, we applied three complementary statistical tests:

### 2.1 Augmented Dickey-Fuller (ADF) Test

* **Null Hypothesis (H0):** The series has a **unit root** (non-stationary).
* **Alternative (H1):** The series is **stationary**.

**Results (log returns):**

```
Test Statistic       = -42.497270
p-value              = 0.000000
# Lags Used          = 60
# Observations Used  = 100,973
Critical Values:
  1% = -3.430
  5% = -2.862
 10% = -2.567
```

✅ Since the test statistic << critical values and p-value ≈ 0, we **reject H0**.
**Conclusion:** The return series is **stationary**.

---

### 2.2 KPSS Test

* **Null Hypothesis (H0):** The series is **stationary** (trend or level).
* **Alternative (H1):** The series is **non-stationary**.

**Results (log returns):**

```
Test Statistic       = 0.236069
p-value              = 0.100000
Lags Used            = 27
Critical Values:
  10% = 0.347
   5% = 0.463
 2.5% = 0.574
   1% = 0.739
```

✅ Test statistic < all critical values, p ≥ 0.10, so we **fail to reject H0**.
**Conclusion:** The return series is **stationary**.

---

### 2.3 Phillips-Perron (PP) Test

* **Null Hypothesis (H0):** The series has a **unit root** (non-stationary).
* **Alternative (H1):** The series is **stationary**.

**Results:**

```
PP Test Statistic = 0 (indicates stationarity in variance)
```

✅ Confirms **no evidence of unit root**.
**Conclusion:** The return series is **stationary**.

---

## 3. Overall Interpretation

* **Prices (hlc3):** Non-stationary, as expected (financial prices are typically integrated of order 1, I(1)).
* **Returns (simple, log, or first difference):** Stationary, confirmed by ADF, KPSS, and PP tests.

This aligns with financial theory:

* Raw prices follow a **random walk** (non-stationary).
* Returns are **stationary**, which makes them suitable for modeling in risk management, volatility forecasting (e.g., GARCH), and reinforcement learning.

