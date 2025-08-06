from libc.math cimport atan, M_PI
import numpy as np
cimport numpy as np

from nautilus_trader.core.correctness cimport Condition
from nautilus_trader.indicators.base.indicator cimport Indicator
from nautilus_trader.model.data cimport Bar
from nautilus_trader.model.data cimport QuoteTick
from nautilus_trader.model.data cimport TradeTick
from nautilus_trader.model.objects cimport Price


cdef class LinearRegressionChannel(Indicator):
    """
    Linear Regression Bands on High, Low & Close.

    This indicator fits three separate Ordinary Least Squares (OLS) regressions over the last `period` bars:
      • High prices → upper band
      • Close prices → center line (value)
      • Low prices → lower band
    
    Usage and Benefits:
    -------------------
    - It can be used to identify trends and volatility in the market by analyzing the distance between the bands and the center line.
    - The bands can help traders identify potential overbought or oversold conditions, as well as provide dynamic support and resistance levels.
    - It can act as a drop-in replacement for Bollinger Bands, but with a more precise regression-based approach.

    How it works:
    -------------
    - Each band is calculated independently using its own regression (no averaging of High/Low/Close).
    - The center line (.value) is derived from the Close regression.
    - The upper and lower bands are derived from the High and Low regressions, respectively.
    - R² is computed only for the Close regression, but can be extended to High/Low if needed.

    Attributes:
        period (int): The number of periods for the regression calculation.
        slope (float): The slope of the regression line for the prices.
        intercept (float): The intercept of the regression line for the prices.
        R2 (float): The R-squared value of the regression for the prices.
        degree (float): The angle of the regression line for the prices in degrees.
        cfo (float): The coefficient of variation of the residuals for the prices.
        upper (float): The upper band value derived from the High regression.
        lower (float): The lower band value derived from the Low regression.
        value (float): The current value of the regression line derived from the hlc3 regression.
                       Specifically, this is the value of the line at the (period - 1) x-coordinate,
                       assuming x-coordinates range from 0 to (period - 1) or a similar mapping for your x_arr.
        next_estimated_price (float): The predicted value of the regression line for the next period.
        count (int): The number of bars processed.
        trend (int): An indicator of the trend direction and its momentum.
                     - 2: Accelerating Up (Strong Uptrend)
                     - 1: Decelerating Up (Moderate Uptrend)
                     - 0: Sideways/Neutral Trend
                     - -1: Decelerating Down (Moderate Downtrend)
                     - -2: Accelerating Down (Strong Downtrend)

    Raises:
        ValueError: If `period` is not greater than zero.

    # TODO: Integrate error feedback mechanism, should constantly adjust LRF(self.value) based on previous errors.
    # TODO: The geometry indicator is to be merged with this indicator 
    """

    def __init__(self, int period=152, bint enable_dev=False, double dev_multiplier=2.0):
        Condition.positive_int(period, "period")
        Condition.positive(dev_multiplier, "dev_multiplier")
        super().__init__(params=[period, enable_dev, dev_multiplier])

        self.period = period
        self.enable_dev = enable_dev 
        self.dev_multiplier = dev_multiplier
        
        self.count = 0
        self.index = 0

        # Preallocate ring buffers
        self._highs = np.zeros(period, dtype=np.float64)
        self._lows = np.zeros(period, dtype=np.float64)
        self._prices = np.zeros(period, dtype=np.float64)
        self._prev_slope = 0.0

        # Cache x and regression constants
        self.x_arr = np.arange(1, period + 1, dtype=np.float64)
        self.x_sum = 0.5 * period * (period + 1)
        self.x2_sum = self.x_sum * (2 * period + 1) / 3.0
        self.denom = period * self.x2_sum - self.x_sum * self.x_sum

        self.slope = 0.0
        self.intercept = 0.0
        self.degree = 0.0
        self.cfo = 0.0
        self.R2 = 0.0
        self.dev = 0.0
        self.upper = 0.0
        self.lower = 0.0
        self.value = 0.0
        self.next_estimated_price = 0.0 
        self.count = 0
        self.trend = 0 

    cpdef void handle_quote_tick(self, QuoteTick tick):
        """
        Update the indicator with the given tick.

        Parameters
        ----------
        tick : TradeTick
            The tick for the update.

        """
        Condition.not_none(tick, "tick")

        cdef double bid = Price.raw_to_f64_c(tick._mem.bid_price.raw)
        cdef double ask = Price.raw_to_f64_c(tick._mem.ask_price.raw)
        cdef double mid = (ask + bid) / 2.0
        self.update_raw(ask, bid, mid)

    cpdef void handle_trade_tick(self, TradeTick tick):
        """
        Update the indicator with the given tick.

        Parameters
        ----------
        tick : TradeTick
            The tick for the update.

        """
        Condition.not_none(tick, "tick")

        cdef double price = Price.raw_to_f64_c(tick._mem.price.raw)
        self.update_raw(price, price, price)

    cpdef void handle_bar(self, Bar bar):
        """
        Update the indicator with the given bar.

        Parameters
        ----------
        bar : Bar
            The update bar.

        """
        Condition.not_none(bar, "bar")

        self.update_raw(bar.high.as_double(),
                        bar.low.as_double(),
                        bar.close.as_double())
    
    cpdef void update_raw(self, double high, double low, double close):
        """
        Updates the indicator with the given high, low, and close prices.

        This method processes the high, low, and close prices for the current bar and updates the internal state of the indicator.
        It calculates the regression lines for high, low, and close prices, computes residuals, and updates the upper, lower, and center bands.

        Parameters:
        ----------
        high : double
            The high price of the current bar.
        low : double
            The low price of the current bar.
        close : double
            The close price of the current bar.

        Notes:
        ------
        - The method appends the high, low, and close prices to their respective deques.
        - It performs a warm-up check to ensure enough data points are available for regression calculations.
        - The center band is derived from the regression of close prices.
        - The upper and lower bands are derived from the regression of high and low prices, respectively.
        - Residuals and R² are computed for the price regression.

        Sources:
        https://tradethatswing.com/how-to-use-regression-channels-to-aid-in-forex-trade-selection-and-analysis/
        """
        cdef double hlc3 = (high + low + close) / 3.0

        self._highs[self.index] = high
        self._lows[self.index] = low
        self._prices[self.index] = hlc3
        self.index = (self.index + 1) % self.period
        self.count += 1

        # warm-up indicator logic
        if not self.initialized:
            self._set_has_inputs(True)
            if self.count < self.period: # Changed from len(self._prices) as self.count tracks actual filled elements
                return
            self._set_initialized(True)

        # Align rolling buffers
        # Ensure that the data used for regression is in the correct order (oldest to newest relative to x_arr)
        cdef np.ndarray hi = np.roll(self._highs, -self.index).copy()
        cdef np.ndarray lo = np.roll(self._lows, -self.index).copy()
        cdef np.ndarray cl = np.roll(self._prices, -self.index).copy()

        # x = [1,2,...,period] - self.x_arr is pre-calculated in __init__
        # cdef np.ndarray x_arr = np.arange(1, self.period+1, dtype=np.float64)
        # cdef double x_sum  = 0.5 * self.period * (self.period + 1)
        # cdef double x2_sum = x_sum * (2*self.period + 1) / 3.0
        # cdef double denom  = self.period * x2_sum - x_sum*x_sum

        # Fit regression lines
        # Initialize variables to avoid uninitialized reference errors
        cdef double m_hi = 0.0, b_hi = 0.0, m_lo = 0.0, b_lo = 0.0, m_cl = 0.0, b_cl = 0.0
        # fit each series
        fit_line(self.x_arr, hi, self.x_sum, self.x2_sum, self.denom, &m_hi, &b_hi)
        fit_line(self.x_arr, lo, self.x_sum, self.x2_sum, self.denom, &m_lo, &b_lo)
        fit_line(self.x_arr, cl, self.x_sum, self.x2_sum, self.denom, &m_cl, &b_cl)

        # Residuals and R²
        cdef np.ndarray fit = m_cl * self.x_arr + b_cl # Fitted values for close prices
        cdef np.ndarray resid = fit - cl # Residuals for close prices (fit - actual)
        cdef double mean_cl = cl.mean()
        cdef double sst = np.sum((cl - mean_cl) ** 2)  # Total sum of squares for close prices
        cdef double ssr = np.sum(resid ** 2)          # Sum of squared residuals for close prices

        self.degree = 180.0 * atan(m_cl) / M_PI
        self.cfo = 100.0 * resid[-1] / cl[-1]
        self.R2 = 1.0 - ssr / sst if sst != 0.0 else -np.inf
        self.slope = m_cl
        self.intercept = b_cl
        # Value of the regression line at the current end of the period (x = self.period, since x_arr is 1 to period)
        self.value = m_cl * self.period + b_cl # Adjusted for x_arr from 1 to period

        # Calculate the predicted price for the next period
        # This is the current value (at x=period) extended by one 'slope' increment (for x=period+1)
        self.next_estimated_price = self.value + self.slope

        # Calculate the Standard Deviation of Residuals (often referred to as 'dev' or Standard Error of Estimate)
        # This measures the typical distance of the close prices from the regression line.
        cdef double current_dev = np.sqrt(ssr / self.period)
        self.dev = current_dev # Store the calculated standard deviation of residuals

        if self.enable_dev:
            # If dev is enabled, upper and lower bands are calculated based on the central line (self.value)
            # plus/minus 'dev' multiplied by a configurable 'dev_multiplier'.
            self.upper = self.value + self.dev * self.dev_multiplier
            self.lower = self.value - self.dev * self.dev_multiplier
        else:
            # If dev is not enabled, upper and lower bands are derived from separate High and Low regressions.
            # These are also calculated for the end of the period (x = self.period)
            self.upper = m_hi * self.period + b_hi # Adjusted for x_arr from 1 to period
            self.lower = m_lo * self.period + b_lo # Adjusted for x_arr from 1 to period

        # Direction - Combined trend and momentum 
        if self.slope > 0:
            # Current trend is upwards
            if self.slope > self._prev_slope:
                self.trend = 2  # Accelerating Up (Strong Uptrend)
            else:
                self.trend = 1  # Decelerating Up (Moderate Uptrend)
        elif self.slope < 0:
            # Current trend is downwards
            if self.slope < self._prev_slope: # Slope becoming more negative
                self.trend = -2 # Accelerating Down (Strong Downtrend)
            else:
                self.trend = -1 # Decelerating Down (Moderate Downtrend)
        else:
            # Trend is flat or neutral (slope is 0 or very close to 0)
            self.trend = 0 # Sideways/Neutral trend

        self._prev_slope = self.slope

    cpdef void _reset(self):
        self._highs.fill(0.0)
        self._lows.fill(0.0)
        self._prices.fill(0.0)
        self._prev_slope = 0.0
        self.slope = 0.0
        self.intercept = 0.0
        self.degree = 0.0
        self.cfo = 0.0
        self.upper = 0.0
        self.lower = 0.0
        self.value = 0.0
        self.next_estimated_price = 0.0 # Reset new attribute
        self.R2 = 0.0
        self.dev = 0.0
        self.count = 0
        self.trend = 0


cdef inline void fit_line(np.ndarray x_arr, np.ndarray y, double x_sum, double x2_sum, double denom, double* slope, double* intercept):
    """
    Helper function to calculate the slope and intercept of a regression line using OLS.

    Parameters:
    ----------
    x_arr : np.ndarray
        The x values (e.g., time indices).
    y : np.ndarray
        The y values (e.g., prices).
    x_sum : double
        The sum of x values.
    x2_sum : double
        The sum of squared x values.
    denom : double
        The denominator for the OLS calculation.
    slope : double*
        Pointer to store the calculated slope.
    intercept : double*
        Pointer to store the calculated intercept.
    """
    cdef double y_sum = y.sum()
    cdef double xy_sum = (x_arr * y).sum()
    # Check for zero denominator to prevent division by zero
    if denom == 0:
        slope[0] = 0.0
        intercept[0] = y_sum / len(x_arr) # If denom is zero, assume horizontal line at average
    else:
        slope[0] = (len(x_arr) * xy_sum - x_sum * y_sum) / denom
        intercept[0] = (y_sum * x2_sum - x_sum * xy_sum) / denom
