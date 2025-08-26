cimport numpy as np
from nautilus_trader.indicators.base.indicator cimport Indicator

cdef class LinearRegressionChannel(Indicator):
    cdef np.ndarray _highs
    cdef np.ndarray _lows
    cdef np.ndarray _prices
    cdef double _prev_slope

    cdef np.ndarray x_arr
    cdef double x_sum, x2_sum, denom

    cdef readonly int period
    """The window period.\n\n:returns: `int`"""

    cdef readonly bint enable_dev
    cdef readonly double dev_multiplier

    cdef readonly double slope
    """The current slope.\n\n:returns: `double`"""
    cdef readonly double intercept
    """The current intercept.\n\n:returns: `double`"""
    cdef readonly double degree
    """The current degree.\n\n:returns: `double`"""
    cdef readonly double cfo
    """The current cfo value.\n\n:returns: `double`"""
    cdef readonly double R2
    """The current R2 value.\n\n:returns: `double`"""

    cdef readonly double dev

    cdef readonly double value
    """The current value.\n\n:returns: `double`"""
    cdef readonly double upper
    """The upper band value.\n\n:returns: `double`"""
    cdef readonly double lower
    """The lower band value.\n\n:returns: `double`"""
    cdef readonly int index
    """The index of bar been processed.\n\n:returns: `int`"""
    cdef readonly int count
    """The number of bars processed.\n\n:returns: `int`"""

    cdef readonly double next_estimated_price 
    """The predicted price for the next period based on the regression line.\n\n:returns: `double`"""

    cdef readonly int trend
    """The current trend.\n\n:returns: `int`"""

    cpdef void update_raw(self, double high, double low, double close)
    