cimport numpy as np
from cpython.datetime cimport datetime
from nautilus_trader.indicators.base.indicator cimport Indicator

cdef class BinPivots(Indicator):
    cdef readonly int lookback, num_bins, rank
    cdef readonly double min_success_rate

    # sliding‐window buffers
    cdef object _highs, _lows, _closes, _bin_edges
    
    # output arrays
    cdef readonly object levels
    cdef readonly np.ndarray pivots_high, pivots_low, pivots_low_stats, pivots_high_stats
    cdef readonly double low, high
    cdef readonly bint bullish_breakout, bearish_breakout

    cpdef void update_raw(self, double high, double low, double close)
    cpdef tuple calculate_statistics(self, double level)
    cpdef dict get_ranked_pivots(self, int max_rank=*)
