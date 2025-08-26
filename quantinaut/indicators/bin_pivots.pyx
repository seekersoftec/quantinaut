import numpy as np
import pandas as pd
from collections import deque

cimport numpy as np
from cpython.datetime cimport datetime
from nautilus_trader.core.correctness cimport Condition
from nautilus_trader.indicators.base.indicator cimport Indicator
from nautilus_trader.model.data cimport Bar

cdef class BinPivots(Indicator):
    """
    BinPivots.

    Pivot‑based Support/Resistance indicator.

    This indicator identifies support and resistance levels using bin-based techniques.

    Parameters:
    - `lookback` (int): Length of the sliding window(default 252, usually 250-500).
    - `num_bins` (int): Number of bins to use.
    - `min_success_rate` (double): Minimum pivot sucess rate to use as support/resistance level (in range 0-1).
    - `rank` (int): Order statistic to choose (1 = closest, 2 = 2nd closest, etc.).
    Notes:
    -----
    The minimum pivot sucess rate is combined with the closeness of the pivot to the current price.

    The min_success_rate influences:
    - The minimum number of touches required for a pivot to be considered valid.

    """
    def __init__(self,
                 int lookback=252,
                 int num_bins=10,
                 double min_success_rate=0.4,
                 int rank=1):
        Condition.positive_int(lookback, "lookback")
        Condition.positive_int(num_bins, "num_bins")
        Condition.positive(min_success_rate, "min_success_rate")
        Condition.positive_int(rank, "rank")

        Condition.in_range(min_success_rate, 0, 1, "min_success_rate must be between 0 and 1")
        Condition.is_true(rank > 0, "rank must be > 0")

        super().__init__(params=[lookback, num_bins, min_success_rate, rank])
        self.lookback = lookback
        self.num_bins = num_bins
        self.min_success_rate = min_success_rate
        self.rank = rank

        # sliding‐window buffers
        self._highs      = deque(maxlen=lookback)
        self._lows       = deque(maxlen=lookback)
        self._closes     = deque(maxlen=lookback)
        self._bin_edges  = np.full(num_bins+1, np.nan, dtype=np.float64)

        # output arrays
        self.levels = deque(maxlen=num_bins)
        self.pivots_low_stats = np.full(num_bins+1, np.nan, dtype=np.float64) # for support levels
        self.pivots_high_stats = np.full(num_bins+1, np.nan, dtype=np.float64) # for resistance levels
        self.pivots_high  = np.full(num_bins+1, np.nan, dtype=np.float64) # for resistance levels
        self.pivots_low   = np.full(num_bins+1, np.nan, dtype=np.float64) # for support levels

        self.low  = 0.0
        self.high = 0.0
        self.bullish_breakout = False
        self.bearish_breakout = False

    cpdef void handle_bar(self, Bar bar):
        """
        Handles a new bar of data and updates the indicator.

        Parameters:
            bar (Bar): The bar of data to process.
        """
        Condition.not_none(bar, "bar")

        self.update_raw(bar.high.as_double(),
                        bar.low.as_double(), 
                        bar.close.as_double())

    cpdef void update_raw(self, double high, double low, double close):
        # 1) slide window
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        # warm‐up until full window
        cdef int n = len(self._closes)
        if not self.initialized:
            self._set_has_inputs(True)
            if n < self.lookback:
                return
            self._set_initialized(True)

        # Compute Bins
        min_price = np.min(list(self._lows))
        max_price = np.max(list(self._highs))
        self._bin_edges = np.linspace(min_price, max_price, self.num_bins + 1)

        # Group prices by bin
        cdef np.ndarray closes_arr = np.array(list(self._closes), dtype=np.float64)
        cdef int i
        for i in range(self.num_bins):
            lower_bound = self._bin_edges[i]
            upper_bound = self._bin_edges[i+1]

            # Filter closes: [lower, upper) for most bins, [lower, upper] for the last.
            if i == self.num_bins - 1:  # Last bin (e.g., index num_bins - 1)
                in_bin_mask = (closes_arr >= lower_bound) & (closes_arr <= upper_bound)
            else:
                in_bin_mask = (closes_arr >= lower_bound) & (closes_arr < upper_bound)

            closes_in_bin = closes_arr[in_bin_mask]

            if closes_in_bin.size > 0:
                self.levels.append(np.mean(closes_in_bin))
            else:
                self.levels.append(np.nan)

        stats_per_bin = []
        for level in list(self.levels):
            if not np.isnan(level):
                support_stat, resistance_stat = self.calculate_statistics(level)
                if level >= high and resistance_stat >= self.min_success_rate: # above current high
                    self.pivots_high = np.append(self.pivots_high, level)
                if level <= low and support_stat >= self.min_success_rate: # below current low
                    self.pivots_low = np.append(self.pivots_low, level)
            else:
                support_stat, resistance_stat = (0.0, 0.0)
            stats_per_bin.append((support_stat, resistance_stat))

        pivots_low_stats, pivots_high_stats = zip(*stats_per_bin)
        self.pivots_low_stats, self.pivots_high_stats = np.array(pivots_low_stats, dtype=np.float64), np.array(pivots_high_stats, dtype=np.float64)

        if len(self.pivots_high) == 0 or len(self.pivots_low) == 0:
            return

        self.bullish_breakout = high >= self.pivots_high[-1]
        self.bearish_breakout = low <= self.pivots_low[-1]

        # Step 9: Store sorted levels (optional, e.g., by distance from current price)
        # Sort the pivots by distance to current price (forces ranking) in ascending order
        self.pivots_high = self.pivots_high[np.argsort(np.abs(self.pivots_high - high))] # sorted by distance from current level
        self.pivots_low  = self.pivots_low[np.argsort(np.abs(self.pivots_low - low))] # sorted by distance from current level

        # Detect if the first element in both pivots is (approximately) the same
        overlap = (
            len(self.pivots_high) > 0 and
            len(self.pivots_low) > 0 and
            np.isclose(self.pivots_high[0], self.pivots_low[0])
        )

        # If overlapping, remove from the longer list
        if overlap:
            if len(self.pivots_high) > len(self.pivots_low):
                # Remove the first pivot from pivots_high
                self.pivots_high = self.pivots_high[1:]
            else:
                # Remove the first pivot from pivots_low
                self.pivots_low = self.pivots_low[1:]

        # Safety check: ensure enough pivot levels exist
        if len(self.pivots_high) >= self.rank:
            self.high = float(self.pivots_high[self.rank - 1])
        if len(self.pivots_low) >= self.rank:
            self.low = float(self.pivots_low[self.rank - 1])


    cpdef tuple calculate_statistics(self, double level):
        cdef np.ndarray highs_arr = np.array(list(self._highs), dtype=np.float64)
        cdef np.ndarray lows_arr = np.array(list(self._lows), dtype=np.float64)
        cdef np.ndarray closes_arr = np.array(list(self._closes), dtype=np.float64)

        cdef np.ndarray highs_shifted = np.roll(highs_arr, 2)
        cdef np.ndarray lows_shifted = np.roll(lows_arr, 2)

        bull_rejects = ((highs_shifted >= level) & (closes_arr < level)).sum()
        bear_rejects = ((lows_shifted <= level) & (closes_arr > level)).sum()

        total = bull_rejects + bear_rejects
        if total == 0:
            return (0.0, 0.0)

        support_success = bear_rejects / total * 100
        resistance_success = bull_rejects / total * 100
        return support_success, resistance_success

    cpdef dict get_ranked_pivots(self, int max_rank=2):
        """
        Retrieves the top-ranked support and resistance levels based on clustering analysis.

        This method identifies the highest-ranked resistance (R1) and support (S1) levels. 
        It also determines the strongest resistance (SR) and support (SS) levels, defined as those closest to the current price.

        Parameters
        ----------
        max_rank : int
            The maximum number of ranked levels to return for each category. Default is gotten from 2.

        Returns
        -------
        dict
            A dictionary containing the top-ranked resistance and support levels, along with the strongest levels

        Examples
        --------
            The dictionary has the following structure:
        
            {
                "resistances": (R1, R2, ..., Rn),
                "supports": (S1, S2, ..., Sn),
                "SR": strongest_resistance_level,
                "SS": strongest_support_level
            }

        Notes
        -----
        - The resistance levels (R1, R2, ...) are sorted in ascending order of price.
        - The support levels (S1, S2, ...) are sorted in descending order of price.
        - The strongest resistance (SR) and support (SS) are determined by the highest touch counts.
        """
        # Fallback to default rank if invalid
        if max_rank <= 0:
            max_rank = self.rank

        # Select top-N by proximity
        top_highs = self.pivots_high[:max_rank]
        top_lows = self.pivots_low[:max_rank]

        # Build output
        cdef dict pivots_dict = {}

        for i, pivot in enumerate(top_highs):
            pivots_dict[f"R{i+1}"] = float(pivot)

        for i, pivot in enumerate(top_lows):
            pivots_dict[f"S{i+1}"] = float(pivot)

        if top_highs.size > 0:
            pivots_dict["SR"] = float(np.min(top_highs))
        if top_lows.size > 0:
            pivots_dict["SS"] = float(np.max(top_lows))

        return pivots_dict

    cpdef void _reset(self):
            """
            Reset all internal state for a new backtest.
            """
            self._highs.clear()
            self._lows.clear()
            self._closes.clear()
            self._bin_edges[:]  = np.nan

            self.levels.clear()
            self.pivots_low_stats[:] = np.nan
            self.pivots_high_stats[:] = np.nan
            self.pivots_high[:]  = np.nan
            self.pivots_low[:]   = np.nan

            self.low  = 0.0
            self.high = 0.0
            self.bullish_breakout = False
            self.bearish_breakout = False
