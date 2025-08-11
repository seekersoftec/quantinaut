# generators/labels_highlow.py

from typing import List
import pandas as pd
import numpy as np
from nautilus_ai.strategies.itb.generators.utils import add_future_aggregations, add_threshold_feature
from nautilus_ai.strategies.itb.generators.generator import Generator, register_generator

"""
Label generation. Labels are features which are used for training.
In forecasting, they are typically computed from future values as 
opposed to normal features computed from past values.
"""


@register_generator("highlow")
class HighLow(LabelGenerator):
    """
    Label generator using relative high/low movements from future windows.
    Implements thresholds like high >= 1%, low <= -2%, etc.
    
    We use the following conventions and dimensions for generating binary labels:
    - Threshold is used to compare all values of some parameter, for example, 0.5 or 2.0 (sign is determined from the context)
    - Greater or less than the threshold. Note that the threshold has a sign which however is determined from the context
    - High or low column to compare with the threshold. Note that relative deviations from the close are used.
      Hence, high is always positive and low is always negative.
    - horizon which determines the future window used to compute all or one
    Thus, a general label is computed via the condition: [all or one] [relative high or low] [>= or <=] threshold
    However, we do not need all combinations of parameters but rather only some of them which are grouped as follows:
    - high >= large_threshold - at least one higher than threshold: 0.5, 1.0, 1.5, 2.0, 2.5
    - high <= small_threshold - all lower than threshold: 0.1, 0.2, 0.3, 0.4
    - low >= -small_threshold - all higher than threshold: 0.1, 0.2, 0.3, 0.4
    - low <= -large_threshold - at least one lower than (negative) threshold: 0.5, 1.0, 1.5, 2.0, 2.5
    Accordingly, we encode the labels as follows (60 is horizon):
    - high_xx (xx is threshold): for big xx - high_xx means one is larger, for small xx - all are less
    - low_xx (xx is threshold): for big xx - low_xx means one is larger, for small xx - all are less
    """
    def generate(self, df: pd.DataFrame, last_rows: int = 0, **kwargs) -> List[str]:
        horizon = self.config["horizon"]
        labels = []
        windows = [horizon]

        # --- HIGH max rel to close ---
        labels += add_future_aggregations(
            df, "high", np.max, windows=windows,
            suffix="_max", rel_column_name="close", rel_factor=100.0
        )
        high_column_name = f"high_max_{horizon}"

        labels += add_threshold_feature(
            df, high_column_name,
            thresholds=[1.0, 1.5, 2.0, 2.5, 3.0],
            out_names=["high_10", "high_15", "high_20", "high_25", "high_30"]
        )

        labels += add_threshold_feature(
            df, high_column_name,
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
            out_names=["high_01", "high_02", "high_03", "high_04", "high_05"]
        )

        # --- LOW min rel to close ---
        labels += add_future_aggregations(
            df, "low", np.min, windows=windows,
            suffix="_min", rel_column_name="close", rel_factor=100.0
        )
        low_column_name = f"low_min_{horizon}"

        labels += add_threshold_feature(
            df, low_column_name,
            thresholds=[-0.1, -0.2, -0.3, -0.4, -0.5],
            out_names=["low_01", "low_02", "low_03", "low_04", "low_05"]
        )

        labels += add_threshold_feature(
            df, low_column_name,
            thresholds=[-1.0, -1.5, -2.0, -2.5, -3.0],
            out_names=["low_10", "low_15", "low_20", "low_25", "low_30"]
        )

        # --- Ratio high_to_low ---
        df[high_column_name] = df[high_column_name].clip(lower=0)
        df[low_column_name] = df[low_column_name].clip(upper=0) * -1

        column_sum = df[high_column_name] + df[low_column_name]
        ratio_column = df[high_column_name] / column_sum
        df[f"high_to_low_{horizon}"] = (ratio_column * 2) - 1

        labels.append(f"high_to_low_{horizon}")
        return labels


@register_generator("highlow2")
class HighLow2(LabelGenerator):
    """
    Label generator based on high/low threshold crossing.
    Adapts `generate_labels_highlow2` into the interface framework.
    
    Generate multiple increase/decrease labels which are typically used for training.

    """

    def generate(self, df: pd.DataFrame, last_rows: int = 0, **kwargs) -> List[str]:
        """
        Generate binary classification labels based on future high/low moves crossing certain thresholds.
        
        Returns:
            List of generated label column names.
        """
        config = self.config

        column_names = config.get('columns')
        close_column = column_names[0]
        high_column = column_names[1]
        low_column = column_names[2]

        function = config.get('function')
        if function not in ['high', 'low']:
            raise ValueError(f"Invalid function: {function}. Must be 'high' or 'low'.")

        thresholds = config.get('thresholds', [])
        if not isinstance(thresholds, list):
            thresholds = [thresholds]

        tolerance = config.get('tolerance', 0.05)
        names = config.get('names')
        horizon = config.get('horizon')

        if not names or len(names) != len(thresholds):
            raise ValueError("Must provide one output name per threshold in `names`.")

        if function == 'high':
            thresholds = [abs(t) for t in thresholds]
            price_columns = [high_column, low_column]
        else:  # function == 'low'
            thresholds = [-abs(t) for t in thresholds]
            price_columns = [low_column, high_column]

        tolerances = [round(-t * tolerance, 6) for t in thresholds]

        labels = []
        for i, threshold in enumerate(thresholds):
            name = names[i]
            first_cross_labels(
                df, horizon, [threshold, tolerances[i]],
                close_column, price_columns, name
            )
            labels.append(name)

        return labels



@register_generator("highlow_combined")
class HighLowCombined(Generator):
    """
    Combined highlow label generator.
    Config must contain 'mode': either 'v1' (HighLow1) or 'v2' (HighLow2).
    """

    def __init__(self, config):
        super().__init__(config)

        mode = config.get("mode", "v1").lower()
        if mode == "v1":
            self._generator = HighLow(config)
        elif mode == "v2":
            self._generator = HighLow2(config)
        else:
            raise ValueError(f"Unsupported HighLow label mode: {mode}. Use 'v1' or 'v2'.")

    def generate(self, df: pd.DataFrame, last_rows: int = 0, **kwargs) -> List[str]:
        return self._generator.generate(df, last_rows, **kwargs)


def _first_location_of_crossing_threshold(df, horizon, threshold, close_column_name, price_column_name):
    """
    First location of crossing the threshold.
    For each point, take its close price, and then find the distance (location, index)
    to the _first_ future point with high or low price higher or lower, respectively
    than the close price.

    If the location (index) is 0 then it is the next point. If location (index) is NaN,
    then the price does not cross the specified threshold during the horizon
    (or there is not enough data, e.g., at the end of the series). Therefore, this
    function can be used to find whether the price will cross the threshold at all
    during the specified horizon.

    The function is somewhat similar to the tsfresh function first_location_of_maximum
    or minimum. The difference is that this function does not search for maximum but rather
    first cross of the threshold.

    Horizon specifies how many points are considered after this point and without this point.

    Threshold is increase or decrease coefficient, say, 50.0 means 50% increase with respect to
    the current close price.
    """

    def fn_high(x):
        if len(x) < 2:
            return np.nan
        p = x[0, 0]  # Reference price
        p_threshold = p*(1+(threshold/100.0))  # Cross line
        idx = np.argmax(x[1:, 1] > p_threshold)  # First index where price crosses the threshold

        # If all False, then index is 0 (first element of constant series) and we are not able to distinguish it from first element being True
        # If index is 0 and first element False (under threshold) then NaN (not exceeds)
        if idx == 0 and x[1, 1] <= p_threshold:
            return np.nan
        return idx

    def fn_low(x):
        if len(x) < 2:
            return np.nan
        p = x[0, 0]  # Reference price
        p_threshold = p*(1+(threshold/100.0))  # Cross line
        idx = np.argmax(x[1:, 1] < p_threshold)  # First index where price crosses the threshold

        # If all False, then index is 0 (first element of constant series) and we are not able to distinguish it from first element being True
        # If index is 0 and first element False (under threshold) then NaN (not exceeds)
        if idx == 0 and x[1, 1] >= p_threshold:
            return np.nan
        return idx

    # Window df will include the current row as well as horizon of past rows with 0 index starting from the oldest row and last index with the current row
    rl = df[[close_column_name, price_column_name]].rolling(horizon + 1, min_periods=(horizon // 2), method='table')

    if threshold > 0:
        df_out = rl.apply(fn_high, raw=True, engine='numba')
    elif threshold < 0:
        df_out = rl.apply(fn_low, raw=True, engine='numba')
    else:
        raise ValueError(f"Threshold cannot be zero.")

    # Because rolling apply processes past records while we need future records
    df_out = df_out.shift(-horizon)

    # For some unknown reason (bug?), rolling apply (with table and numba) returns several columns rather than one column
    out_column = df_out.iloc[:, 0]

    return out_column


def first_cross_labels(df, horizon, thresholds, close_column, price_columns, out_column):
    """
    Produce one boolean column which is true if the price crosses the first threshold
    but does not cross the second threshold in the opposite direction before that.

    For example, if columns are (high, low) and thresholds are [5.0, -1.0]
    then the result is true if price increases by 5% but never decreases lower than 1% during this growth.

    If columns are (low, high) and thresholds are [-5.0, 1.0]
    the result is true if price decreases by 5% but never increases higher than 1% before that.
    """

    # High label - find first (forward) index like +5 of the value exceeds the threshold. Or 0/nan if not found within window
    df["first_idx_column"] = _first_location_of_crossing_threshold(df, horizon, thresholds[0], close_column, price_columns[0])

    # Low label - find first (forward) index like +6 of the value lower than threshold. Or 0/nan if not found within window
    df["second_idx_column"] = _first_location_of_crossing_threshold(df, horizon, thresholds[1], close_column, price_columns[1])

    # The final value is chosen from these two whichever is smaller (as absolute value), that is, closer to this point
    def is_high_true(x):
        if np.isnan(x[0]):
            return False
        elif np.isnan(x[1]):
            return True
        else:
            return x[0] <= x[1]  # If the first cross point is closer to this point than the second one

    df[out_column] = df[["first_idx_column", "second_idx_column"]].apply(is_high_true, raw=True, axis=1)

    # Indexes are not needed anymore
    df.drop(columns=['first_idx_column', 'second_idx_column'], inplace=True)

    return out_column

