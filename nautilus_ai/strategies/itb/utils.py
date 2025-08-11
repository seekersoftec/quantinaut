import itertools
from typing import Any, Dict, Union, List
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats


def add_past_weighted_aggregations(df, column_name: str, weight_column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    return _add_weighted_aggregations(df, False, column_name, weight_column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def add_past_aggregations(df, column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    return _add_aggregations(df, False, column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def add_future_aggregations(df, column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    return _add_aggregations(df, True, column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)
    #return _add_weighted_aggregations(df, True, column_name, None, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def _add_aggregations(df, is_future: bool, column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    """
    Compute moving aggregations over past or future values of the specified base column using the specified windows.

    Windowing. Window size is the number of elements to be aggregated.
    For past aggregations, the current value is always included in the window.
    For future aggregations, the current value is not included in the window.

    Naming. The result columns will start from the base column name then suffix is used and then window size is appended (separated by underscore).
    If suffix is not provided then it is function name.
    The produced names will be returned as a list.

    Relative values. If the base column is provided then the result is computed as a relative change.
    If the coefficient is provided then the result is multiplied by it.

    The result columns are added to the data frame (and their names are returned).
    The length of the data frame is not changed even if some result values are None.
    """

    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]

    if rel_column_name:
        rel_column = df[rel_column_name]

    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:
        if not last_rows:
            feature = column.rolling(w).apply(fn)
        else:
            feature = _aggregate_last_rows(column, w, last_rows, fn)

        if is_future:
            feature = feature.shift(-w)

        feature_name = column_name + suffix + '_' + str(w)
        features.append(feature_name)
        if rel_column_name:
            df = df.with_columns((rel_factor * (feature - rel_column) / rel_column).alias(feature_name))
        else:
            df = df.with_columns((rel_factor * feature).alias(feature_name))

    return features


def _add_weighted_aggregations(df, is_future: bool, column_name: str, weight_column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    """
    Weighted rolling aggregation. Normally using np.sum function which means area under the curve.
    """

    column = df[column_name]

    if weight_column_name:
        weight_column = df[weight_column_name]
    else:
        weight_column = pl.Series([1.0] * len(column))

    products_column = column * weight_column

    if isinstance(windows, int):
        windows = [windows]

    if rel_column_name:
        rel_column = df[rel_column_name]

    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:
        if not last_rows:
            feature = products_column.rolling(w).apply(fn)
            weights = weight_column.rolling(w).apply(fn)
        else:
            feature = _aggregate_last_rows(products_column, w, last_rows, fn)
            weights = _aggregate_last_rows(weight_column, w, last_rows, fn)

        feature = feature / weights

        if is_future:
            feature = feature.shift(-w)

        feature_name = column_name + suffix + '_' + str(w)
        features.append(feature_name)
        if rel_column_name:
            df = df.with_columns((rel_factor * (feature - rel_column) / rel_column).alias(feature_name))
        else:
            df = df.with_columns((rel_factor * feature).alias(feature_name))

    return features


def add_area_ratio(df, is_future: bool, column_name: str, windows: Union[int, List[int]], suffix=None, last_rows: int = 0):
    """
    For past, we take this element and compare the previous sub-series: the area under and over this element
    For future, we take this element and compare the next sub-series: the area under and over this element
    """
    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]

    if suffix is None:
        suffix = "_" + "area_ratio"

    features = []
    for w in windows:
        if not last_rows:
            ro = column.rolling(window=w, min_periods=max(1, w // 2))
            feature = ro.apply(area_fn, kwargs=dict(is_future=is_future), raw=True)
        else:  # Only for last row
            feature = _aggregate_last_rows(column, w, last_rows, area_fn, is_future)

        feature_name = column_name + suffix + '_' + str(w)

        if is_future:
            df[feature_name] = feature.shift(periods=-(w-1))
        else:
            df[feature_name] = feature

        features.append(feature_name)

    return features


def area_fn(x, is_future):
    if is_future:
        level = x[0]  # Relative to the oldest element
    else:
        level = x[-1]  # Relative to the newest element
    x_diff = x - level  # Difference from the level
    a = np.nansum(x_diff)
    b = np.nansum(np.absolute(x_diff))
    pos = (b+a)/2
    #neg = (b-a)/2
    ratio = pos / b  # in [0,1]
    ratio = (ratio * 2) - 1  # scale to [-1,+1]
    return ratio


def add_linear_trends(df, is_future: bool, column_name: str, windows: Union[int, List[int]], suffix=None, last_rows: int = 0):
    """
    Use a series of points to compute slope of the fitted line and return it.
    For past, we use previous series.
    For future, we use future series.
    This point is included in series in both cases.
    """
    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]

    if suffix is None:
        suffix = "_" + "trend"

    features = []
    for w in windows:
        if not last_rows:
            ro = column.rolling(window=w, min_periods=max(1, w // 2))
            feature = ro.apply(slope_fn, raw=True)
        else:  # Only for last row
            feature = _aggregate_last_rows(column, w, last_rows, slope_fn)

        feature_name = column_name + suffix + '_' + str(w)

        if is_future:
            df[feature_name] = feature.shift(periods=-(w-1))
        else:
            df[feature_name] = feature

        features.append(feature_name)

    return features


def slope_fn(x):
    """
    Given a Series, fit a linear regression model and return its slope interpreted as a trend.
    The sequence of values in X must correspond to increasing time in order for the trend to make sense.
    """
    X_array = np.asarray(range(len(x)))
    y_array = x
    if np.isnan(y_array).any():
        nans = ~np.isnan(y_array)
        X_array = X_array[nans]
        y_array = y_array[nans]

    #X_array = X_array.reshape(-1, 1)  # Make matrix
    #model = linear_model.LinearRegression()
    #model.fit(X_array, y_array)
    # slope = model.coef_[0]

    slope, intercept, r, p, se = stats.linregress(X_array, y_array)

    return slope


def to_log_diff(sr):
    return np.log(sr).diff()


def to_diff_NEW(sr):
    return 100 * sr.diff() / sr


def to_diff(sr):
    """
    Convert the specified input column to differences.
    Each value of the output series is equal to the difference between current and previous values divided by the current value.
    """

    def diff_fn(x):  # ndarray. last element is current row and first element is most old historic value
        return 100 * (x[1] - x[0]) / x[0]

    diff = sr.rolling(window=2, min_periods=2).apply(diff_fn, raw=True)
    return diff


def _aggregate_last_rows(column, window, last_rows, fn, *args):
    """Rolling aggregation for only n last rows"""
    length = len(column)
    values = [fn(column.iloc[-window - r:length - r].to_numpy(), *args) for r in range(last_rows)]
    feature = pd.Series(data=np.nan, index=column.index, dtype=float)
    feature.iloc[-last_rows:] = list(reversed(values))
    return feature



def _convert_to_relative(fn_outs: list, rel_base, rel_func, percentage):
    # Convert to relative values and percentage (except for the last output)
    rel_outs = []
    size = len(fn_outs)
    for i, feature in enumerate(fn_outs):
        if not rel_base:
            rel_out = fn_outs[i]  # No change requested
        elif (rel_base == "next" or rel_base == "last") and i == size - 1:
            rel_out = fn_outs[i]  # No change because it is the last (no next - it is the base)
        elif (rel_base == "prev" or rel_base == "first") and i == 0:
            rel_out = fn_outs[i]  # No change because it is the first (no previous - it is the base)

        elif rel_base == "next" or rel_base == "last":
            if rel_base == "next":
                base = fn_outs[i + 1]  # Relative to next
            elif rel_base == "last":
                base = fn_outs[size-1]  # Relative to last
            else:
                raise ValueError(f"Unknown value of the 'rel_base' config parameter: {rel_base=}")

            if rel_func == "rel":
                rel_out = feature / base
            elif rel_func == "diff":
                rel_out = (feature - base)
            elif rel_func == "rel_diff":
                rel_out = (feature - base) / base
            else:
                raise ValueError(f"Unknown value of the 'rel_func' config parameter: {rel_func=}")

        elif rel_base == "prev" or rel_base == "first":
            if rel_base == "prev":
                base = fn_outs[i - 1]  # Relative to previous
            elif rel_base == "first":
                base = fn_outs[size-1]  # Relative to first
            else:
                raise ValueError(f"Unknown value of the 'rel_base' config parameter: {rel_base=}")

            if rel_func == "rel":
                rel_out = feature / base
            elif rel_func == "diff":
                rel_out = (feature - base)
            elif rel_func == "rel_diff":
                rel_out = (feature - base) / base
            else:
                raise ValueError(f"Unknown value of the 'rel_func' config parameter: {rel_func=}")

        if percentage:
            rel_out = rel_out * 100.0

        rel_out.name = fn_outs[i].name
        rel_outs.append(rel_out)

    return rel_outs


def fmax_fn(x):
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


def lsbm_fn(x):
    """
    The longest consecutive interval of values higher than the mean.
    A similar feature might be higher than the last (current) value.
    Area under mean/last value is also a variation of this approach but instead of computing the sum of length, we compute their integral (along with the values).

    Equivalent of tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean
    """

    def _get_length_sequences_where(x):
        # [0,1,0,0,1,1,1,0,0,1,0,1,1] -> [1, 3, 1, 2]
        # [0,True,0,0,True,True,True,0,0,True,0,True,True] -> [1, 3, 1, 2]
        # [0,True,0,0,1,True,1,0,0,True,0,1,True] -> [1, 3, 1, 2]
        if len(x) == 0:
            return [0]
        else:
            res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
            return res if len(res) > 0 else [0]

    return np.max(_get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0



def add_threshold_feature(df, column_name: str, thresholds: list, out_names: list):
    """

    :param df:
    :param column_name: Column with values to compare with the thresholds
    :param thresholds: List of thresholds. For each of them an output column will be generated
    :param out_names: List of output column names (same length as thresholds)
    :return: List of output column names
    """

    for i, threshold in enumerate(thresholds):
        out_name = out_names[i]
        if threshold > 0.0:  # Max high
            if abs(threshold) >= 0.75:  # Large threshold
                df[out_name] = df[column_name] >= threshold  # At least one high is greater than the threshold
            else:  # Small threshold
                df[out_name] = df[column_name] <= threshold  # All highs are less than the threshold
        else:  # Min low
            if abs(threshold) >= 0.75:  # Large negative threshold
                df[out_name] = df[column_name] <= threshold  # At least one low is less than the (negative) threshold
            else:  # Small threshold
                df[out_name] = df[column_name] >= threshold  # All lows are greater than the (negative) threshold

    return out_names


def combine_scores_relative(df, buy_column, sell_column, trade_column_out):
    """
    Mutually adjust input buy and sell scores by producing two output scores.
    The idea is that if both scores (buy and sell) are equally high then in the output
    they both will be 0. The output score describe if this score is higher relative to the other.
    The two output scores are in [-1, +1] but have opposite values.
    """

    # compute proportion in the sum
    buy_plus_sell = df[buy_column] + df[sell_column]
    buy_sell_score = ((df[buy_column] / buy_plus_sell) * 2) - 1.0  # in [-1, +1]

    df[trade_column_out] = buy_sell_score  # High values mean buy signal
    #df[buy_column_out] = df[df[buy_column_out] < 0] = 0  # Set negative values to 0

    return buy_sell_score


def combine_scores_difference(df, buy_column, sell_column, trade_column_out):
    """
    This transformation represents how much buy score higher than sell score.
    If they are equal then the output is 0. The output scores have opposite signs.
    """

    # difference
    buy_minus_sell = df[buy_column] - df[sell_column]

    df[trade_column_out] = buy_minus_sell  # High values mean buy signal
    #df[buy_column_out] = df[df[buy_column_out] < 0] = 0  # Set negative values to 0

    return buy_minus_sell


def compute_score_slope(df, model, buy_score_columns_in, sell_score_columns_in):
    """
    Experimental. Currently not used.
    Compute slope of the numeric score over model.get("buy_window") and model.get("sell_window")
    """

    from scipy import stats
    from sklearn import linear_model
    def linear_regr_fn(X):
        """
        Given a Series, fit a linear regression model and return its slope interpreted as a trend.
        The sequence of values in X must correspond to increasing time in order for the trend to make sense.
        """
        X_array = np.asarray(range(len(X)))
        y_array = X
        if np.isnan(y_array).any():
            nans = ~np.isnan(y_array)
            X_array = X_array[nans]
            y_array = y_array[nans]

        # X_array = X_array.reshape(-1, 1)  # Make matrix
        # model = linear_model.LinearRegression()
        # model.fit(X_array, y_array)
        # slope = model.coef_[0]

        slope, intercept, r, p, se = stats.linregress(X_array, y_array)

        return slope

    # if 'buy_score_slope' not in df.columns:
    #    w = 10  #model.get("buy_window")
    #    df['buy_score_slope'] = df['buy_score_column'].rolling(window=w, min_periods=max(1, w // 2)).apply(linear_regr_fn, raw=True)
    #    w = 10  #model.get("sell_window")
    #    df['sell_score_slope'] = df['sell_score_column'].rolling(window=w, min_periods=max(1, w // 2)).apply(linear_regr_fn, raw=True)


def apply_rule_with_score_thresholds_one_row(row, score_column_names, model):
    """
    Same as above but applied to one row rather than data frame. It is used for online predictions.

    Returns signals as a tuple with two values: buy_signal and sell_signal
    """
    parameters = model.get("parameters", {})

    score_column = score_column_names[0]

    buy_score = row[score_column]

    buy_signal = \
        (buy_score >= parameters.get("buy_signal_threshold"))
    sell_signal = \
        (buy_score <= parameters.get("sell_signal_threshold"))

    return buy_signal, sell_signal


def apply_rule_with_slope_thresholds(df, model, buy_score_column, sell_score_column):
    """
    Experimental. Currently not used.
    This rule type evaluates the score itself and also its slope.
    """
    # df['buy_signal_column'] = (df['buy_score_column'] >= model.get("buy_signal_threshold")) & (df['buy_score_slope'].abs() <= model.get("buy_slope_threshold"))
    # df['sell_signal_column'] = (df['sell_score_column'] >= model.get("sell_signal_threshold")) & (df['sell_score_slope'].abs() <= model.get("sell_slope_threshold"))


#
# Helper and exploration functions
#

def find_interval_precision(df: pd.DataFrame, label_column: str, score_column: str, threshold: float):
    """
    Convert point-wise score/label pairs to interval-wise score/label.

    We assume that for each point there is a score and a boolean label. The score can be a future
    prediction while boolean label is whether this forecast is true. Or the score can be a prediction
    that this is a top/bottom while the label is whether it is indeed so.
    Importantly, the labels are supposed to represent contiguous intervals because the algorithm
    will output results for them by aggregating scores within these intervals.

    The output is a data frame with one row per contiguous interval. The intervals are interleaving
    like true, false, true, false etc. Accordingly, there is one label column which takes these
    values true, false etc. The score column for each interval is computed by using these rules:
    - for true interval: true (positive) if there is at least one point with score higher than the threshold
    - for true interval: false (positive) if all points are lower than the threshold
    - for false interval: true (negative) if all points are lower than the threshold
    - for false interval: false (negative) if there is at least one (wrong) points with the score higher than the thresond
    Essentially, we need only one boolean "all lower" function

    The input point-wise score is typically aggregated by applying some kind of rolling aggregation
    but it is performed separately.

    The function is supposed to be used for scoring during hyper-parameter search.
    We can search in level, tolerance, threshold, aggregation hyper-paraemters (no forecasting parameters).
    Or we can also search through various ML forecasting hyper-parameters like horizon etc.
    In any case, after we selected hyper-parameters, we apply interval selection, score aggregation,
    then apply this function, and finally computing the interval-wise score.

    Input data frame is supposed to be sorted (important for the algorithm of finding contiguous intervals).
    """

    #
    # Count all intervals by finding them as groups of points. Input is a boolean column with interleaving true-false
    # Mark true intervals (extremum) and false intervals (non-extremum)
    #

    # Find indexes with transfer from 0 to 1 (+1) and from 1 to 0 (-1)
    out = df[label_column].diff()
    out.iloc[0] = False  # Assume no change
    out = out.astype(int)

    # Find groups (intervals, starts-stops) and assign true-false label to them
    interval_no_column = 'interval_no'
    df[interval_no_column] = out.cumsum()

    #
    # For each group (with true-false label), compute their interval-wise score (using all or none principle)
    #

    # First, compute "score lower" (it will be used during interval-based aggregation)
    df[score_column + '_greater_than_threshold'] = (df[score_column] >= threshold)

    # Interval objects
    by_interval = df.groupby(interval_no_column)

    # Find interval label
    # Either 0 (all false) or 1 (at least one true - but must be all true)
    interval_label = by_interval[label_column].max()

    # Apply "all lower" function to each interval scores.
    # Either 0 (all lower) or 1 (at least one higher)
    interval_score = by_interval[score_column + '_greater_than_threshold'].max()
    interval_score.name = score_column

    # Compute into output
    interval_df = pd.concat([interval_label, interval_score], axis=1)
    interval_df = interval_df.reset_index(drop=False)

    return interval_df


# NOT USED
def generate_signals(df, models: dict):
    """
    Use predicted labels in the data frame to decide whether to buy or sell.
    Use rule-based approach by comparing the predicted scores with some thresholds.
    The decision is made for the last row only but we can use also previous data.

    TODO: In future, values could be functions which return signal 1 or 0 when applied to a row

    :param df: data frame with features which will be used to generate signals
    :param models: dict where key is a signal name which is also an output column name and value a dict of parameters of the model
    :return: A number of binary columns will be added each corresponding to one signal and having same name
    """

    # Define one function for each signal type.
    # A function applies a predicates by using the provided parameters and qualifies this row as true or false
    # TODO: Access to model parameters and row has to be rubust and use default values (use get instead of [])

    def all_higher_fn(row, model):
        keys = model.keys()
        for field, value in model.items():
            if row.get(field) >= value:
                continue
            else:
                return 0
        return 1

    def all_lower_fn(row, model):
        keys = model.keys()
        for field, value in model.items():
            if row.get(field) <= value:
                continue
            else:
                return 0
        return 1

    for signal, model in models.items():
        # Choose function which implements (knows how to generate) this signal
        fn = None
        if signal == "buy":
            fn = all_higher_fn
        elif signal == "sell":
            fn = all_lower_fn
        else:
            print("ERROR: Wrong use. Unexpected signal name.")

        # Model will be passed as the second argument (the first one is the row)
        df[signal] = df.apply(fn, axis=1, args=[model])

    return models.keys()


def _merge_data_sources(data_sources: List[Dict[str, Any]], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Merges multiple data sources into a single DataFrame based on the configured time column and frequency.

    Args:
        data_sources (List[Dict]): A list of dicts where each dict includes a DataFrame under the "df" key,
                                   along with metadata such as "column_prefix".
        config (Dict): Configuration dictionary containing:
            - "time_column" (str): Name of the timestamp column
            - "freq" (str): Pandas frequency string (e.g. "1h")
            - "merge_interpolate" (bool, optional): Whether to interpolate numeric columns

    Returns:
        pd.DataFrame: A merged DataFrame with a regular time index.
    """
    time_column = config["time_column"]
    freq = config["freq"]

    # Process each data source
    for source in data_sources:
        df = source["df"]

        # Ensure time column is the index
        if time_column in df.columns:
            df = df.set_index(time_column)
        elif df.index.name != time_column:
            raise ValueError(f"Missing or misaligned time index in source: expected '{time_column}'")

        # Apply column prefix if defined
        prefix = source.get("column_prefix", "")
        if prefix:
            df.columns = [
                f"{prefix}_{col}" if not col.startswith(f"{prefix}_") else col
                for col in df.columns
            ]

        # Store processed df and its time range
        source["df"] = df
        source["start"] = df.first_valid_index()
        source["end"] = df.last_valid_index()

    # Determine common time range for all sources
    common_start = min(src["start"] for src in data_sources)
    common_end = min(src["end"] for src in data_sources)

    # Create unified time index
    unified_index = pd.date_range(start=common_start, end=common_end, freq=freq)
    merged_df = pd.DataFrame(index=unified_index)
    merged_df.index.name = time_column

    # Join all data sources
    for source in data_sources:
        merged_df = merged_df.join(source["df"], how="left")

    # Interpolate numeric columns if enabled
    if config.get("merge_interpolate", False):
        numeric_cols = merged_df.select_dtypes(include=["float", "int"]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].interpolate()

    return merged_df
