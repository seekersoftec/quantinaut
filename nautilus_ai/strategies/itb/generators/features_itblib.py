# generators/features_itblib.py

import numpy as np
from nautilus_ai.strategies.itb.generators.utils import add_area_ratio, add_linear_trends, add_past_aggregations, add_past_weighted_aggregations, to_diff
from nautilus_ai.strategies.itb.generators.generator import Generator, register_generator

@register_generator("itblib")
class ITBLib(Generator):
    def generate(self, df, last_rows=0, **kwargs):
        return _itblib(df, self.config, last_rows)



def _itblib(df, config: dict, last_rows: int = 0):
    """
    Generate derived features by adding them as new columns to the data frame.
    It is important that the same parameters are used for both training and prediction.

    Most features compute rolling aggregation. However, instead of absolute values, the difference
    of this rolling aggregation to the (longer) base rolling aggregation is computed.

    The window sizes are used for encoding feature/column names and might look like 'close_120'
    for average close price for the last 120 minutes (relative to the average base price).
    The column names are needed when preparing data for training or prediction.
    The easiest way to get them is to return from this function and copy and the
    corresponding config attribute.
    """
    use_differences = config.get('use_differences', True)
    base_window = config.get('base_window', True)
    windows = config.get('windows', True)
    functions = config.get('functions', True)

    features = []
    to_drop = []

    if use_differences:
        df['close'] = to_diff(df['close'])
        df['volume'] = to_diff(df['volume'])
        df['trades'] = to_diff(df['trades'])

    # close rolling mean. format: 'close_<window>'
    if not functions or "close_WMA" in functions:
        weight_column_name = 'volume'  # None: no weighting; 'volume': volume average
        to_drop += add_past_weighted_aggregations(df, 'close', weight_column_name, np.nanmean, base_window, suffix='', last_rows=last_rows)  # Base column
        features += add_past_weighted_aggregations(df, 'close', weight_column_name, np.nanmean, windows, '', to_drop[-1], 100.0, last_rows=last_rows)

    # close rolling std. format: 'close_std_<window>'
    if not functions or "close_STD" in functions:
        to_drop += add_past_aggregations(df, 'close', np.nanstd, base_window, last_rows=last_rows)  # Base column
        features += add_past_aggregations(df, 'close', np.nanstd, windows, '_std', to_drop[-1], 100.0, last_rows=last_rows)

    # volume rolling mean. format: 'volume_<window>'
    if not functions or "volume_SMA" in functions:
        to_drop += add_past_aggregations(df, 'volume', np.nanmean, base_window, suffix='', last_rows=last_rows)  # Base column
        features += add_past_aggregations(df, 'volume', np.nanmean, windows, '', to_drop[-1], 100.0, last_rows=last_rows)

    # Span: high-low difference. format: 'span_<window>'
    if not functions or "span_SMA" in functions:
        df['span'] = df['high'] - df['low']
        to_drop.append('span')
        to_drop += add_past_aggregations(df, 'span', np.nanmean, base_window, suffix='', last_rows=last_rows)  # Base column
        features += add_past_aggregations(df, 'span', np.nanmean, windows, '', to_drop[-1], 100.0, last_rows=last_rows)

    # Number of trades format: 'trades_<window>'
    if not functions or "trades_SMA" in functions:
        to_drop += add_past_aggregations(df, 'trades', np.nanmean, base_window, suffix='', last_rows=last_rows)  # Base column
        features += add_past_aggregations(df, 'trades', np.nanmean, windows, '', to_drop[-1], 100.0, last_rows=last_rows)

    # tb_base_av / volume varies around 0.5 in base currency. format: 'tb_base_<window>>'
    if not functions or "tb_base_SMA" in functions:
        df['tb_base'] = df['tb_base_av'] / df['volume']
        to_drop.append('tb_base')
        to_drop += add_past_aggregations(df, 'tb_base', np.nanmean, base_window, suffix='', last_rows=last_rows)  # Base column
        features += add_past_aggregations(df, 'tb_base', np.nanmean, windows, '', to_drop[-1], 100.0, last_rows=last_rows)

    # UPDATE: do not generate, because very high correction (0.99999) with tb_base
    # tb_quote_av / quote_av varies around 0.5 in quote currency. format: 'tb_quote_<window>>'
    #df['tb_quote'] = df['tb_quote_av'] / df['quote_av']
    #to_drop.append('tb_quote')
    #to_drop += add_past_aggregations(df, 'tb_quote', np.nanmean, base_window, suffix='', last_rows=last_rows)  # Base column
    #features += add_past_aggregations(df, 'tb_quote', np.nanmean, windows, '', to_drop[-1], 100.0, last_rows=last_rows)

    # Area over and under latest close price
    if not functions or "close_AREA" in functions:
        features += add_area_ratio(df, is_future=False, column_name="close", windows=windows, suffix = "_area", last_rows=last_rows)

    # Linear trend
    if not functions or "close_SLOPE" in functions:
        features += add_linear_trends(df, is_future=False, column_name="close", windows=windows, suffix="_trend", last_rows=last_rows)
    if not functions or "volume_SLOPE" in functions:
        features += add_linear_trends(df, is_future=False, column_name="volume", windows=windows, suffix="_trend", last_rows=last_rows)

    df.drop(columns=to_drop, inplace=True)

    return features
