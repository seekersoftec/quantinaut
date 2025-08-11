# generators/signals_smoothen.py

from nautilus_ai.strategies.itb.generators.generator import Generator, register_generator


@register_generator("smoothen")
class Smoothen(Generator):
    """
    Smoothen several columns and rows. Used for smoothing scores.

    The following operations are applied:
        - find average of the specified input columns (row-wise)
        - find moving average with the specified window
        - apply threshold to source buy/sell column(s) according to threshold parameter(s) by producing a boolean column

    Notes:
        - Input point-wise scores in buy and sell columns are always positive
    """
    def generate(self, df, last_rows=0, **kwargs):
        columns = self.config.get('columns')
        if not columns:
            raise ValueError(f"The 'columns' parameter must be a non-empty string. {type(columns)}")
        elif isinstance(columns, str):
            columns = [columns]

        # TODO: check that all columns exist
        #if columns not in df.columns:
        #    raise ValueError(f"{columns} do not exist  in the input data. Existing columns: {df.columns.to_list()}")

        # Average all buy and sell columns
        out_column = df[columns].mean(skipna=True, axis=1)

        # Apply thresholds (if specified) and binarize the score
        point_threshold = self.config.get("point_threshold")
        if point_threshold:
            out_column = out_column >= point_threshold

        # Moving average
        window = self.config.get("window")
        if isinstance(window, int):
            out_column = out_column.rolling(window, min_periods=window // 2).mean()
        elif isinstance(window, float):
            out_column = out_column.ewm(span=window, min_periods=window // 2, adjust=False).mean()

        names = self.config.get('names')
        if not isinstance(names, str):
            raise ValueError(f"'names' parameter must be a non-empty string. {type(names)}")

        df[names] = out_column

        return df, [names]
