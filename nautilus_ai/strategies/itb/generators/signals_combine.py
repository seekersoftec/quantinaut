# generators/signals_combine.py

from nautilus_ai.strategies.itb.generators.generator import Generator, register_generator
from nautilus_ai.strategies.itb.generators.utils import combine_scores_difference, combine_scores_relative


@register_generator("combine")
class Combine(Generator):
    """
    ML algorithms predict score which is always positive and typically within [0,1].
    One score for price growth and one score for price fall. This function combines pairs
    of such scores and produce one score within [-1,+1]. Positive values mean growth
    and negative values mean fall of price.
    """
    def generate(self, df, last_rows=0, **kwargs):
        columns = self.config.get('columns')
        if not columns:
            raise ValueError(f"The 'columns' parameter must be a non-empty string. {type(columns)}")
        elif not isinstance(columns, list) or len(columns) != 2:
            raise ValueError(f"'columns' parameter must be a list with buy column name and sell column name. {type(columns)}")

        up_column = columns[0]
        down_column = columns[1]

        out_column = self.config.get('names')

        if self.config.get("combine") == "relative":
            combine_scores_relative(df, up_column, down_column, out_column)
        elif self.config.get("combine") == "difference":
            combine_scores_difference(df, up_column, down_column, out_column)
        else:
            # If buy score is greater than sell score then positive buy, otherwise negative sell
            df[out_column] = df[[up_column, down_column]].apply(lambda x: x[0] if x[0] >= x[1] else -x[1], raw=True, axis=1)

        # Scale the score distribution to make it symmetric or normalize
        # Always apply the transformation to buy score. It might be in [0,1] or [-1,+1] depending on combine parameter
        if self.config.get("coefficient"):
            df[out_column] = df[out_column] * self.config.get("coefficient")
        if self.config.get("constant"):
            df[out_column] = df[out_column] + self.config.get("constant")

        return df, [out_column]


