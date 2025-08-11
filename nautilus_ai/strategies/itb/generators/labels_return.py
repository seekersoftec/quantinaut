# generators/labels_return.py
import polars as pl
from typing import List, Tuple
from nautilus_ai.strategies.itb.generators.generator import Generator, GeneratorType, register_generator


@register_generator("return")
class Return(Generator):
    """
    A generator that calculates the percentage change of a closing price.
    """
    def __init__(self, config):
        super().__init__(config)
        # Set the generator type to LABEL upon initialization
        self.set_generator_type(GeneratorType.LABEL)
        self.generator_id = f"{GeneratorType.LABEL}-return"

    def generate(self, df: pl.DataFrame, **kwargs) -> Tuple[pl.DataFrame, List[str]]:
        """
        Generates a 'return' column, with a configurable look-ahead period.

        Parameters
        ----------
        df : pl.DataFrame
            The input DataFrame containing the 'close' price column.
        **kwargs
            Optional keyword arguments, including `shift` to specify the look-ahead period.
            Defaults to -1 for a one-period-ahead percentage change.

        Returns
        -------
        Tuple[pl.DataFrame, List[str]]
            The DataFrame with the new 'return' column and a list of the new column name.
        """
        # TODO: Verify and fix this logic, should shift be used in pct_change itself instead of a 1 step process?
        shift = kwargs.get("shift", -1)
        df = df.with_columns(
            (pl.col("close").pct_change().shift(shift)).alias("return")
        )
        return df, ["return"]
