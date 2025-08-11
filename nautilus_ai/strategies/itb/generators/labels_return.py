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
            Optional keyword arguments:
            - shift: int (default=-1) → number of periods to look ahead (negative for future returns)
            - use_native_pct_change: bool (default=False) → if True, uses Polars' built-in pct_change(periods=shift)

        Returns
        -------
        Tuple[pl.DataFrame, List[str]]
            DataFrame with new 'return' column and list of the new column names.
        """
        shift = kwargs.get("shift", -1)
        use_native_pct_change = kwargs.get("use_native_pct_change", False)

        if use_native_pct_change:
            df = df.with_columns(
                pl.col("close").pct_change(periods=shift).alias("return")
            )
        else:
            df = df.with_columns(
                ((pl.col("close").shift(-shift) - pl.col("close")) / pl.col("close")).alias("return")
                if shift < 0 else
                ((pl.col("close") - pl.col("close").shift(shift)) / pl.col("close").shift(shift)).alias("return")
            )

        return df, ["return"]
