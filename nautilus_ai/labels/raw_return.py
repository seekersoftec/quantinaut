# nautilus_ai/labels/raw_return.py
"""
============================
Raw Returns Labeling Method

Most basic form of labeling based on raw return of each observation relative to its previous value.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union
from nautilus_ai.labels.label import Label


class RawReturn(Label):
    """
    Raw returns labeling method.

    This is the most basic and ubiquitous labeling method used as a precursor to almost any kind of financial data
    analysis or machine learning. User can specify simple or logarithmic returns, numerical or binary labels, a
    resample period, and whether returns are lagged to be forward looking.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Time-indexed price data on stocks with which to calculate return.
    binary : bool
        If False, will return numerical returns. If True, will return the sign of the raw return.
    logarithmic : bool
        If False, will calculate simple returns. If True, will calculate logarithmic returns.
    resample_by : str or None
        If not None, the resampling period for price data prior to calculating returns. 'B' = per
        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
        For full details see `here.
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    lag : bool
        If True, returns will be lagged to make them forward-looking.

    Returns
    -------
    pd.Series or pd.DataFrame
        Raw returns on market data. User can specify whether returns will be based on
        simple or logarithmic return, and whether the output will be numerical or categorical.
    """
    def __init__(self, binary: bool = False, logarithmic: bool = False, resample_by: Optional[str] = None, lag: bool = False):
        super().__init__() 
        self.binary = binary
        self.logarithmic = logarithmic
        self.resample_by = resample_by
        self.lag = lag
        
    def transform_one(self, X: Optional[Dict[str, Any]] = None) -> Union[int, float]:
        """
        Transform input features for a single sample into a label.

        Parameters
        ----------
            X (Optional[Dict[str, Any]]):
                Input features for one sample.

        Returns:
            Any:
                The label value (e.g., int for classification, float for regression).
                Returns 0 if no prediction is made.
        """
        returns = self.transform_many(X)
        return returns[-1] if not returns.empty else 0.0  # Return last value or 0 if empty

    def transform_many(self, X: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Transform input features for multiple samples into labels.

        Parameters
        ----------
            X (Optional[Dict[str, Any]]):
                Input features for multiple samples (batched or iterable).

        Returns:
            pd.Series:
                The predicted label values (e.g., list or array for batch prediction).
        """
        prices = X.get("prices", []) if X is not None else []
        prices = pd.Series(prices)

        # Apply resample, if applicable.
        if self.resample_by is not None:
            prices = prices.resample(self.resample_by).last()

        # Get return per period.
        if self.logarithmic:  # Log returns
            if self.lag:
                returns = np.log(prices).diff().shift(-1)
            else:
                returns = np.log(prices).diff()
        else:  # Simple returns
            if self.lag:
                returns = prices.pct_change(periods=1).shift(-1)
            else:
                returns = prices.pct_change(periods=1)

        # Return sign only if categorical labels desired.
        if self.binary:
            returns = returns.apply(np.sign)

        return returns
    