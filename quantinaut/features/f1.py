# quantinaut/features/f1.py
import numpy as np
from typing import Any, Dict, Optional
from quantinaut.features.feature import Feature


class F1(Feature):
    """
    Feature generator for extracting ATR, VWAP, and log-transformed prices from context.

    This class produces a dictionary of features including Average True Range (ATR),
    Volume Weighted Average Price (VWAP), and log-transformed price values for each sample.
    """
    def __init__(self):
        super().__init__()  

    def generate(self, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate ATR, VWAP, and log-transformed price features from the input context.

        Args:
            ctx (Optional[Dict[str, Any]]):
                Input context for a single sample, expected to contain keys:
                - 'atr': Average True Range value
                - 'vwap': Volume Weighted Average Price value
                - 'prices': List of price values

        Returns:
            Dict[str, Any]:
                Dictionary with keys 'atr', 'vwap', and 'last_log_ret',
                where 'last_log_ret' is the log-transformed value of the i-th price.
        """
        features = {}
        if ctx is None:
            return features

        features['atr'] = ctx.get("atr", np.nan)
        features['vwap'] = ctx.get("vwap", np.nan)
        
        prices = ctx.get("prices", [])
        # prices_log = np.log(np.array(prices) + 1e-10)  # Avoid log(0)
        features['last_log_ret'] = float(np.log(prices[-1] / prices[-2])) if len(prices) > 1 else np.nan
        return features
