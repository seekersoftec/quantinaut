"""
Download quotes from Yahoo
"""

import yfinance as yf
import pandas as pd
import logging
from pathlib import Path

def setup_logger():
    """Configure logger for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

def download_yahoo_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download historical market data from Yahoo Finance."""
    logger.info(f"Downloading {symbol} data from {start} to {end} at {interval} intervals.")
    df = yf.download(symbol, start=start, end=end, interval=interval)
    if df.empty:
        logger.warning("No data retrieved. Check the symbol and date range.")
    return df

def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    """Save the DataFrame to a CSV file."""
    path = Path("./data/yahoo/")
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / filename
    df.to_csv(full_path)
    logger.info(f"Data saved to {full_path}")

if __name__ == "__main__":
    logger = setup_logger()
    symbol = "^GSPC"
    start_date = "1994-01-07"
    end_date = "2022-01-01"
    interval = "1d"
    
    df = download_yahoo_data(symbol, start_date, end_date, interval)
    if not df.empty:
        save_to_csv(df, f"{symbol}_data.csv")


# def load_data(
#     pair: str = "BTC-USD",
#     path: Optional[Path] = None,
#     **kwargs
# ) -> pd.DataFrame:
#     """
#     Loads historical price data either from Yahoo Finance or a local CSV file.

#     Parameters
#     ----------
#     pair : str, optional
#         Ticker symbol to download from Yahoo Finance, by default "BTC-USD".
#     path : Path or None, optional
#         Path to a CSV file containing historical data, by default None.
#     **kwargs : dict
#         Additional keyword arguments passed to `yfinance.Ticker().history()` if `path` is None,
#         or to `pandas.read_csv()` if `path` is provided.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame with historical price data.
#     """
#     if path is None:
#         pair_ticker = yf.Ticker(pair)
#         df = pair_ticker.history(**kwargs)
#         df.reset_index(inplace=True)
#     else:
#         df = pd.read_csv(path, parse_dates=["timestamp"], **kwargs)
#         if "Unnamed: 0" in df.columns:
#             df.drop(columns=["Unnamed: 0"], inplace=True)
#         df.set_index("timestamp", inplace=True)
#     return df