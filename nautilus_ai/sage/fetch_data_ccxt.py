import time
import ccxt
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Union

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fetch_ohlcv_with_retries(exchange: ccxt.Exchange, symbol: str, timeframe: str, since: int, limit: int, max_retries: int = 3) -> List[List[Union[int, float]]]:
    """Fetch OHLCV data with retry logic."""
    for attempt in range(max_retries):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {timeframe} {symbol} OHLCV after {max_retries} attempts: {e}")
                raise
    return []

def load_existing_data(filename: Path) -> pd.DataFrame:
    """Load existing OHLCV data if available."""
    if filename.exists():
        return pd.read_csv(filename, parse_dates=["timestamp"], index_col="timestamp")
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

def scrape_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, since: int, until: int, limit: int, max_retries: int = 3) -> List[List[Union[int, float]]]:
    """Scrape historical OHLCV data from an exchange between two dates."""
    all_ohlcv: List[List[Union[int, float]]] = []
    
    while since < until:
        ohlcv: List[List[Union[int, float]]] = fetch_ohlcv_with_retries(exchange, symbol, timeframe, since, limit, max_retries)
        
        if not ohlcv:
            break
        
        since = ohlcv[-1][0] + 1  # Move forward in time
        all_ohlcv.extend(ohlcv)
        logger.info(f"{len(all_ohlcv)} {symbol} candles collected from {exchange.iso8601(all_ohlcv[0][0])} to {exchange.iso8601(all_ohlcv[-1][0])}")
    
    return all_ohlcv

def save_to_csv(filename: Path, data: pd.DataFrame) -> None:
    """Save OHLCV data to a CSV file, appending new data if necessary."""
    if filename.exists():
        data.to_csv(filename, mode='a', header=False)
    else:
        data.to_csv(filename)
    logger.info(f"Data saved to {filename}")

def scrape_and_save_candles(exchange_id: str, symbol: str, timeframe: str, since: Union[int, str], until: Union[int, str], limit: int, max_retries: int = 3, filename: Optional[str] = None, exchange_options: Optional[dict] = None) -> None:
    """Scrape OHLCV data and save to a CSV file, supporting resuming downloads."""

    if filename is None or len(filename) == 0:
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv".lower()
    
    exchange_options = exchange_options or {}
    exchange: ccxt.Exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True, 'options': exchange_options})
    
    if isinstance(since, str):
        since = exchange.parse8601(since)
    if not until:
        until = exchange.milliseconds()
    elif isinstance(until, str):
        until = exchange.parse8601(until)
    
    exchange.load_markets()
    file_path = Path("./ccxt/") / exchange_id / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing_data = load_existing_data(file_path)
    
    if not existing_data.empty:
        last_timestamp = existing_data.index[-1].timestamp() * 1000  # Convert to ms
        if last_timestamp > since:
            since = int(last_timestamp) + 1  # Resume from the next candle
    
    ohlcv = scrape_ohlcv(exchange, symbol, timeframe, since, until, limit, max_retries)
    
    if ohlcv:
        new_data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        new_data["timestamp"] = pd.to_datetime(new_data["timestamp"], unit='ms')
        if not new_data.empty:
            if not existing_data.empty:
                existing_data.reset_index(inplace=True)
                combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            else:
                combined_data = new_data.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            save_to_csv(file_path, combined_data)
            logger.info(f"Saved {len(new_data)} new candles from {new_data.iloc[0, 0]} to {new_data.iloc[-1, 0]} to {filename}")
        else:
            logger.warning("No new OHLCV data to save.")
    else:
        logger.warning("No new OHLCV data retrieved.")




# if __name__ == "__main__":
#     # scrape_and_save_candles(exchange_id="bybit", symbol="XRP/USDT", timeframe="5m", 
#     #                         since="2018-01-01T00:00:00Z", until="2025-06-01T23:59:59Z", limit=1000)
    
#     # Exchange options: | binanceusdm | bybit
#     # scrape_and_save_candles("binance", "BTC/USDT", "4h", "2011-01-01T00:00:00Z", "2023-12-01T00:00:00Z", 1000, exchange_options={'defaultType': 'future'})
#     # 1000RATS - 5m
#     # XRP - 15m
    
#     exchanges = {
#         'bybit': ("XRP/USDT", "5m"),
#         'binanceusdm': ("ETH/USDT", "5m"),
#         # 'binanceusdm': ("XRP/USDT", "5m"),
#     }
#     for exchange_id, value in exchanges.items():
#         scrape_and_save_candles(exchange_id=exchange_id, symbol=value[0], timeframe=value[1], 
#                             since="2018-01-01T00:00:00Z", until="2025-06-01T23:59:59Z", limit=1000)
#         time.sleep(0.1)