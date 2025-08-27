import os
import re
import time
import itertools
from pathlib import Path
from decimal import Decimal
from typing import Optional
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, InstrumentProviderConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, BookType, OmsType, TriggerType
from nautilus_trader.model.identifiers import ClientId, TraderId, InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Money, Currency
from nautilus_trader.model.currencies import USDT
from nautilus_trader.persistence.wranglers import BarDataWrangler
# from nautilus_trader.persistence.wranglers_v2 import BarDataWranglerV2
from nautilus_trader.persistence.loaders import CSVBarDataLoader

from examples.backtest.components import risk_engine, rule_policy_strategy
from examples.backtest.providers import TestInstrumentProvider



class Backtest:
    """
    Backtest
            
    Parameters:
    -----------
        symbol (list[Symbol]): The trading symbols.
        venue (list[Venue]): The trading venues.
    """
    def __init__(self, symbols: list[Symbol] = [Symbol("BTCUSDT-PERP")], venues: list[Venue] = [Venue("BINANCE")], skip_errors: bool = False):
        self.symbols = symbols
        self.venues = venues
        self.skip_errors = skip_errors
        self.default_quote_currency = USDT
        self.engine: Optional[BacktestEngine] = None
        self.default_files_to_remove = [
            "c3_metrics_log.csv",
            "n3_metrics_log.csv",
            "p3_metrics_log.csv",
            "sst_metrics_log.csv",
            "trendy_metrics_log.csv",
            "swing_cross_metrics_log.csv",
            # "*_metrics_log.csv",
            # "account_report.csv",
            # "order_fills_report.csv",
            # "positions_report.csv"
        ]
    
    def clean_up(self, files: list = []):
        for venue in self.venues:
            files.append(f"account_report-{venue}.csv")
            files.append(f"order_fills_report-{venue}.csv")
            files.append(f"positions_report-{venue}.csv")
            
        all_files = self.default_files_to_remove + files
        for file_name in all_files:
            if os.path.exists(file_name):
                os.remove(file_name)
    
    def configure_engine(self) -> BacktestEngine:
        """Configure and return a BacktestEngine instance."""
        config = BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(
                log_level="INFO",
                log_colors=True,
                use_pyo3=False,
            ),
        )
        self.engine = BacktestEngine(config=config)
        return self.engine
    
    def setup_instruments(self) -> TestInstrumentProvider:
        """Initialize and return a TestInstrumentProvider with instruments added."""
        provider = TestInstrumentProvider(config=InstrumentProviderConfig(load_all=True, log_warnings=False))
        raw_instruments = [
            # provider.btcusdt_binance(),
            # provider.ethusdt_binance(),
            # provider.adausdt_binance(),
            # provider.adabtc_binance(),
            provider.btcusdt_perp_binance(),
            provider.ethusdt_perp_binance(),
            provider.solusdt_perp_binance(),
            provider.onethousandrats_perp_binance(),
            provider.xrpusdt_linear_bybit(),
            # provider.xbtusd_bitmex(),
        ]
        for instrument in raw_instruments:
            try:
                provider.add_currency(currency=instrument.base_currency)
            except AttributeError:
                provider.add_currency(currency=instrument.underlying)
                
            provider.add_currency(currency=instrument.quote_currency)
            provider.add(instrument=instrument)
        return provider

    def add_venue_to_engine(self, venue: Venue, quote_currency: Currency = USDT) -> None:
        """Add a trading venue to the engine."""
        self.engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            book_type=BookType.L1_MBP,
            account_type=AccountType.MARGIN,
            base_currency=None,
            starting_balances=[Money(1_000.0, quote_currency)],
            trade_execution=True,
            default_leverage=Decimal(5),
        )
    
    
    def add_data(
        self,
        instrument: Instrument,
        data_sub_str: str = "15-MINUTE-LAST-EXTERNAL",
        start_date: str = "2020-01-01 00:00:00",
        end_date: str = "2023-12-31 23:59:59"
    ):
        """
        Loads and processes historical bar data for a given instrument and adds it to the backtest engine.

        Parameters:
        -----------
            engine (BacktestEngine): The backtest engine instance.
            instrument (Instrument): The instrument for which data is to be loaded.
            data_sub_str (str): The data suffix to identify the bar type.
            start_date (str): The start date for the data range.
            end_date (str): The end date for the data range.

        Returns:
            tuple: The updated engine and the bar type used.
        """

        spans = {
            "ms": "MILLISECOND",
            "s": "SECOND",
            "m": "MINUTE",
            "h": "HOUR",
            "D": "DAY",
            "W": "WEEK",
            "M": "MONTH",
            "tick": "TICK",
            "tick_imbalance": "TICK_IMBALANCE",
            "tick_runs": "TICK_RUNS",
            "volume": "VOLUME",
            "volume_imbalance": "VOLUME_IMBALANCE",
            "volume_runs": "VOLUME_RUNS",
            "value": "VALUE",
            "value_imbalance": "VALUE_IMBALANCE",
            "value_runs": "VALUE_RUNS",
        }
        
        venues = {
            "BINANCE": "binanceusdm",
            "BYBIT": "bybit",
        }
        venue = venues.get(str(instrument.venue).upper(), None)
        if venue is None:
            error = ValueError(f"Venue {venue} not found")
            if not self.skip_errors:
                raise error
            else:
                print(error)
                print("Using binance as the default...")
                venue = venues.get("BINANCE")

        # Parse the timeframe value and unit from data_sub_str
        match = re.match(r"(\d+)-([A-Za-z_]+)-", data_sub_str)
        if not match:
            raise ValueError(f"Invalid data_sub_str format: {data_sub_str}")

        timeframe_value = int(match.group(1))
        timeframe_unit_key = match.group(2).upper()

        # Map the extracted unit key to the full span name if available
        timeframe_unit = None
        for key, value in spans.items():
            if value.upper() == timeframe_unit_key:
                timeframe_unit = key
                break
            
        if timeframe_unit is None:
            raise ValueError(f"Unknown timeframe unit key: {timeframe_unit_key}")

        # Construct symbol pair and filename based on instrument and timeframe
        symbol_pair = f"{instrument.base_currency}/{instrument.quote_currency}"
        filename_timeframe = f"{timeframe_value}{timeframe_unit}"
        filepath = Path(f"../../data/ccxt/{venue}/{symbol_pair.replace('/', '_').lower()}_{filename_timeframe}.csv")

        # Load historical bar data from CSV
        try:
            bars_df = CSVBarDataLoader.load(filepath, index_col="timestamp")
        except FileNotFoundError as e:
            filepath = filepath.with_suffix(".csv.parquet")
            bars_df = pd.read_parquet(filepath)
            if '' in bars_df.columns:
                bars_df.drop(columns=[''], inplace=True)
                
            if "tick_volume" in bars_df.columns:
                bars_df.drop(columns=["volume", "spread"], inplace=True)
                bars_df.rename(columns={"tick_volume": "volume"}, inplace=True)
                
            if "timestamp" not in bars_df.columns:
                bars_df["date"] = bars_df["date"] + " " + bars_df["time"]
                bars_df.rename(columns={"date": "timestamp"}, inplace=True)
                bars_df.drop(columns=["time"], inplace=True)

            bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])
            bars_df.set_index("timestamp", inplace=True)
        except Exception as e:
            raise e
        
        if "Unnamed: 0" in bars_df.columns:
            bars_df.drop(columns=["Unnamed: 0"], inplace=True)
        
        # Filter data within the specified date range
        bars_df = bars_df.loc[start_date:end_date] if isinstance(start_date, str) and isinstance(end_date, str) else bars_df

        # Define the bar type
        bar_type = BarType.from_str(f"{instrument.id.value}-{data_sub_str}")

        # Initialize the data wrangler
        # wrangler = BarDataWranglerV2(bar_type=bar_type, price_precision=instrument.price_precision, size_precision=instrument.size_precision)
        wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)

        # Process the DataFrame into bar data
        bars = wrangler.process(bars_df, default_volume=0.0, ts_init_delta=0)
        # bars = wrangler.from_pandas(bars_df, default_volume=0.0, ts_init_delta=0)

        # Add the processed bar data to the engine
        self.engine.add_data(bars)

        return bar_type

    def generate_reports(self, venue: Venue) -> None:
        """Generate and save account, order fills, and positions reports."""
        with pd.option_context(
            "display.max_rows", 100,
            "display.max_columns", None,
            "display.width", 300,
        ):
            print(self.engine.trader.generate_account_report(venue))
            # print(engine.trader.generate_order_fills_report())
            # print(engine.trader.generate_positions_report())

        pd.DataFrame(self.engine.trader.generate_account_report(venue)).to_csv(f"account_report-{venue}.csv")
        pd.DataFrame(self.engine.trader.generate_order_fills_report()).to_csv(f"order_fills_report-{venue}.csv")
        pd.DataFrame(self.engine.trader.generate_positions_report()).to_csv(f"positions_report-{venue}.csv")

    def run(self, 
            entry_data_sub_str: str = "5-MINUTE-LAST-EXTERNAL",
            anchor_data_sub_strs: list[str] = ["15-MINUTE-LAST-EXTERNAL", "4-HOUR-LAST-EXTERNAL"], 
            start_date: str = "2020-01-01 00:00:00",
            end_date: str = "2023-12-31 23:59:59"
    ):
        """
            Run the backtest with the specified parameters.
            
            Parameters:
            -----------
                symbol (list[Symbol]): The trading symbols.
                venue (list[Venue]): The trading venues.
                entry_data_sub_str (str): The data suffix for the entry data.
                anchor_data_sub_strs (list[str]): List of data suffixes for additional data. 
                start_date (str): The start date for the backtest.
                end_date (str): The end date for the backtest.
            
            Notes:
            ------
            - Typically, anchor timeframes are derived by multiplying the base timeframe by a factor of 4 or 6.
            - Only supports external data for now.
        """
        self.clean_up() 
        self.configure_engine()
        
        for venue in self.venues:
            self.add_venue_to_engine(venue, self.default_quote_currency)
        
        strategies = []
        instrument_provider: TestInstrumentProvider = self.setup_instruments()
        instrument_ids = [InstrumentId(symbol=symbol, venue=venue) for symbol, venue in itertools.product(self.symbols, self.venues)]
        for instrument_id in instrument_ids:
            instrument = instrument_provider.find(instrument_id)
            if instrument is None:
                error = RuntimeError(f"Unable to find instrument {instrument_id}")
                if not self.skip_errors:
                    raise error
                else:
                    print(error)
                    continue
                
            self.engine.add_instrument(instrument)
        
            for data_sub_str in anchor_data_sub_strs + [entry_data_sub_str]:
                _ = self.add_data(instrument, data_sub_str=data_sub_str, start_date=start_date, end_date=end_date)

            entry_data_type = BarType.from_str(f"{instrument.id.value}-{entry_data_sub_str}")
            rule_policy = rule_policy_strategy(entry_data_type)
            strategies.append(rule_policy)

        # Add strategies to the engine
        _risk_engine = risk_engine()
        self.engine.add_strategies(strategies=[_risk_engine] + strategies)

        time.sleep(0.1)
        self.engine.run()
        for venue in self.venues:
            self.generate_reports(venue)
        
        time.sleep(0.1)
        self.engine.reset()
        self.engine.dispose()
            


if __name__ == "__main__":
    SYMBOLS = [Symbol("BTCUSDT-PERP")] # Symbol("ETHUSDT-PERP"), Symbol("XRPUSDT-LINEAR")  | BTC, ETH, SOL, 1000RATS, AVAX, ADA, PEPE, MASK | 
    VENUES = [Venue("BINANCE")] 
    entry_data_sub_str = "15-MINUTE-LAST-EXTERNAL" # "15-MINUTE-LAST-EXTERNAL" | "4-HOUR-LAST-EXTERNAL"
    anchor_data_sub_strs = [] # ["30-MINUTE-LAST-EXTERNAL", "1-HOUR-LAST-EXTERNAL" , "4-HOUR-LAST-EXTERNAL" ] # "15-MINUTE-LAST-EXTERNAL", "30-MINUTE-LAST-EXTERNAL", "1-DAY-LAST-EXTERNAL" | change the 1 hour to 2 hours => 30 * 4 = 120 minutes => 2 hours 
    start_date = "2023-09-01 00:00:00" # 2020-2021 => mostly buys | 2022-2023 => mostly sells | 2023-2024 => no clear trend
    end_date = "2025-06-01 00:00:00"
    
    backtest = Backtest(SYMBOLS, VENUES, True)
    backtest.run(entry_data_sub_str, anchor_data_sub_strs, start_date, end_date)

   
"""
- Start with BTC/ETH/SOL for stability and reliable backtest results.

- Add AVAX and ADA to assess behavior in mid-cap alt market regimes.

- Include PEPE and MASK for high-volatility, edge-seeking signals—but expect more noise.
"""