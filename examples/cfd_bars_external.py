#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import time
from decimal import Decimal
from dotenv import load_dotenv
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from examples.backtest.components import risk_engine, rule_policy_strategy

# Load environment variables from .env file
load_dotenv("./.env")


if __name__ == "__main__":
    # Configure backtest engine
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO"),
        risk_engine=RiskEngineConfig(
            bypass=True,  # Example of bypassing pre-trade risk checks for backtests
        ),
    )

    # Build backtest engine
    engine = BacktestEngine(config=config)

    # Create a fill model (optional)
    fill_model = FillModel(
        prob_fill_on_limit=0.2,
        prob_fill_on_stop=0.95,
        prob_slippage=0.5,
        random_seed=42,
    )

    # Add a trading venue (multiple venues possible)
    OANDA = Venue("OANDA")
    engine.add_venue(
        venue=OANDA,
        oms_type=OmsType.HEDGING,  # Venue will generate position IDs
        account_type=AccountType.MARGIN,
        base_currency=USD,  # Standard single-currency account
        starting_balances=[Money(10_000, USD)],  # Single-currency or multi-currency accounts
        fill_model=fill_model,
        # bar_execution=True,  # If bar data should move the market (True by default)
        trade_execution=True,
        default_leverage=Decimal(25),
    )

    # Add instruments
    AUDUSD = TestInstrumentProvider.audusd_cfd()
    engine.add_instrument(AUDUSD)

    bar_type = BarType.from_str("AUDUSD.OANDA-15-MINUTE-LAST-EXTERNAL")
    
    # Set up wranglers
    wrangler = BarDataWrangler(
        bar_type=bar_type,
        instrument=AUDUSD,
    )

    # Add data
    # file_path = "../data/mt5/DXYm_H4_202001020000_202508012000.csv"
    file_path = "../data/mt5/audusd_15min_klines.csv.parquet"
    start_date = "2023-01-01 00:00:00"
    end_date = "2024-12-31 23:59:59"

    if ".parquet" in file_path:
        bars_df = pd.read_parquet(file_path)
    else:
        bars_df = pd.read_csv(file_path)
        
    if '' in bars_df.columns:
        bars_df.drop(columns=[''], inplace=True)
        
    if "Unnamed: 0" in bars_df.columns:
        bars_df.drop(columns=["Unnamed: 0"], inplace=True)
        
    if "tick_volume" in bars_df.columns:
        bars_df.drop(columns=["volume", "spread"], inplace=True)
        bars_df.rename(columns={"tick_volume": "volume"}, inplace=True)
        
    if "timestamp" not in bars_df.columns:
        bars_df["date"] = bars_df["date"] + " " + bars_df["time"]
        bars_df.rename(columns={"date": "timestamp"}, inplace=True)
        bars_df.drop(columns=["time"], inplace=True)

    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])
    bars_df.set_index("timestamp", inplace=True)
    # Filter data within the specified date range
    bars_df = bars_df.loc[start_date:end_date] if isinstance(start_date, str) and isinstance(end_date, str) else bars_df
    bars = wrangler.process(bars_df, default_volume=0.0, ts_init_delta=0)
    # Process the DataFrame into bar data
    engine.add_data(bars)


    # Instantiate and add your strategy
    strategies = [
        risk_engine(),
        rule_policy_strategy(bar_type=bar_type),
    ]
    engine.add_strategies(strategies=strategies)

    time.sleep(0.1)
    input("Press Enter to continue...")

    # Run the engine (from start to end of data)
    engine.run()

    # Optionally view reports
    with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        None,
        "display.width",
        300,
    ):
        print(engine.trader.generate_account_report(OANDA))
        print(engine.trader.generate_order_fills_report())
        print(engine.trader.generate_positions_report())

    # For repeated backtest runs make sure to reset the engine
    engine.reset()

    # Good practice to dispose of the object when done
    engine.dispose()
