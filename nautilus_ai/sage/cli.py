import json
import re
import click
import fire 
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from typing import Union
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from nautilus_ai.common import handle_config


# from common.model_store import *

np.random.seed(100)


PACKAGE_ROOT = Path(__file__).parent.parent


class State:
    """Globally visible variables."""
    #
    # Constant configuration parameters
    #
    config = {
        # Venue 
        "venue": "ccxt",  # mt5, ccxt
        
        # MetaTrader5
        "mt5_account_id": "",
        "mt5_password": "",
        "mt5_server": "",
        # MetaTrader5 specific parameters
        "mt5_params": {},  # Additional parameters for MT5 connection
        
        # CCXT
        "ccxt_exchange": "",  # e.g., 'binance', 'bitfinex', etc.
        "ccxt_api_key": "",
        "ccxt_api_secret": "",
        # CCXT specific parameters
        "ccxt_params": {},  # Additional parameters for CCXT exchange connection

        #
        # Conventions for the file and column names
        #
        "merge_file_name": "data.csv",
        "feature_file_name": "features.csv",
        "matrix_file_name": "matrix.csv",
        "predict_file_name": "predictions.csv",  # predict, predict-rolling
        "signal_file_name": "signals.csv",
        "signal_models_file_name": "signal_models",

        "model_folder": "models",  # Folder for all models

        "time_column": "timestamp",

        # File locations
        "data_folder": "./data",  # Location for all source and generated data/models

        # ==============================================
        # === DOWNLOADER, MERGER and (online) READER ===

        # Symbol determines sub-folder and used in other identifiers
        "symbol": "BTCUSDT",  # BTCUSDT ETHUSDT ^gspc EURUSD

        # This parameter determines time raster (granularity) for the data
        # It is pandas frequency
        "timeframe": "1min",

        # This list is used for downloading and then merging data
        # "folder" is symbol name for downloading. prefix will be added column names during merge
        "data_sources": [],

        # ==========================
        # === FEATURE GENERATION ===

        # What columns to pass to which feature generator and how to prefix its derived features
        # Each executes one feature generation function applied to columns with the specified prefix
        "feature_sets": [],

        # ========================
        # === LABEL GENERATION ===

        "label_sets": [],

        # ===========================
        # === MODEL TRAIN/PREDICT ===
        #     predict off-line and on-line

        "label_horizon": 0,  # This number of tail rows will be excluded from model training
        "train_length": 0,  # train set maximum size. algorithms may decrease this length

        # List all features to be used for training/prediction by selecting them from the result of feature generation
        # The list of features can be found in the output of the feature generation (but not all must be used)
        # Currently the same feature set for all algorithms
        "train_features": [],

        # Labels to be used for training/prediction by all algorithms
        # List of available labels can be found in the output of the label generation (but not all must be used)
        "labels": [],

        # Algorithms and their configurations to be used for training/prediction
        "algorithms": [],

        # ===============
        # === SIGNALS ===

        "signal_sets": [],

        # =====================
        # === NOTIFICATIONS ===

        "score_notification_model": {},
        "diagram_notification_model": {},

    }



class SageCli:
    """
    📊 Sage CLI — Intelligent Trading Automation

    A unified control interface for managing data, ML models/pipelines, and automated trading operations.

    🛠️  Core Commands: \n
    ──────────────────────────────────────────────\n
    ⚙️  sage config     → Load and inspect configuration files \n
    🔗  sage merge      → Merge multiple data sources \n
    🧠  sage generate   → Generate features, labels and signals \n
    🏋️  sage train      → Train ML models \n
    🤖  sage predict    → Run predictions (including rolling forecasts) \n
    🧪  sage simulate   → Simulate/backtest strategies \n
    💸  sage live       → Execute trades in live/paper mode \n
    """
    def __init__(self):
        self._config = {}

    def config(self, path: Path = Path('./'), name: str = 'workflow.yml'):
        """
        ⚙️  Load and display the contents of a configuration file.

        Supports: JSON, and YAML files.

        Parameters
        ----------
            - path : Path : Path to the configuration file directory
            - name : str : Name of the configuration file

        Example:
            sage config --path configs --name workflow.json
        """
        from nautilus_ai.sage.cli import State
        
        try:
            file = path / name
            if not file.exists():
                click.secho(f"❌ Configuration file not found: {file}", fg='red')
                return
            
            if not file.suffix in ['.yml', '.yaml', '.json']:
                click.secho(f"❌ Unsupported file format: {file.suffix}. Supported formats are .yml, .yaml, .json", fg='red')
                return

            config = handle_config(file)
            State.config.update(config)
            self._config = State.config
        except Exception as e:
            click.secho(f"❌ Failed to load config: {e}", fg='red')
            raise SystemExit(1)

        click.secho(f"\n✅ Loaded configuration from: {file}\n", fg='green')
        click.echo(json.dumps(self._config, indent=4, default=str))

    def fetch(self, path: Path = Path('./'), name: str = 'workflow.yml'):
        """
        🧠 Fetch data from one or multiple configured source(s)(e.g., MT5, CCXT).

        Reads the configuration file to determine data sources and fetches data accordingly.
        """
        
        self.config(path, name)

        if not self._config.get("venue"):
            click.secho("❌ ERROR: No venue specified in the configuration.", fg="red")
            return
        
        data_sources = self._config.get("data_sources", [])
        if not data_sources:
            click.secho("❌ ERROR: No data sources defined in the configuration.", fg="red")
            return
        
        click.secho("📥 Fetching data from configured sources...", fg="blue")
        
        if "mt5" in self._config["venue"]:
            from nautilus_ai.sage.fetch_data_mt5 import connect_mt5, scrape_and_save_candles
            
            if not connect_mt5(self._config["mt5_account_id"], self._config["mt5_password"], self._config["mt5_server"]):
                click.secho("❌ Failed to connect to MT5", fg="red")
                return
            
            scrape_and_save_candles(
                data_sources,
                mt5_account_id=self._config["mt5_account_id"],
                mt5_password=self._config["mt5_password"],
                mt5_server=self._config["mt5_server"],  
                timeframe=self._config["timeframe"],
                time_column=self._config["time_column"],
                since="2011-01-01T00:00:00Z", 
                until="2024-01-01T23:59:59Z",
            )

        elif "ccxt" in self._config["venue"]:
            from nautilus_ai.sage.fetch_data_ccxt import scrape_and_save_candles

            try:
                for ds in data_sources:
                    symbol = str(ds.get("symbol"))
                    data_type = str(ds.get("type")).lower()

                    if not symbol:
                        click.secho(f"❌ ERROR: {symbol} is not specified in data_sources.")
                        continue

                    if data_type != "ticks":
                        scrape_and_save_candles(exchange_id=self._config["ccxt_exchange"], 
                                            symbol=symbol, 
                                            timeframe=self._config["timeframe"],
                                            since="2018-01-01T00:00:00Z", 
                                            until="2025-06-01T23:59:59Z", 
                                            limit=1000,
                                            exchange_options=self._config["ccxt_params"]
                                        )
            except Exception:
                click.secho("❌ Failed to connect to CCXT", fg="red")
                return
    
        else:
            click.secho("❌ ERROR: Unsupported venue specified in the configuration.", fg="red")
            return

        click.secho("✅ Data fetching completed successfully!", fg="green")


    

def run():
    fire.Fire(SageCli)

if __name__ == "__main__":
    run()