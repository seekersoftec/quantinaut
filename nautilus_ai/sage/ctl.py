import json
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from nautilus_ai.common import handle_config

np.random.seed(100)


@click.group()
def sage_ctl():
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
    pass


@sage_ctl.command(name="config", help="⚙️  Load and display the contents of a configuration file")
@click.option(
    '--file', '-f',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help='Path to the configuration file (YAML/TOML/JSON)'
)
def read_config(file):
    """
    Load and display the contents of a configuration file.
    
    Supports: JSON, YAML, TOML, INI, and NumPy files.

    Example:
        sage config -f configs/example.json
    """
    try:
        config = handle_config(file)
    except Exception as e:
        click.secho(f"❌ Failed to load config: {e}", fg='red')
        raise SystemExit(1)

    click.secho(f"\n✅ Loaded configuration from: {file}\n", fg='green')
    click.echo(json.dumps(config, indent=4, default=str))


# Data Preparation (Fetch data, Merge, etc.)

"""
Fetch data from one or multiple source(s).
"""
@sage_ctl.command(name="fetch")
@click.option(
    '--file', '-f',
    type=click.Path(exists=True, readable=True, dir_okay=False),
    required=True,
    help="Path to the configuration file"
)
def fetch_data(file):
    """
    🧠 Fetch data from configured sources (e.g., MT5, CCXT).

    Reads the configuration file to determine data sources and fetches data accordingly.
    """
    from nautilus_ai.sage.state import State

    config = handle_config(file)
    State.config.update(config)
    config = State.config

    if not config.get("venue"):
        click.secho("❌ ERROR: No venue specified in the configuration.", fg="red")
        return
    
    data_sources = config.get("data_sources", [])
    if not data_sources:
        click.secho("❌ ERROR: No data sources defined in the configuration.", fg="red")
        return
    
    click.secho("📥 Fetching data from configured sources...", fg="blue")
    
    if "mt5" in config["venue"]:
        from nautilus_ai.sage.fetch_data_mt5 import connect_mt5, scrape_and_save_candles
        
        if not connect_mt5(config["mt5_account_id"], config["mt5_password"], config["mt5_server"]):
            click.secho("❌ Failed to connect to MT5", fg="red")
            return
        
        scrape_and_save_candles(
            data_sources,
            mt5_account_id=config["mt5_account_id"],
            mt5_password=config["mt5_password"],
            mt5_server=config["mt5_server"],  
            timeframe=config["timeframe"],
            time_column=config["time_column"],
            since="2011-01-01T00:00:00Z", 
            until="2024-01-01T23:59:59Z",
        )

    elif "ccxt" in config["venue"]:
        from nautilus_ai.sage.fetch_data_ccxt import scrape_and_save_candles
        
        if not connect_ccxt(**config["ccxt"]):
            click.secho("❌ Failed to connect to CCXT", fg="red")
            return
    
    else:
        click.secho("❌ ERROR: Unsupported venue specified in the configuration.", fg="red")
        return

    click.secho("✅ Data fetching completed successfully!", fg="green")


"""
Create one output file from multiple input data files. 
"""
@sage_ctl.command(name="merge")
@click.option(
    '--file', '-f',
    type=click.Path(exists=True, readable=True, dir_okay=False),
    required=True,
    help="Path to the configuration file"
)
def merge(file):
    """
    🧩 Merge multiple data sources into a unified dataframe.

    Reads CSV data sources defined in the config, aligns them on a regular time index, 
    interpolates if specified, and stores a merged output file.
    """
    from nautilus_ai.sage.helpers import merge_data_sources
    from nautilus_ai.sage.state import State

    config = handle_config(file)
    State.config.update(config)
    config = State.config
    time_column = config["time_column"]
    data_sources = config.get("data_sources", [])

    if not data_sources:
        click.secho("❌ ERROR: No data sources defined.", fg="red")
        return

    now = datetime.now()
    data_path = Path(config["data_folder"]) / Path(config["venue"])
    symbol = config["symbol"]
    timeframe = config["timeframe"]

    is_train = config.get("train")
    if is_train:
        window_size = config.get("train_length")
    else:
        window_size = config.get("predict_length")
    features_horizon = config.get("features_horizon")
    if window_size:
        window_size += features_horizon

    for ds in data_sources:
        symbol = ds.get("symbol")
        data_type = ds.get("type")
        file_path = (data_path / f"{symbol}_{timeframe}_{data_type}".lower()).with_suffix(".csv")
        if not file_path.suffix:
            file_path = file_path.with_suffix(".csv")

        if not file_path.exists():
            click.secho(f"❌ File not found: {file_path}", fg="red")
            return

        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
        else:
            click.secho(f"❌ Unsupported input format: {file_path.suffix}", fg="red")
            return
        click.secho(f"📄 Loaded {file_path} with {len(df)} rows", fg="green")
        if window_size:
            df = df.tail(window_size)
            df = df.reset_index(drop=True)
        ds["df"] = df

    df_merged = merge_data_sources(data_sources, config)

    out_path = data_path / symbol / config.get("merge_file_name")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_merged = df_merged.reset_index()
    if out_path.suffix == ".parquet":
        df_merged.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df_merged.to_csv(out_path, index=False)
    else:
        click.secho(f"❌ Unsupported output format: {out_path.suffix}", fg="red")
        return

    click.secho(f"✅ Merged file saved: {out_path} with {len(df_merged)} rows", fg="cyan")
    click.secho(f"⏱️ Completed in {str(datetime.now() - now).split('.')[0]}", fg="blue")


# Features

# Labels

# Train Model

# Test Model

# Save Model
