from pathlib import Path
from typing import Union
import json
import re

# from common.model_store import *

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

