import collections
import importlib
import logging
import re
import shutil
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict, Dict
from threading import Lock

import numpy as np
import pandas as pd
import psutil

# import rapidjson
from joblib.externals import cloudpickle
from numpy.typing import NDArray
from pandas import DataFrame
from nautilus_ai.common.logging import Logger
from nautilus_ai.config import INautilusAIModelConfig


logger = Logger(__name__)


FEATURE_PIPELINE = "feature_pipeline"
LABEL_PIPELINE = "label_pipeline"
TRAINDF = "trained_df"
METADATA = "metadata"


class InstrumentInfo(TypedDict):
    """
    TypedDict to hold metadata about a specific instrument model.

    Attributes:
        model_filename (str): The filename of the trained model.
        trained_timestamp (int): The Unix timestamp of the model's training.
        data_path (str): Path to the data used for training or inference.
        extras (dict): Additional metadata or configurations.
    """

    model_filename: str
    trained_timestamp: int
    data_path: str
    extras: dict


class NautilusAIDataDrawer:
    """
    Manages instrument models and associated metadata in memory for better
    inference, retraining, and persistent storage to disk.

    This class is designed to persist across live/dry runs, holding:
    - Instrument metadata
    - Inference models
    - Historical data and predictions
    - Metric tracking

    It provides functionality to load/save these resources to/from disk.
    """

    def __init__(self, full_path: Path, config: INautilusAIModelConfig):
        """
        Initializes the NautilusAIDataDrawer.

        Parameters:
        -----------
            full_path (Path): Base path for storing and loading data.
            config (INautilusAIModelConfig): Configuration object for the Nautilus AI model.
        """
        self.full_path = full_path
        self.config = config
        self.nautilus_ai_info = config

        # In-memory storage for instrument metadata and models
        self.instrument_dict: Dict[str, InstrumentInfo] = {}
        self.model_dictionary: Dict[str, Any] = {}
        self.meta_data_dictionary: Dict[str, Dict[str, Any]] = {}
        self.model_return_values: Dict[str, DataFrame] = {}
        self.historic_data: Dict[str, Dict[str, DataFrame]] = {}
        self.historic_predictions: Dict[str, DataFrame] = {}

        # Paths for persistent storage
        self.historic_predictions_path = self.full_path / "historic_predictions.pkl"
        self.historic_predictions_bkp_path = (
            self.full_path / "historic_predictions.backup.pkl"
        )
        self.instrument_dictionary_path = self.full_path / "instrument_dictionary.json"
        self.global_metadata_path = self.full_path / "global_metadata.json"
        self.metric_tracker_path = self.full_path / "metric_tracker.json"

        # Load stored data into memory
        self.load_drawer_from_disk()
        self.load_historic_predictions_from_disk()

        # Metric tracking and locks for thread-safe operations
        self.metric_tracker: Dict[str, Dict[str, Dict[str, list]]] = {}
        self.load_metric_tracker_from_disk()
        self.training_queue: Dict[str, int] = {}
        self.history_lock = Lock()
        self.save_lock = Lock()
        self.instrument_dict_lock = Lock()
        self.metric_tracker_lock = Lock()

        # Additional metadata for retraining or analysis
        self.old_DBSCAN_eps: Dict[str, float] = {}
        self.empty_instrument_dict: InstrumentInfo = {
            "model_filename": "",
            "trained_timestamp": 0,
            "data_path": "",
            "extras": {},
        }
        self.model_type = self.nautilus_ai_info.model_save_type

    def load_drawer_from_disk(self) -> None:
        """
        Loads the instrument dictionary and other metadata from disk.
        Override this method as needed to implement custom loading logic.
        """
        # TODO: Implement disk loading logic
        pass

    def load_historic_predictions_from_disk(self) -> None:
        """
        Loads historic predictions from disk into memory.
        This ensures past predictions are available for analysis or backtesting.
        """
        # TODO: Implement historic prediction loading logic
        pass

    def load_metric_tracker_from_disk(self) -> None:
        """
        Loads the metric tracker from disk to restore state.
        This ensures metrics are persistent across sessions.
        """
        # TODO: Implement metric tracker loading logic
        pass
