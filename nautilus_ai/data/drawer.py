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

import rapidjson
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
    Manages instrument models, metrics, historical predictions, and associated metadata in memory for better
    inference, retraining with capabilities for persistent storage and thread-safe updates.

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

    def update_metric_tracker(self, metric: str, value: float, instrument: str) -> None:
        """
        Add or update metrics in the metric tracker for a given instrument.

        Args:
            metric (str): The name of the metric (e.g., "train_time", "cpu_load").
            value (float): The value of the metric.
            instrument (str): The instrument associated with the metric.
        """
        with self.metric_tracker_lock:
            instrument_metrics = self.metric_tracker.setdefault(instrument, {})
            metric_data = instrument_metrics.setdefault(
                metric, {"timestamp": [], "value": []}
            )

            timestamp = int(datetime.now(timezone.utc).timestamp())
            metric_data["value"].append(value)
            metric_data["timestamp"].append(timestamp)

    def collect_metrics(self, time_spent: float, instrument: str):
        """
        Collect training and system metrics and update the metric tracker.

        Args:
            time_spent (float): Time spent in training or processing.
            instrument (str): The instrument associated with the metrics.
        """
        load1, load5, load15 = psutil.getloadavg()
        cpu_count = psutil.cpu_count()

        self.update_metric_tracker("train_time", time_spent, instrument)
        self.update_metric_tracker("cpu_load1min", load1 / cpu_count, instrument)
        self.update_metric_tracker("cpu_load5min", load5 / cpu_count, instrument)
        self.update_metric_tracker("cpu_load15min", load15 / cpu_count, instrument)

    def load_global_metadata_from_disk(self) -> Dict[str, Any]:
        """
        Load global metadata from the disk.

        Returns:
            dict: The loaded metadata dictionary, or an empty dictionary if not found.
        """
        if self.global_metadata_path.is_file():
            with self.global_metadata_path.open("r") as fp:
                return rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
        return {}

    def load_drawer_from_disk(self):
        """
        Load the instrument dictionary and metadata from disk.
        """
        if self.instrument_dictionary_path.is_file():
            with self.instrument_dictionary_path.open("r") as fp:
                self.instrument_dict = rapidjson.load(
                    fp, number_mode=rapidjson.NM_NATIVE
                )
        else:
            logger.info("No existing data drawer found. Starting from scratch.")

    def load_metric_tracker_from_disk(self):
        """
        Load the metric tracker from disk if enabled in the configuration.
        """
        if self.freqai_info.get("write_metrics_to_disk", False):
            if self.metric_tracker_path.is_file():
                with self.metric_tracker_path.open("r") as fp:
                    self.metric_tracker = rapidjson.load(
                        fp, number_mode=rapidjson.NM_NATIVE
                    )
                logger.info("Metric tracker loaded from disk.")
            else:
                logger.info("No existing metric tracker found. Starting from scratch.")

    def load_historic_predictions_from_disk(self) -> bool:
        """
        Load historic predictions from disk, falling back to a backup if necessary.

        Returns:
            bool: Whether the predictions were successfully loaded.
        """
        if self.historic_predictions_path.is_file():
            try:
                with self.historic_predictions_path.open("rb") as fp:
                    self.historic_predictions = cloudpickle.load(fp)
                logger.info("Historic predictions loaded successfully.")
            except EOFError:
                logger.warning("Corrupted predictions file. Attempting to load backup.")
                with self.historic_predictions_bkp_path.open("rb") as fp:
                    self.historic_predictions = cloudpickle.load(fp)
                logger.warning("Backup predictions loaded successfully.")
            return True
        logger.info("No existing historic predictions found. Starting from scratch.")
        return False

    def save_historic_predictions_to_disk(self):
        """
        Save historic predictions to disk and create a backup.
        """
        with self.historic_predictions_path.open("wb") as fp:
            cloudpickle.dump(
                self.historic_predictions, fp, protocol=cloudpickle.DEFAULT_PROTOCOL
            )
        shutil.copy(self.historic_predictions_path, self.historic_predictions_bkp_path)

    def save_metric_tracker_to_disk(self):
        """
        Save the metric tracker to disk.
        """
        with self.save_lock:
            with self.metric_tracker_path.open("w") as fp:
                rapidjson.dump(
                    self.metric_tracker,
                    fp,
                    default=self.np_encoder,
                    number_mode=rapidjson.NM_NATIVE,
                )

    def save_drawer_to_disk(self):
        """
        Save the instrument dictionary to disk.
        """
        with self.save_lock:
            with self.instrument_dictionary_path.open("w") as fp:
                rapidjson.dump(
                    self.instrument_dict,
                    fp,
                    default=self.np_encoder,
                    number_mode=rapidjson.NM_NATIVE,
                )

    def save_global_metadata_to_disk(self, metadata: Dict[str, Any]):
        """
        Save global metadata to disk.

        Args:
            metadata (dict): The metadata dictionary to save.
        """
        with self.save_lock:
            with self.global_metadata_path.open("w") as fp:
                rapidjson.dump(
                    metadata,
                    fp,
                    default=self.np_encoder,
                    number_mode=rapidjson.NM_NATIVE,
                )

    @staticmethod
    def np_encoder(obj: Any) -> Any:
        """
        Encoder for NumPy data types to make them JSON serializable.

        Args:
            obj: The object to encode.

        Returns:
            Any: A JSON serializable representation.
        """
        if isinstance(obj, np.generic):
            return obj.item()
