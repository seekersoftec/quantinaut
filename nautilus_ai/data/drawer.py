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
from nautilus_trader.core.data import Data
from nautilus_trader.model.instruments import Instrument
from nautilus_ai.common.logging import Logger
from nautilus_ai.data import NautilusAIDataKitchen
from nautilus_ai.config import INautilusAIModelConfig
from nautilus_ai.exceptions import OperationalException


logger = Logger(__name__)


FEATURE_PIPELINE = "feature_pipeline"
LABEL_PIPELINE = "label_pipeline"
TRAIN_DF = "trained_df"
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


class NautilusAIDataDrawer(Data):
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

        TODO: Find a better type instead of Instrument.
        """
        self.full_path = full_path
        self.config = config
        self.nautilus_ai_info = config

        # In-memory storage for instrument metadata and models
        self.instrument_dict: Dict[Instrument, InstrumentInfo] = {}
        self.model_dictionary: Dict[Instrument, Any] = {}
        self.meta_data_dictionary: Dict[Instrument, Dict[str, Any]] = {}
        self.model_return_values: Dict[Instrument, DataFrame] = {}
        self.historic_data: Dict[Instrument, Dict[str, DataFrame]] = {}
        self.historic_predictions: Dict[Instrument, DataFrame] = {}

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

    def update_metric_tracker(
        self, metric: str, value: float, instrument: Instrument
    ) -> None:
        """
        Add or update metrics in the metric tracker for a given instrument.

        Args:
            metric (str): The name of the metric (e.g., "train_time", "cpu_load").
            value (float): The value of the metric.
            instrument (Instrument): The instrument associated with the metric.
        """
        with self.metric_tracker_lock:
            instrument_metrics = self.metric_tracker.setdefault(instrument, {})
            metric_data = instrument_metrics.setdefault(
                metric, {"timestamp": [], "value": []}
            )

            timestamp = int(datetime.now(timezone.utc).timestamp())
            metric_data["value"].append(value)
            metric_data["timestamp"].append(timestamp)

    def collect_metrics(self, time_spent: float, instrument: Instrument):
        """
        Collect training and system metrics and update the metric tracker.

        Args:
            time_spent (float): Time spent in training or processing.
            instrument (Instrument): The instrument associated with the metrics.
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
        if self.nautilus_ai_info.write_metrics_to_disk:
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

    def get_instrument_dict_info(self, instrument: Instrument) -> tuple[str, int]:
        """
        Retrieves or initializes metadata for a given instrument.

        If metadata for the specified instrument exists in `instrument_dict`, it is retrieved.
        Otherwise, a new entry is created using the `empty_instrument_dict` template.

        Args:
            instrument (Instrument): The instrument to lookup.

        Returns:
            tuple[str, int]:
                - `model_filename` (str): The unique filename for loading persistent objects.
                - `trained_timestamp` (int): The last training timestamp for the instrument.
        """
        instrument_dict = self.instrument_dict.get(instrument)
        if instrument_dict:
            model_filename = instrument_dict["model_filename"]
            trained_timestamp = instrument_dict["trained_timestamp"]
        else:
            self.instrument_dict[instrument] = self.empty_instrument_dict.copy()
            model_filename = ""
            trained_timestamp = 0
        return model_filename, trained_timestamp

    def set_instrument_dict_info(self, metadata: dict) -> None:
        """
        Adds an instrument's metadata to `instrument_dict` if it does not already exist.

        Args:
            metadata (dict): The metadata containing the instrument's details.
        """
        if metadata["instrument"] not in self.instrument_dict:
            self.instrument_dict[metadata["instrument"]] = (
                self.empty_instrument_dict.copy()
            )

    def set_initial_return_values(
        self, instrument: Instrument, pred_df: DataFrame, dataframe: DataFrame
    ) -> None:
        """
        Initializes historical predictions for an instrument by aligning historical and
        new predictions while accounting for downtime.

        Zeros are filled during downtime to ensure consistent historical data for the UI.

        Args:
            instrument (Instrument): The instrument being initialized.
            pred_df (DataFrame): New predictions DataFrame.
            dataframe (DataFrame): Original input DataFrame with time series data.
        """
        new_pred = pred_df.copy()
        new_pred["date_pred"] = dataframe["date"]
        new_pred[new_pred.columns.difference(["date_pred", "date"])] = None

        hist_preds = self.historic_predictions[instrument].copy()
        new_pred["date_pred"] = pd.to_datetime(new_pred["date_pred"])
        hist_preds["date_pred"] = pd.to_datetime(hist_preds["date_pred"])

        # Align new predictions with existing history
        common_dates = pd.merge(new_pred, hist_preds, on="date_pred", how="inner")
        if len(common_dates.index) > 0:
            new_pred = new_pred.iloc[len(common_dates) :]
        else:
            logger.warning(
                "No common dates found between new and historical predictions. "
                "FreqAI instance may have been offline for an extended period."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            new_pred_reindexed = new_pred.reindex(columns=hist_preds.columns)
            df_concat = pd.concat([hist_preds, new_pred_reindexed], ignore_index=True)

        df_concat = df_concat.fillna(0)
        self.historic_predictions[instrument] = df_concat
        self.model_return_values[instrument] = df_concat.tail(
            len(dataframe.index)
        ).reset_index(drop=True)

    def append_model_predictions(
        self,
        instrument: Instrument,
        predictions: DataFrame,
        do_preds: NDArray[np.int_],
        data_kitchen: NautilusAIDataKitchen,
        strat_df: DataFrame,
    ) -> None:
        """
        Appends new model predictions to historical data and updates strategy returns.

        Args:
            instrument (Instrument): The instrument being updated.
            predictions (DataFrame): Model prediction outputs.
            do_preds (NDArray[np.int_]): Outlier detection predictions.
            data_kitchen (NautilusAIDataKitchen): Data handler for predictions and stats.
            strat_df (DataFrame): Strategy data used for alignment.
        """
        len_df = len(strat_df)
        df = self.historic_predictions[instrument]
        columns = df.columns
        index = df.index[-1:]

        # Append a zeroed row
        zeros_df = pd.DataFrame(
            np.zeros((1, len(columns))), index=index, columns=columns
        )
        df = pd.concat([df, zeros_df], ignore_index=True, axis=0)

        # Update model outputs
        for label in predictions.columns:
            df.iloc[-1, df.columns.get_loc(label)] = predictions.iloc[
                -1, predictions.columns.get_loc(label)
            ]
            if df[label].dtype != object:
                df.iloc[-1, df.columns.get_loc(f"{label}_mean")] = data_kitchen.data[
                    "labels_mean"
                ][label]
                df.iloc[-1, df.columns.get_loc(f"{label}_std")] = data_kitchen.data[
                    "labels_std"
                ][label]

        # Update outlier and custom data
        df.iloc[-1, df.columns.get_loc("do_predict")] = do_preds[-1]
        if self.nautilus_ai_info.feature_parameters.DI_threshold > 0:
            df.iloc[-1, df.columns.get_loc("DI_values")] = data_kitchen.DI_values[-1]

        for return_str, value in data_kitchen.data["extra_returns_per_train"].items():
            df.iloc[-1, df.columns.get_loc(return_str)] = value

        # Update price and date information
        for col_name, strat_col in [
            ("high_price", "high"),
            ("low_price", "low"),
            ("close_price", "close"),
            ("date_pred", "date"),
        ]:
            df.iloc[-1, df.columns.get_loc(col_name)] = strat_df.iloc[
                -1, strat_df.columns.get_loc(strat_col)
            ]

        self.historic_predictions[instrument] = df
        self.model_return_values[instrument] = df.tail(len_df).reset_index(drop=True)

    def attach_return_values_to_dataframe(
        self, instrument: Instrument, dataframe: DataFrame
    ) -> DataFrame:
        """
        Attach computed return values to the provided strategy DataFrame.

        :param instrument: Instrument for which return values are attached.
        :param dataframe: Strategy DataFrame to which return values are added.
        :return: Updated DataFrame with return values appended.
        """
        return_values = self.model_return_values[instrument]
        # Retain only non-return columns
        base_columns = [col for col in dataframe.columns if not col.startswith("&")]
        # Merge base columns with return values
        return pd.concat([dataframe[base_columns], return_values], axis=1)

    def return_null_values_to_strategy(
        self, dataframe: DataFrame, data_kitchen: NautilusAIDataKitchen
    ) -> None:
        """
        Populate the strategy DataFrame with zero-filled placeholders for labels and features.

        :param dataframe: Strategy DataFrame to populate.
        :param data_kitchen: Data kitchen object for feature and label management.
        """
        # Identify features and labels
        data_kitchen.find_features(dataframe)
        data_kitchen.find_labels(dataframe)

        full_labels = data_kitchen.label_list + data_kitchen.unique_class_list

        # Initialize zero-filled columns for labels, means, and std deviations
        for label in full_labels:
            dataframe[label] = 0
            dataframe[f"{label}_mean"] = 0
            dataframe[f"{label}_std"] = 0

        # Add prediction and DI-related placeholders if applicable
        dataframe["do_predict"] = 0
        if self.nautilus_ai_info.feature_parameters.DI_threshold > 0:
            dataframe["DI_values"] = 0

        # Add extra returns if defined in the data kitchen
        for return_str in data_kitchen.data.get("extra_returns_per_train", []):
            dataframe[return_str] = 0

        # Store the updated DataFrame back in the data kitchen
        data_kitchen.return_dataframe = dataframe

    def purge_old_models(self) -> None:
        """
        Remove older model files to maintain storage efficiency.

        Keeps a defined number of the most recent model files for each instrument.
        """
        num_keep = self.nautilus_ai_info.purge_old_models or 2
        model_folders = [x for x in self.full_path.iterdir() if x.is_dir()]
        pattern = re.compile(r"sub-train-(\w+)_(\d{10})")
        delete_dict: dict[str, Any] = {}

        # Group model folders by instrument and timestamps
        for folder in model_folders:
            match = pattern.match(folder.name)
            if not match:
                continue
            instrument, timestamp = match.groups()
            timestamp = int(timestamp)
            delete_dict.setdefault(instrument, {"num_folders": 0, "timestamps": {}})
            delete_dict[instrument]["num_folders"] += 1
            delete_dict[instrument]["timestamps"][timestamp] = folder

        # Delete older models beyond the defined limit
        for instrument, details in delete_dict.items():
            if details["num_folders"] > num_keep:
                sorted_timestamps = collections.OrderedDict(
                    sorted(details["timestamps"].items())
                )
                to_delete = len(sorted_timestamps) - num_keep
                for _, folder in list(sorted_timestamps.items())[:to_delete]:
                    logger.info(f"Purging old model file: {folder}")
                    shutil.rmtree(folder)

    def save_metadata(self, data_kitchen: NautilusAIDataKitchen) -> None:
        """
        Save only metadata for backtesting to reduce storage usage.

        :param data_kitchen: Data kitchen containing model-related metadata.
        """
        save_path = data_kitchen.data_path
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "data_path": str(data_kitchen.data_path),
            "model_filename": str(data_kitchen.model_filename),
            "training_features_list": list(
                data_kitchen.data_dictionary["train_features"].columns
            ),
            "label_list": data_kitchen.label_list,
        }

        with (save_path / f"{data_kitchen.model_filename}_metadata.json").open(
            "w"
        ) as file:
            rapidjson.dump(
                metadata, file, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE
            )

    def save_data(
        self, model: Any, instrument: Instrument, data_kitchen: NautilusAIDataKitchen
    ) -> None:
        """
        Save all data associated with a trained model for a specific sub-train time range.

        :param model: Trained model object.
        :param instrument: Instrument associated with the model.
        :param data_kitchen: Data kitchen containing related data and pipelines.
        """
        save_path = data_kitchen.data_path
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the model based on its type
        model_filename = save_path / f"{data_kitchen.model_filename}_model"
        if self.model_type == "joblib":
            with model_filename.with_suffix(".joblib").open("wb") as file:
                cloudpickle.dump(model, file)
        elif self.model_type == "keras":
            model.save(model_filename.with_suffix(".h5"))
        elif self.model_type in ["stable_baselines3", "sb3_contrib", "pytorch"]:
            model.save(model_filename.with_suffix(".zip"))

        # Save metadata and pipelines
        metadata = {
            "data_path": str(data_kitchen.data_path),
            "model_filename": str(data_kitchen.model_filename),
            "training_features_list": data_kitchen.training_features_list,
            "label_list": data_kitchen.label_list,
        }

        with (save_path / f"{data_kitchen.model_filename}_metadata.json").open(
            "w"
        ) as file:
            rapidjson.dump(
                metadata, file, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE
            )

        with (save_path / f"{data_kitchen.model_filename}_feature_pipeline.pkl").open(
            "wb"
        ) as file:
            cloudpickle.dump(data_kitchen.feature_pipeline, file)

        with (save_path / f"{data_kitchen.model_filename}_label_pipeline.pkl").open(
            "wb"
        ) as file:
            cloudpickle.dump(data_kitchen.label_pipeline, file)

        # Save training data
        data_kitchen.data_dictionary["train_features"].to_pickle(
            save_path / f"{data_kitchen.model_filename}_train_features.pkl"
        )
        data_kitchen.data_dictionary["train_dates"].to_pickle(
            save_path / f"{data_kitchen.model_filename}_train_dates.pkl"
        )

        # Update internal dictionaries
        self.model_dictionary[instrument] = model
        self.instrument_dict[instrument] = {
            "model_filename": data_kitchen.model_filename,
            "data_path": str(data_kitchen.data_path),
        }
        self.meta_data_dictionary[instrument] = {
            "metadata": metadata,
            "feature_pipeline": data_kitchen.feature_pipeline,
            "label_pipeline": data_kitchen.label_pipeline,
        }
        self.save_drawer_to_disk()

    def load_metadata(self, data_kitchen: NautilusAIDataKitchen) -> None:
        """
        Load only metadata into the data kitchen to optimize performance during
        pre-saved backtesting, such as when loading prediction files.

        :param data_kitchen: NautilusAIDataKitchen instance to load metadata into.
        """
        metadata_path = (
            data_kitchen.data_path / f"{data_kitchen.model_filename}_{METADATA}.json"
        )
        try:
            with metadata_path.open("r") as fp:
                data_kitchen.data = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
            data_kitchen.training_features_list = data_kitchen.data[
                "training_features_list"
            ]
            data_kitchen.label_list = data_kitchen.data["label_list"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    def load_data(
        self, instrument: Instrument, data_kitchen: NautilusAIDataKitchen
    ) -> Any:
        """
        Load all necessary data required for predictions on a sub-train time range.

        :param instrument: The instrument for which data is to be loaded.
        :param data_kitchen: NautilusAIDataKitchen instance containing data paths and metadata.
        :returns: The user-trained model ready for inference.

        :raises FileNotFoundError: If required model or pipeline files are missing.
        :raises OperationalException: If the model fails to load.
        """
        model_filename = self.instrument_dict[instrument].get("model_filename")
        if not model_filename:
            return None

        # Update data_kitchen properties in live mode
        if data_kitchen.live:
            data_kitchen.model_filename = model_filename
            data_kitchen.data_path = Path(self.instrument_dict[instrument]["data_path"])

        # Load metadata and pipelines into memory
        if instrument in self.meta_data_dictionary:
            metadata = self.meta_data_dictionary[instrument]
            data_kitchen.data = metadata[METADATA]
            data_kitchen.feature_pipeline = metadata[FEATURE_PIPELINE]
            data_kitchen.label_pipeline = metadata[LABEL_PIPELINE]
        else:
            try:
                metadata_path = (
                    data_kitchen.data_path
                    / f"{data_kitchen.model_filename}_{METADATA}.json"
                )
                with metadata_path.open("r") as fp:
                    data_kitchen.data = rapidjson.load(
                        fp, number_mode=rapidjson.NM_NATIVE
                    )

                feature_pipeline_path = (
                    data_kitchen.data_path
                    / f"{data_kitchen.model_filename}_{FEATURE_PIPELINE}.pkl"
                )
                label_pipeline_path = (
                    data_kitchen.data_path
                    / f"{data_kitchen.model_filename}_{LABEL_PIPELINE}.pkl"
                )

                with feature_pipeline_path.open("rb") as fp:
                    data_kitchen.feature_pipeline = cloudpickle.load(fp)
                with label_pipeline_path.open("rb") as fp:
                    data_kitchen.label_pipeline = cloudpickle.load(fp)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Required file missing: {e.filename}")

        # Update data kitchen attributes
        data_kitchen.training_features_list = data_kitchen.data[
            "training_features_list"
        ]
        data_kitchen.label_list = data_kitchen.data["label_list"]

        # Load the model from memory or disk
        model = self.model_dictionary.get(instrument)
        if not model:
            model_path = data_kitchen.data_path / f"{data_kitchen.model_filename}_model"
            try:
                if self.model_type == "joblib":
                    with model_path.with_suffix(".joblib").open("rb") as fp:
                        model = cloudpickle.load(fp)
                elif self.model_type in ["stable_baselines3", "sb3_contrib"]:
                    if self.nautilus_ai_info.rl_config is None:
                        raise OperationalException(
                            f"Failed to load RL config from {self.nautilus_ai_info.rl_config}: {e}"
                        )

                    mod = importlib.import_module(
                        self.model_type, self.nautilus_ai_info.rl_config.model_type
                    )
                    MODELCLASS = getattr(
                        mod, self.nautilus_ai_info.rl_config.model_type
                    )
                    model = MODELCLASS.load(model_path)
                elif self.model_type == "pytorch":
                    import torch

                    zipfile = torch.load(model_path.with_suffix(".zip"))
                    model = zipfile["pytrainer"].load_from_checkpoint(zipfile)
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")
            except Exception as e:
                raise OperationalException(
                    f"Failed to load model from {model_path}: {e}"
                )

        # Cache the model in memory if loaded from disk
        self.model_dictionary.setdefault(instrument, model)

        return model

    #
    # TODO: Fix the methods below
    #

    def update_historic_data(
        self, strategy: IStrategy, data_kitchen: NautilusAIDataKitchen
    ) -> None:
        """
        Append new candles to our stores historic data (in memory) so that
        we do not need to load candle history from disk and we dont need to
        pinging exchange multiple times for the same candle.
        :param dataframe: DataFrame = strategy provided dataframe
        """
        feat_params = self.nautilus_ai_info.feature_parameters
        with self.history_lock:
            history_data = self.historic_data

            for instrument in data_kitchen.all_instruments:
                for tf in feat_params.include_timeframes:
                    hist_df = history_data[instrument][tf]
                    # check if newest candle is already appended
                    df_dp = strategy.dp.get_instrument_dataframe(instrument, tf)
                    if len(df_dp.index) == 0:
                        continue
                    if str(hist_df.iloc[-1]["date"]) == str(
                        df_dp.iloc[-1:]["date"].iloc[-1]
                    ):
                        continue

                    try:
                        index = (
                            df_dp.loc[df_dp["date"] == hist_df.iloc[-1]["date"]].index[
                                0
                            ]
                            + 1
                        )
                    except IndexError:
                        if hist_df.iloc[-1]["date"] < df_dp["date"].iloc[0]:
                            raise OperationalException(
                                "In memory historical data is older than "
                                f"oldest DataProvider candle for {instrument} on "
                                f"timeframe {tf}"
                            )
                        else:
                            index = -1
                            logger.warning(
                                f"No common dates in historical data and dataprovider for {instrument}. "
                                f"Appending latest dataprovider candle to historical data "
                                "but please be aware that there is likely a gap in the historical "
                                "data. \n"
                                f"Historical data ends at {hist_df.iloc[-1]['date']} "
                                f"while dataprovider starts at {df_dp['date'].iloc[0]} and"
                                f"ends at {df_dp['date'].iloc[0]}."
                            )

                    history_data[instrument][tf] = pd.concat(
                        [
                            hist_df,
                            df_dp.iloc[index:],
                        ],
                        ignore_index=True,
                        axis=0,
                    )

            self.current_candle = history_data[data_kitchen.instrument][
                self.config["timeframe"]
            ].iloc[-1]["date"]

    def load_all_instrument_histories(
        self, timerange: TimeRange, data_kitchen: NautilusAIDataKitchen
    ) -> None:
        """
        Load instrument histories for all whitelist and corr_instrumentlist instruments.
        Only called once upon startup of bot.
        :param timerange: TimeRange = full timerange required to populate all indicators
                          for training according to user defined train_period_days
        """
        history_data = self.historic_data

        for instrument in data_kitchen.all_instruments:
            if instrument not in history_data:
                history_data[instrument] = {}
            for tf in self.nautilus_ai_info.feature_parameters.include_timeframes:
                history_data[instrument][tf] = load_instrument_history(
                    datadir=self.config["datadir"],
                    timeframe=tf,
                    instrument=instrument,
                    timerange=timerange,
                    data_format=self.config.get("dataformat_ohlcv", "feather"),
                    candle_type=self.config.get("candle_type_def", CandleType.SPOT),
                )

    def get_base_and_corr_dataframes(
        self, timerange: TimeRange, instrument: str, data_kitchen: NautilusAIDataKitchen
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """
        Searches through our historic_data in memory and returns the dataframes relevant
        to the present instrument.
        :param timerange: TimeRange = full timerange required to populate all indicators
                          for training according to user defined train_period_days
        :param metadata: dict = strategy furnished instrument metadata
        """
        with self.history_lock:
            corr_dataframes: dict[Any, Any] = {}
            base_dataframes: dict[Any, Any] = {}
            historic_data = self.historic_data
            instruments = (
                self.nautilus_ai_info.feature_parameters.include_corr_instrumentlist
            )

            for tf in self.nautilus_ai_info.feature_parameters.include_timeframes:
                base_dataframes[tf] = data_kitchen.slice_dataframe(
                    timerange, historic_data[instrument][tf]
                ).reset_index(drop=True)
                if instruments:
                    for p in instruments:
                        if instrument in p:
                            continue  # dont repeat anything from whitelist
                        if p not in corr_dataframes:
                            corr_dataframes[p] = {}
                        corr_dataframes[p][tf] = data_kitchen.slice_dataframe(
                            timerange, historic_data[p][tf]
                        ).reset_index(drop=True)

        return corr_dataframes, base_dataframes

    def get_timerange_from_live_historic_predictions(self) -> TimeRange:
        """
        Returns timerange information based on historic predictions file
        :return: timerange calculated from saved live data
        """
        if not self.historic_predictions_path.is_file():
            raise OperationalException(
                "Historic predictions not found. Historic predictions data is required "
                "to run backtest with the freqai-backtest-live-models option "
            )

        self.load_historic_predictions_from_disk()

        all_instruments_end_dates = []
        for instrument in self.historic_predictions:
            instrument_historic_data = self.historic_predictions[instrument]
            all_instruments_end_dates.append(instrument_historic_data.date_pred.max())

        global_metadata = self.load_global_metadata_from_disk()
        start_date = datetime.fromtimestamp(int(global_metadata["start_dry_live_date"]))
        end_date = max(all_instruments_end_dates)
        # add 1 day to string timerange to ensure BT module will load all dataframe data
        end_date = end_date + timedelta(days=1)
        backtesting_timerange = TimeRange(
            "date", "date", int(start_date.timestamp()), int(end_date.timestamp())
        )
        return backtesting_timerange
