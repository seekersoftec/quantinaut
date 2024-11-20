import copy
import inspect
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
from datasieve.pipeline import Pipeline
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from nautilus_trader.core.data import Data
from nautilus_trader.model.instruments import Instrument

from nautilus_ai.common.logging import Logger
from nautilus_ai.config import INautilusAIModelConfig
from nautilus_ai.data.types import TimeRange
from nautilus_ai.exceptions import OperationalException
from nautilus_ai.common.constants import DOCS_LINK

pd.set_option("future.no_silent_downcasting", True)

SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600

logger = Logger(__name__)


class NautilusAIDataKitchen(Data):
    """
    Class designed for data analysis and management of a single instrument.

    Used by the `INautilusAIModel` class, this class provides functionalities for
    holding, saving, loading, and analyzing data required for inference or training.

    Note:
        This object is not persistent. It is re-instantiated for each instrument whenever
        the model needs to be inferenced or trained.
    """

    def __init__(
        self,
        instrument: Instrument,
        config: INautilusAIModelConfig,
        ts_init: int,
        live: bool = False,
    ):
        """
        Initializes the NautilusAIDataKitchen instance.

        Parameters:
        -----------
        instrument : Instrument
            The financial instrument to be analyzed.
        config : INautilusAIModelConfig
            Configuration object containing Nautilus AI model settings.
        ts_init : int
            Initial timestamp for the data.
        live : bool, optional
            Indicates if the kitchen is being used for live data analysis (default is False).
        """
        super().__init__(ts_init=ts_init, ts_event=ts_init)

        self.config = config
        self.instrument = instrument
        self.ts_init = ts_init
        self.live = live

        # Data storage and processing
        self.data: dict[str, Any] = {}
        self.data_dictionary: dict[str, DataFrame] = {}
        self.full_df: DataFrame = DataFrame()
        self.append_df: DataFrame = DataFrame()
        self.label_list: list = []
        self.training_features_list: list = []
        self.unique_classes: dict[str, list] = {}
        self.unique_class_list: list = []
        self.train_dates: DataFrame = DataFrame()
        self.DI_values: npt.NDArray = np.array([])

        # Model and pipeline attributes
        self.model_filename: str = ""
        self.feature_pipeline = Pipeline()
        self.label_pipeline = Pipeline()

        # Paths and settings
        self.data_path = Path()
        self.full_path = Path()
        self.backtesting_results_path = Path()
        self.backtest_predictions_folder: str = "backtesting_predictions"
        self.keras = config.keras
        self.backtest_live_models = config.get("freqai_backtest_live_models", False)
        self.backtest_live_models_data: dict[str, Any] = {}

        # Thread management
        self.thread_count = config.data_kitchen_thread_count or max(
            int(psutil.cpu_count() * 2 - 2), 1
        )

        # Timerange management for backtesting
        if not self.live:
            self.full_path = self.get_full_models_path(self.config)

            if not self.backtest_live_models:
                self.full_timerange = self.create_full_timerange(
                    self.config["timerange"],
                    self.config.train_period_days,
                )
                (
                    self.training_timeranges,
                    self.backtesting_timeranges,
                ) = self.split_timerange(
                    self.full_timerange,
                    self.config.train_period_days,
                    self.config.backtest_period_days,
                )

        # Extra configuration
        self.data["extra_returns_per_train"] = config.extra_returns_per_train
        self.set_all_instruments()

    def set_paths(
        self,
        instrument: str,
        trained_timestamp: int | None = None,
    ) -> None:
        """
        Sets the paths for data storage and retrieval for the current instrument.

        Parameters:
        -----------
        instrument : str
            Name of the instrument (e.g., "BTC/USD").
        trained_timestamp : int, optional
            Timestamp of the most recent training (default is None).
        """
        self.full_path = self.get_full_models_path(self.config)
        self.data_path = Path(
            self.full_path / f"sub-train-{instrument.split('/')[0]}_{trained_timestamp}"
        )

    def make_train_test_datasets(
        self, filtered_dataframe: DataFrame, labels: DataFrame
    ) -> dict[str, Any]:
        """
        Splits the provided dataset into training and testing sets based on user-specified
        configuration parameters. If test size is set to 0, the entire dataset is used for training.

        Args:
            filtered_dataframe (DataFrame): Preprocessed dataset ready for splitting.
            labels (DataFrame): Labels corresponding to the dataset.

        Returns:
            dict[str, Any]: A dictionary containing training and test datasets, labels, and weights.
        """
        feat_params = self.config.feature_parameters
        split_params = self.config.data_split_parameters

        # Ensure shuffle key exists in configuration
        split_params.setdefault("shuffle", False)

        # Set weights for training samples
        weights = (
            self.set_weights_higher_recent(len(filtered_dataframe))
            if feat_params.weight_factor > 0
            else np.ones(len(filtered_dataframe))
        )

        # Split data into training and testing sets
        if split_params.get("test_size", 0.1) > 0:
            (
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_weights,
                test_weights,
            ) = train_test_split(
                filtered_dataframe,
                labels,
                weights,
                **split_params,
            )
        else:
            # Use the entire dataset for training when test size is 0
            train_features, train_labels, train_weights = (
                filtered_dataframe,
                labels,
                weights,
            )
            test_features, test_labels, test_weights = (
                pd.DataFrame(),
                np.zeros(2),
                np.zeros(2),
            )

        # Shuffle the datasets after splitting, if required
        if feat_params.shuffle_after_split:
            seed_train = random.randint(0, 100)
            seed_test = random.randint(0, 100)

            train_features = train_features.sample(
                frac=1, random_state=seed_train
            ).reset_index(drop=True)
            train_labels = train_labels.sample(
                frac=1, random_state=seed_train
            ).reset_index(drop=True)
            train_weights = (
                pd.DataFrame(train_weights)
                .sample(frac=1, random_state=seed_train)
                .reset_index(drop=True)
                .to_numpy()
                .ravel()
            )
            test_features = test_features.sample(
                frac=1, random_state=seed_test
            ).reset_index(drop=True)
            test_labels = test_labels.sample(
                frac=1, random_state=seed_test
            ).reset_index(drop=True)
            test_weights = (
                pd.DataFrame(test_weights)
                .sample(frac=1, random_state=seed_test)
                .reset_index(drop=True)
                .to_numpy()
                .ravel()
            )

        # Optionally reverse the train-test order
        if feat_params.reverse_train_test_order:
            return self.build_data_dictionary(
                test_features,
                train_features,
                test_labels,
                train_labels,
                test_weights,
                train_weights,
            )
        return self.build_data_dictionary(
            train_features,
            test_features,
            train_labels,
            test_labels,
            train_weights,
            test_weights,
        )

    def filter_features(
        self,
        unfiltered_df: DataFrame,
        training_feature_list: list[str],
        label_list: list[str] = None,
        training_filter: bool = True,
    ) -> tuple[DataFrame, DataFrame]:
        """
        Filters a raw dataset to include only the specified features and labels while
        handling missing values (NaNs). Removes or adjusts rows with NaNs based on whether
        the data is for training or prediction.

        Args:
            unfiltered_df (DataFrame): Raw dataset for the current period.
            training_feature_list (list[str]): List of feature columns to retain.
            label_list (list[str], optional): List of label columns to retain. Defaults to an empty list.
            training_filter (bool, optional): Indicates if the data is for training. If False, data
                                              is adjusted for prediction. Defaults to True.

        Returns:
            tuple[DataFrame, DataFrame]: A tuple of the cleaned feature and label DataFrames.
        """
        label_list = label_list or []

        # Select requested features and replace infinities with NaNs
        filtered_df = unfiltered_df.filter(training_feature_list, axis=1).replace(
            [np.inf, -np.inf], np.nan
        )

        drop_index = pd.isnull(filtered_df).any(axis=1)

        if training_filter:
            # For training: remove rows with NaNs from both features and labels
            labels = unfiltered_df.filter(label_list, axis=1)
            drop_index_labels = pd.isnull(labels).any(axis=1)
            drop_rows = drop_index | drop_index_labels

            filtered_df = filtered_df[~drop_rows]
            labels = labels[~drop_rows]
            self.train_dates = unfiltered_df.loc[~drop_rows, "date"]

            logger.info(
                f"{self.instrument}: Dropped {drop_rows.sum()} rows with NaNs from training data "
                f"({len(unfiltered_df)} total rows)."
            )

            # Raise an exception if no training data is left (only in backtest mode)
            if len(filtered_df) == 0 and not self.live:
                raise OperationalException(
                    f"{self.instrument}: All training data was dropped due to NaNs. Ensure sufficient "
                    "data coverage for your backtest period. See documentation: "
                    f"{DOCS_LINK}/freqai-running/#downloading-data-to-cover-the-full-backtest-period"
                )
        else:
            # For prediction: replace NaNs with zeros but keep track of affected rows
            filtered_df.fillna(0, inplace=True)
            drop_index = ~drop_index
            self.do_predict = drop_index.astype(int)
            labels = []

            logger.info(
                f"Dropped {len(drop_index) - drop_index.sum()} prediction data points due to NaNs "
                f"({len(filtered_df)} total rows)."
            )

        return filtered_df, labels

    def build_data_dictionary(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        train_labels: DataFrame,
        test_labels: DataFrame,
        train_weights: Any,
        test_weights: Any,
    ) -> dict:
        """
        Builds and returns a dictionary containing training and testing datasets along with their labels and weights.

        :param train_df: DataFrame containing training features.
        :param test_df: DataFrame containing testing features.
        :param train_labels: DataFrame containing training labels.
        :param test_labels: DataFrame containing testing labels.
        :param train_weights: Training sample weights.
        :param test_weights: Testing sample weights.
        :return: A dictionary encapsulating all provided data.
        """
        self.data_dictionary = {
            "train_features": train_df,
            "test_features": test_df,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "train_weights": train_weights,
            "test_weights": test_weights,
            "train_dates": self.train_dates,
        }
        return self.data_dictionary

    def split_timerange(
        self, tr: str, train_split: int = 28, bt_split: float = 7
    ) -> Tuple[List, List]:
        """
        Splits a given time range into training and backtesting sub-ranges.

        :param tr: String representing the full time range to split.
        :param train_split: Number of days for each training period.
        :param bt_split: Number of days for each backtesting period.
        :return: A tuple of lists containing training and backtesting time ranges.
        """
        if not isinstance(train_split, int) or train_split < 1:
            raise ValueError(
                f"train_split must be an integer greater than 0. Got {train_split}."
            )
        train_period_days = train_split * SECONDS_IN_DAY
        bt_period = bt_split * SECONDS_IN_DAY

        full_timerange = TimeRange.parse_timerange(tr)
        config_timerange = TimeRange.parse_timerange(self.config["timerange"])

        if config_timerange.stopts == 0:
            config_timerange.stopts = int(datetime.now(tz=timezone.utc).timestamp())

        timerange_train = copy.deepcopy(full_timerange)
        timerange_backtest = copy.deepcopy(full_timerange)

        tr_training_list = []
        tr_backtesting_list = []
        first_iteration = True

        while True:
            if not first_iteration:
                timerange_train.startts += int(bt_period)
            timerange_train.stopts = timerange_train.startts + train_period_days

            tr_training_list.append(copy.deepcopy(timerange_train))

            timerange_backtest.startts = timerange_train.stopts
            timerange_backtest.stopts = timerange_backtest.startts + int(bt_period)

            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts

            tr_backtesting_list.append(copy.deepcopy(timerange_backtest))

            if timerange_backtest.stopts == config_timerange.stopts:
                break
            first_iteration = False

        return tr_training_list, tr_backtesting_list

    def slice_dataframe(self, timerange: TimeRange, df: DataFrame) -> DataFrame:
        """
        Extracts a specific time window from a DataFrame based on the provided timerange.

        :param timerange: The timerange object specifying the desired window.
        :param df: The input DataFrame containing all data.
        :return: A sliced DataFrame for the specified timerange.
        """
        if not self.live:
            return df.loc[
                (df["date"] >= timerange.startdt) & (df["date"] < timerange.stopdt)
            ]
        return df.loc[df["date"] >= timerange.startdt]

    def find_features(self, dataframe: DataFrame) -> None:
        """
        Identifies feature columns in the DataFrame.

        :param dataframe: DataFrame containing data for feature identification.
        :raises ValueError: If no features are found.
        """
        features = [col for col in dataframe.columns if "%" in col]
        if not features:
            raise ValueError("No features found in the provided DataFrame.")
        self.training_features_list = features

    def find_labels(self, dataframe: DataFrame) -> None:
        """
        Identifies label columns in the DataFrame.

        :param dataframe: DataFrame containing data for label identification.
        """
        self.label_list = [col for col in dataframe.columns if "&" in col]

    def set_weights_higher_recent(self, num_weights: int) -> npt.ArrayLike:
        """
        Generates weights that assign higher importance to recent data.

        :param num_weights: Number of weights to generate.
        :return: Array of weights with recent data weighted higher.
        """
        wfactor = self.config.feature_parameters.weight_factor
        return np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]

    def append_predictions(self, append_df: DataFrame) -> None:
        """
        Appends predictions for the current backtesting period to the cumulative dataset.

        :param append_df: DataFrame containing predictions to append.
        """
        self.full_df = pd.concat([self.full_df, append_df], axis=0, ignore_index=True)

    def fill_predictions(self, dataframe: DataFrame) -> DataFrame:
        """
        Backfills missing predictions for earlier periods.

        :param dataframe: DataFrame to backfill with predictions.
        :return: Updated DataFrame with backfilled predictions.
        """
        non_label_cols = [
            col
            for col in dataframe.columns
            if not col.startswith("&") and not col.startswith("%%")
        ]
        self.return_dataframe = pd.merge(
            dataframe[non_label_cols], self.full_df, how="left", on="date"
        )
        self.return_dataframe.fillna(value=0, inplace=True)
        self.full_df = DataFrame()
        return self.return_dataframe

    def create_full_timerange(self, backtest_tr: str, backtest_period_days: int) -> str:
        """
        Creates a full timerange string by extending the start of a given backtest timerange
        backwards by a specified number of days. Also ensures the timerange is valid and
        prepares the configuration file.

        :param backtest_tr: str
            Timerange string for backtesting, formatted as 'start_date:stop_date'.
        :param backtest_period_days: int
            Number of days to extend the start of the backtest timerange backwards.

        :return: str
            A string representing the full timerange for the backtest.

        :raises OperationalException:
            - If `backtest_period_days` is not a positive integer.
            - If the backtest timerange is open-ended (no stop date is provided).
        """
        if not isinstance(backtest_period_days, int) or backtest_period_days <= 0:
            raise OperationalException(
                f"backtest_period_days must be a positive integer. Got {backtest_period_days}."
            )

        backtest_timerange = TimeRange.parse_timerange(backtest_tr)

        # Check for open-ended timeranges
        if backtest_timerange.stopts == 0:
            raise OperationalException(
                "NautilusAI backtesting does not allow open-ended timeranges. "
                "Please specify the end date in your backtest timerange."
            )

        # Extend the start of the timerange backwards by the specified number of days
        backtest_timerange.startts -= backtest_period_days * SECONDS_IN_DAY

        # Generate the full timerange string
        full_timerange = backtest_timerange.timerange_str

        # Ensure the configuration file is prepared in the appropriate directory
        config_path = Path(self.config["config_files"][0])

        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                config_path.resolve(),
                self.full_path / config_path.name,
            )

        return full_timerange
