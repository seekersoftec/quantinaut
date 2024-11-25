import copy
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from nautilus_ai.common.timerange import TimeRange
from nautilus_ai.common.utils import timeframe_to_seconds
from nautilus_ai.config import INautilusAIModelConfig
from nautilus_ai.exceptions import ConfigurationError, OperationalException
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
    ) -> Dict[str, Any]:
        """
        Constructs a dictionary containing training and testing datasets, labels, and weights.

        Parameters:
        -----------
        train_df : DataFrame
            Training features dataset.
        test_df : DataFrame
            Testing features dataset.
        train_labels : DataFrame
            Labels corresponding to training features.
        test_labels : DataFrame
            Labels corresponding to testing features.
        train_weights : Any
            Weights associated with the training samples.
        test_weights : Any
            Weights associated with the testing samples.

        Returns:
        --------
        Dict[str, Any]
            Dictionary encapsulating the training and testing datasets, labels, and weights.
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
        self, tr: str, train_split: int = 28, bt_split: float = 7.0
    ) -> Tuple[List[TimeRange], List[TimeRange]]:
        """
        Splits a time range into training and backtesting sub-ranges.

        Parameters:
        -----------
        tr : str
            Full timerange in a valid format to be split (e.g., "20220101-20230101").
        train_split : int, optional
            Number of days for each training period (default is 28).
        bt_split : float, optional
            Number of days for each backtesting period (default is 7.0).

        Returns:
        --------
        Tuple[List[TimeRange], List[TimeRange]]
            Lists of training and backtesting time ranges.

        Raises:
        -------
        ValueError
            If `train_split` is not a positive integer.
        """
        if not isinstance(train_split, int) or train_split < 1:
            raise ValueError(
                f"train_split must be an integer greater than 0. Got {train_split}."
            )

        train_period_seconds = train_split * SECONDS_IN_DAY
        backtest_period_seconds = int(bt_split * SECONDS_IN_DAY)

        full_timerange = TimeRange.parse_timerange(tr)
        config_timerange = TimeRange.parse_timerange(self.config.timerange)

        # If stop timestamp is undefined in config, set it to the current UTC time
        if config_timerange.stopts == 0:
            config_timerange.stopts = int(datetime.now(tz=timezone.utc).timestamp())

        timerange_train = copy.deepcopy(full_timerange)
        timerange_backtest = copy.deepcopy(full_timerange)

        tr_training_list = []
        tr_backtesting_list = []
        first_iteration = True

        while True:
            if not first_iteration:
                timerange_train.startts += backtest_period_seconds
            timerange_train.stopts = timerange_train.startts + train_period_seconds

            # Add training range
            tr_training_list.append(copy.deepcopy(timerange_train))

            # Set up backtesting range based on the training range
            timerange_backtest.startts = timerange_train.stopts
            timerange_backtest.stopts = (
                timerange_backtest.startts + backtest_period_seconds
            )

            # Ensure backtesting stop date doesn't exceed config's timerange
            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts

            # Add backtesting range
            tr_backtesting_list.append(copy.deepcopy(timerange_backtest))

            # Break loop if we've reached the end of the configured timerange
            if timerange_backtest.stopts == config_timerange.stopts:
                break

            first_iteration = False

        return tr_training_list, tr_backtesting_list

    def slice_dataframe(self, timerange: TimeRange, df: DataFrame) -> DataFrame:
        """
        Extracts a slice of the DataFrame based on the specified timerange.

        Parameters:
        -----------
        timerange : TimeRange
            The timerange object specifying the desired data window.
        df : DataFrame
            The input DataFrame containing all data.

        Returns:
        --------
        DataFrame
            Sliced DataFrame for the specified timerange.

        Notes:
        ------
        - If `self.live` is True, the DataFrame is filtered to include all rows
          from `timerange.startdt` onward.
        - If `self.live` is False, the DataFrame is filtered to include rows
          within the range `[timerange.startdt, timerange.stopdt)`.

        """
        if not self.live:
            return df.loc[
                (df["date"] >= timerange.startdt) & (df["date"] < timerange.stopdt)
            ]
        return df.loc[df["date"] >= timerange.startdt]

    def find_features(self, dataframe: DataFrame) -> None:
        """
        Identifies feature columns in the DataFrame.

        Parameters:
        -----------
        dataframe : DataFrame
            DataFrame containing data for feature identification.

        Raises:
        -------
        ValueError
            If no feature columns are found in the DataFrame.

        Notes:
        ------
        - Feature columns are identified by the presence of `%` in their names.
        """
        features = [col for col in dataframe.columns if "%" in col]
        if not features:
            raise ValueError("No features found in the provided DataFrame.")
        self.training_features_list = features

    def find_labels(self, dataframe: DataFrame) -> None:
        """
        Identifies label columns in the DataFrame.

        Parameters:
        -----------
        dataframe : DataFrame
            DataFrame containing data for label identification.

        Notes:
        ------
        - Label columns are identified by the presence of `&` in their names.
        """
        self.label_list = [col for col in dataframe.columns if "&" in col]

    def set_weights_higher_recent(self, num_weights: int) -> npt.ArrayLike:
        """
        Generates an array of weights that assigns higher importance to recent data.

        Parameters:
        -----------
        num_weights : int
            Number of weights to generate.

        Returns:
        --------
        npt.ArrayLike
            Array of weights with recent data weighted higher.

        Notes:
        ------
        - The weight distribution follows an exponential decay function.
        - The decay rate is determined by the `weight_factor` in the configuration.
        """
        wfactor = self.config["feature_parameters"]["weight_factor"]
        return np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]

    def append_predictions(self, append_df: DataFrame) -> None:
        """
        Appends predictions for the current backtesting period to the cumulative dataset.

        Parameters:
        -----------
        append_df : DataFrame
            DataFrame containing predictions to append.

        Notes:
        ------
        - The predictions are appended to `self.full_df`.
        - Assumes `append_df` contains a `date` column for alignment.
        """
        if append_df.empty:
            raise ValueError(
                "The provided DataFrame `append_df` is empty and cannot be appended."
            )

        self.full_df = pd.concat([self.full_df, append_df], axis=0, ignore_index=True)

    def fill_predictions(self, dataframe: DataFrame) -> DataFrame:
        """
        Backfills missing predictions for earlier periods by merging with cumulative predictions.

        Parameters:
        -----------
        dataframe : DataFrame
            The input DataFrame to backfill with predictions.

        Returns:
        --------
        DataFrame
            Updated DataFrame with backfilled predictions.

        Notes:
        ------
        - Non-label columns are identified and merged with `self.full_df` based on the `date` column.
        - Missing values in predictions are filled with `0`.
        - Resets `self.full_df` to an empty DataFrame after backfilling.
        """
        if dataframe.empty:
            raise ValueError(
                "The provided DataFrame `dataframe` is empty and cannot be backfilled."
            )

        if "date" not in dataframe.columns:
            raise KeyError(
                "The input DataFrame `dataframe` must contain a `date` column for merging."
            )

        non_label_cols = [
            col
            for col in dataframe.columns
            if not col.startswith("&") and not col.startswith("%%")
        ]

        # Merge input DataFrame with cumulative predictions
        self.return_dataframe = pd.merge(
            dataframe[non_label_cols], self.full_df, how="left", on="date"
        )

        # Fill missing predictions with 0
        self.return_dataframe.fillna(value=0, inplace=True)

        # Reset the cumulative DataFrame
        self.full_df = DataFrame()

        return self.return_dataframe

    def create_full_timerange(self, backtest_tr: str, backtest_period_days: int) -> str:
        """
        Creates a full timerange string by extending the start of a given backtest timerange
        backwards by a specified number of days. Ensures the timerange is valid and prepares
        the configuration file.

        Parameters:
        -----------
        backtest_tr : str
            Timerange string for backtesting, formatted as 'start_date:stop_date'.
        backtest_period_days : int
            Number of days to extend the start of the backtest timerange backwards.

        Returns:
        --------
        str
            A string representing the full timerange for the backtest.

        Raises:
        -------
        OperationalException
            If `backtest_period_days` is not a positive integer.
            If the backtest timerange is open-ended (no stop date is provided).
        """
        # Validate `backtest_period_days`
        if not isinstance(backtest_period_days, int) or backtest_period_days <= 0:
            raise OperationalException(
                f"`backtest_period_days` must be a positive integer. Got {backtest_period_days}."
            )

        # Parse the backtest timerange
        backtest_timerange = TimeRange.parse_timerange(backtest_tr)

        # Ensure the timerange has an end date
        if backtest_timerange.stopts == 0:
            raise OperationalException(
                "Backtesting does not allow open-ended timeranges. "
                "Please specify an end date in the backtest timerange."
            )

        # Extend the start of the timerange backwards
        backtest_timerange.startts -= backtest_period_days * SECONDS_IN_DAY

        # Generate the full timerange string
        full_timerange = backtest_timerange.timerange_str

        # Prepare configuration directory and file
        self._prepare_config_file()

        return full_timerange

    def _prepare_config_file(self) -> None:
        """
        Ensures the configuration directory exists and copies the configuration file to it.

        Raises:
        -------
        FileNotFoundError
            If the configuration file does not exist.
        """
        config_path = Path(self.config["config_files"][0])

        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy(config_path.resolve(), self.full_path / config_path.name)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path.resolve()}"
                ) from e

    def check_if_model_expired(self, trained_timestamp: int) -> bool:
        """
        Determines if the model is expired based on user-defined `expiration_hours`.

        Parameters:
        -----------
        trained_timestamp : int
            The timestamp of the model's last training (in seconds since epoch).

        Returns:
        --------
        bool
            True if the model is expired, False otherwise.

        Notes:
        ------
        - If `expiration_hours` is set to 0 or a negative value, the model never expires.
        """
        current_time = datetime.now(tz=timezone.utc).timestamp()
        elapsed_time_hours = (current_time - trained_timestamp) / SECONDS_IN_HOUR
        max_expiration_hours = self.config.expiration_hours

        if max_expiration_hours > 0:
            return elapsed_time_hours > max_expiration_hours
        return False

    def check_if_new_training_required(
        self, trained_timestamp: int
    ) -> Tuple[bool, TimeRange, TimeRange]:
        """
        Determines if retraining is required and prepares time ranges for training and data loading.

        Parameters:
        -----------
        trained_timestamp : int
            The timestamp of the model's last training (in seconds since epoch).

        Returns:
        --------
        Tuple[bool, TimeRange, TimeRange]
            A tuple containing:
            - A boolean indicating if retraining is required.
            - A TimeRange object for the training period.
            - A TimeRange object for the data loading period.

        Notes:
        ------
        - `trained_timerange` defines the period for training data.
        - `data_load_timerange` includes additional time to accommodate rolling indicators.

        Raises:
        -------
        ConfigurationError
            If the configuration is missing required parameters.
        """
        current_time = datetime.now(tz=timezone.utc).timestamp()
        trained_timerange = TimeRange()
        data_load_timerange = TimeRange()

        timeframes = self.config.feature_parameters.include_timeframes
        if not timeframes:
            raise ConfigurationError(
                "Missing or invalid `include_timeframes` in configuration."
            )

        # Calculate maximum timeframe in seconds
        max_tf_seconds = max(timeframe_to_seconds(tf) for tf in timeframes)

        # Safety factor: extend period for rolling indicators
        max_period = self.config.startup_candle_count * 2
        additional_seconds = max_period * max_tf_seconds

        train_period_days = self.config.train_period_days
        train_period_seconds = train_period_days * SECONDS_IN_DAY

        if trained_timestamp:
            elapsed_time_hours = (current_time - trained_timestamp) / SECONDS_IN_HOUR
            retrain = elapsed_time_hours > self.config.live_retrain_hours

            if retrain:
                trained_timerange.startts = int(current_time - train_period_seconds)
                trained_timerange.stopts = int(current_time)

                data_load_timerange.startts = int(
                    current_time - train_period_seconds - additional_seconds
                )
                data_load_timerange.stopts = int(current_time)
        else:  # No prior training timestamp provided
            trained_timerange.startts = int(current_time - train_period_seconds)
            trained_timerange.stopts = int(current_time)

            data_load_timerange.startts = int(
                current_time - train_period_seconds - additional_seconds
            )
            data_load_timerange.stopts = int(current_time)

            retrain = True

        return retrain, trained_timerange, data_load_timerange

    #
    # TODO: Fix the methods below
    #

    # def set_new_model_names(self, instrument: str, timestamp_id: int):
    #     coin, _ = instrument.split("/")
    #     self.data_path = Path(
    #         self.full_path / f"sub-train-{instrument.split('/')[0]}_{timestamp_id}"
    #     )

    #     self.model_filename = f"cb_{coin.lower()}_{timestamp_id}"

    # def set_all_instruments(self) -> None:
    #     self.all_instruments = copy.deepcopy(
    #         self.config.feature_parameters.include_corr_instrumentlist
    #     )
    #     for instrument in self.config.get("exchange", "").get("instrument_whitelist"):
    #         if instrument not in self.all_instruments:
    #             self.all_instruments.append(instrument)

    # def extract_corr_instrument_columns_from_populated_indicators(
    #     self, dataframe: DataFrame
    # ) -> dict[str, DataFrame]:
    #     """
    #     Find the columns of the dataframe corresponding to the corr_instrumentlist, save them
    #     in a dictionary to be reused and attached to other instruments.

    #     :param dataframe: fully populated dataframe (current instrument + corr_instruments)
    #     :return: corr_dataframes, dictionary of dataframes to be attached
    #              to other instruments in same candle.
    #     """
    #     corr_dataframes: dict[str, DataFrame] = {}
    #     instruments = self.freqai_config["feature_parameters"].get(
    #         "include_corr_instrumentlist", []
    #     )

    #     for instrument in instruments:
    #         instrument = instrument.replace(":", "")  # lightgbm does not like colons
    #         instrument_cols = [
    #             col
    #             for col in dataframe.columns
    #             if col.startswith("%") and f"{instrument}_" in col
    #         ]

    #         if instrument_cols:
    #             instrument_cols.insert(0, "date")
    #             corr_dataframes[instrument] = dataframe.filter(instrument_cols, axis=1)

    #     return corr_dataframes

    # def attach_corr_instrument_columns(
    #     self,
    #     dataframe: DataFrame,
    #     corr_dataframes: dict[str, DataFrame],
    #     current_instrument: str,
    # ) -> DataFrame:
    #     """
    #     Attach the existing corr_instrument dataframes to the current instrument dataframe before training

    #     :param dataframe: current instrument strategy dataframe, indicators populated already
    #     :param corr_dataframes: dictionary of saved dataframes from earlier in the same candle
    #     :param current_instrument: current instrument to which we will attach corr instrument dataframe
    #     :return:
    #     :dataframe: current instrument dataframe of populated indicators, concatenated with corr_instruments
    #                 ready for training
    #     """
    #     instruments = self.freqai_config["feature_parameters"].get(
    #         "include_corr_instrumentlist", []
    #     )
    #     current_instrument = current_instrument.replace(":", "")
    #     for instrument in instruments:
    #         instrument = instrument.replace(
    #             ":", ""
    #         )  # lightgbm does not work with colons
    #         if current_instrument != instrument:
    #             dataframe = dataframe.merge(
    #                 corr_dataframes[instrument], how="left", on="date"
    #             )

    #     return dataframe

    # def get_instrument_data_for_features(
    #     self,
    #     instrument: str,
    #     tf: str,
    #     strategy: IStrategy,
    #     corr_dataframes: dict = {},
    #     base_dataframes: dict = {},
    #     is_corr_instruments: bool = False,
    # ) -> DataFrame:
    #     """
    #     Get the data for the instrument. If it's not in the dictionary, get it from the data provider
    #     :param instrument: str = instrument to get data for
    #     :param tf: str = timeframe to get data for
    #     :param strategy: IStrategy = user defined strategy object
    #     :param corr_dataframes: dict = dict containing the df instrument dataframes
    #                             (for user defined timeframes)
    #     :param base_dataframes: dict = dict containing the current instrument dataframes
    #                             (for user defined timeframes)
    #     :param is_corr_instruments: bool = whether the instrument is a corr instrument or not
    #     :return: dataframe = dataframe containing the instrument data
    #     """
    #     if is_corr_instruments:
    #         dataframe = corr_dataframes[instrument][tf]
    #         if not dataframe.empty:
    #             return dataframe
    #         else:
    #             dataframe = strategy.dp.get_instrument_dataframe(
    #                 instrument=instrument, timeframe=tf
    #             )
    #             return dataframe
    #     else:
    #         dataframe = base_dataframes[tf]
    #         if not dataframe.empty:
    #             return dataframe
    #         else:
    #             dataframe = strategy.dp.get_instrument_dataframe(
    #                 instrument=instrument, timeframe=tf
    #             )
    #             return dataframe

    # def merge_features(
    #     self,
    #     df_main: DataFrame,
    #     df_to_merge: DataFrame,
    #     tf: str,
    #     timeframe_inf: str,
    #     suffix: str,
    # ) -> DataFrame:
    #     """
    #     Merge the features of the dataframe and remove HLCV and date added columns
    #     :param df_main: DataFrame = main dataframe
    #     :param df_to_merge: DataFrame = dataframe to merge
    #     :param tf: str = timeframe of the main dataframe
    #     :param timeframe_inf: str = timeframe of the dataframe to merge
    #     :param suffix: str = suffix to add to the columns of the dataframe to merge
    #     :return: dataframe = merged dataframe
    #     """
    #     dataframe = merge_informative_instrument(
    #         df_main,
    #         df_to_merge,
    #         tf,
    #         timeframe_inf=timeframe_inf,
    #         append_timeframe=False,
    #         suffix=suffix,
    #         ffill=True,
    #     )
    #     skip_columns = [
    #         (f"{s}_{suffix}")
    #         for s in ["date", "open", "high", "low", "close", "volume"]
    #     ]
    #     dataframe = dataframe.drop(columns=skip_columns)
    #     return dataframe

    # def populate_features(
    #     self,
    #     dataframe: DataFrame,
    #     instrument: str,
    #     strategy: IStrategy,
    #     corr_dataframes: dict,
    #     base_dataframes: dict,
    #     is_corr_instruments: bool = False,
    # ) -> DataFrame:
    #     """
    #     Use the user defined strategy functions for populating features
    #     :param dataframe: DataFrame = dataframe to populate
    #     :param instrument: str = instrument to populate
    #     :param strategy: IStrategy = user defined strategy object
    #     :param corr_dataframes: dict = dict containing the df instrument dataframes
    #     :param base_dataframes: dict = dict containing the current instrument dataframes
    #     :param is_corr_instruments: bool = whether the instrument is a corr instrument or not
    #     :return: dataframe = populated dataframe
    #     """
    #     tfs: list[str] = self.freqai_config["feature_parameters"].get(
    #         "include_timeframes"
    #     )

    #     for tf in tfs:
    #         metadata = {"instrument": instrument, "tf": tf}
    #         informative_df = self.get_instrument_data_for_features(
    #             instrument,
    #             tf,
    #             strategy,
    #             corr_dataframes,
    #             base_dataframes,
    #             is_corr_instruments,
    #         )
    #         informative_copy = informative_df.copy()

    #         logger.debug(f"Populating features for {instrument} {tf}")

    #         for t in self.freqai_config["feature_parameters"][
    #             "indicator_periods_candles"
    #         ]:
    #             df_features = strategy.feature_engineering_expand_all(
    #                 informative_copy.copy(), t, metadata=metadata
    #             )
    #             suffix = f"{t}"
    #             informative_df = self.merge_features(
    #                 informative_df, df_features, tf, tf, suffix
    #             )

    #         generic_df = strategy.feature_engineering_expand_basic(
    #             informative_copy.copy(), metadata=metadata
    #         )
    #         suffix = "gen"

    #         informative_df = self.merge_features(
    #             informative_df, generic_df, tf, tf, suffix
    #         )

    #         indicators = [col for col in informative_df if col.startswith("%")]
    #         for n in range(
    #             self.freqai_config["feature_parameters"]["include_shifted_candles"] + 1
    #         ):
    #             if n == 0:
    #                 continue
    #             df_shift = informative_df[indicators].shift(n)
    #             df_shift = df_shift.add_suffix("_shift-" + str(n))
    #             informative_df = pd.concat((informative_df, df_shift), axis=1)

    #         dataframe = self.merge_features(
    #             dataframe.copy(),
    #             informative_df,
    #             self.config["timeframe"],
    #             tf,
    #             f"{instrument}_{tf}",
    #         )

    #     return dataframe

    # def use_strategy_to_populate_indicators(  # noqa: C901
    #     self,
    #     strategy: IStrategy,
    #     corr_dataframes: dict = {},
    #     base_dataframes: dict = {},
    #     instrument: str = "",
    #     prediction_dataframe: DataFrame = pd.DataFrame(),
    #     do_corr_instruments: bool = True,
    # ) -> DataFrame:
    #     """
    #     Use the user defined strategy for populating indicators during retrain
    #     :param strategy: IStrategy = user defined strategy object
    #     :param corr_dataframes: dict = dict containing the df instrument dataframes
    #                             (for user defined timeframes)
    #     :param base_dataframes: dict = dict containing the current instrument dataframes
    #                             (for user defined timeframes)
    #     :param instrument: str = instrument to populate
    #     :param prediction_dataframe: DataFrame = dataframe containing the instrument data
    #     used for prediction
    #     :param do_corr_instruments: bool = whether to populate corr instruments or not
    #     :return:
    #     dataframe: DataFrame = dataframe containing populated indicators
    #     """

    #     # check if the user is using the deprecated populate_any_indicators function
    #     new_version = inspect.getsource(strategy.populate_any_indicators) == (
    #         inspect.getsource(IStrategy.populate_any_indicators)
    #     )

    #     if not new_version:
    #         raise OperationalException(
    #             "You are using the `populate_any_indicators()` function"
    #             " which was deprecated on March 1, 2023. Please refer "
    #             "to the strategy migration guide to use the new "
    #             "feature_engineering_* methods: \n"
    #             f"{DOCS_LINK}/strategy_migration/#freqai-strategy \n"
    #             "And the feature_engineering_* documentation: \n"
    #             f"{DOCS_LINK}/freqai-feature-engineering/"
    #         )

    #     tfs: list[str] = self.freqai_config["feature_parameters"].get(
    #         "include_timeframes"
    #     )
    #     instruments: list[str] = self.freqai_config["feature_parameters"].get(
    #         "include_corr_instrumentlist", []
    #     )

    #     for tf in tfs:
    #         if tf not in base_dataframes:
    #             base_dataframes[tf] = pd.DataFrame()
    #         for p in instruments:
    #             if p not in corr_dataframes:
    #                 corr_dataframes[p] = {}
    #             if tf not in corr_dataframes[p]:
    #                 corr_dataframes[p][tf] = pd.DataFrame()

    #     if not prediction_dataframe.empty:
    #         dataframe = prediction_dataframe.copy()
    #         base_dataframes[self.config["timeframe"]] = dataframe.copy()
    #     else:
    #         dataframe = base_dataframes[self.config["timeframe"]].copy()

    #     corr_instruments: list[str] = self.freqai_config["feature_parameters"].get(
    #         "include_corr_instrumentlist", []
    #     )
    #     dataframe = self.populate_features(
    #         dataframe.copy(), instrument, strategy, corr_dataframes, base_dataframes
    #     )
    #     metadata = {"instrument": instrument}
    #     dataframe = strategy.feature_engineering_standard(
    #         dataframe.copy(), metadata=metadata
    #     )
    #     # ensure corr instruments are always last
    #     for corr_instrument in corr_instruments:
    #         if instrument == corr_instrument:
    #             continue  # dont repeat anything from whitelist
    #         if corr_instruments and do_corr_instruments:
    #             dataframe = self.populate_features(
    #                 dataframe.copy(),
    #                 corr_instrument,
    #                 strategy,
    #                 corr_dataframes,
    #                 base_dataframes,
    #                 True,
    #             )

    #     if self.live:
    #         dataframe = strategy.set_freqai_targets(dataframe.copy(), metadata=metadata)
    #         dataframe = self.remove_special_chars_from_feature_names(dataframe)

    #     self.get_unique_classes_from_labels(dataframe)

    #     if self.config.get("reduce_df_footprint", False):
    #         dataframe = reduce_dataframe_footprint(dataframe)

    #     return dataframe

    # def fit_labels(self) -> None:
    #     """
    #     Fit the labels with a gaussian distribution
    #     """
    #     import scipy as spy

    #     self.data["labels_mean"], self.data["labels_std"] = {}, {}
    #     for label in self.data_dictionary["train_labels"].columns:
    #         if self.data_dictionary["train_labels"][label].dtype == object:
    #             continue
    #         f = spy.stats.norm.fit(self.data_dictionary["train_labels"][label])
    #         self.data["labels_mean"][label], self.data["labels_std"][label] = f[0], f[1]

    #     # in case targets are classifications
    #     for label in self.unique_class_list:
    #         self.data["labels_mean"][label], self.data["labels_std"][label] = 0, 0

    #     return

    # def remove_features_from_df(self, dataframe: DataFrame) -> DataFrame:
    #     """
    #     Remove the features from the dataframe before returning it to strategy. This keeps it
    #     compact for Frequi purposes.
    #     """
    #     to_keep = [
    #         col
    #         for col in dataframe.columns
    #         if not col.startswith("%") or col.startswith("%%")
    #     ]
    #     return dataframe[to_keep]

    # def get_unique_classes_from_labels(self, dataframe: DataFrame) -> None:
    #     # self.find_features(dataframe)
    #     self.find_labels(dataframe)

    #     for key in self.label_list:
    #         if dataframe[key].dtype == object:
    #             self.unique_classes[key] = dataframe[key].dropna().unique()

    #     if self.unique_classes:
    #         for label in self.unique_classes:
    #             self.unique_class_list += list(self.unique_classes[label])

    # def save_backtesting_prediction(self, append_df: DataFrame) -> None:
    #     """
    #     Save prediction dataframe from backtesting to feather file format
    #     :param append_df: dataframe for backtesting period
    #     """
    #     full_predictions_folder = Path(
    #         self.full_path / self.backtest_predictions_folder
    #     )
    #     if not full_predictions_folder.is_dir():
    #         full_predictions_folder.mkdir(parents=True, exist_ok=True)

    #     append_df.to_feather(self.backtesting_results_path)

    # def get_backtesting_prediction(self) -> DataFrame:
    #     """
    #     Get prediction dataframe from feather file format
    #     """
    #     append_df = pd.read_feather(self.backtesting_results_path)
    #     return append_df

    # def check_if_backtest_prediction_is_valid(self, len_backtest_df: int) -> bool:
    #     """
    #     Check if a backtesting prediction already exists and if the predictions
    #     to append have the same size as the backtesting dataframe slice
    #     :param length_backtesting_dataframe: Length of backtesting dataframe slice
    #     :return:
    #     :boolean: whether the prediction file is valid.
    #     """
    #     path_to_predictionfile = Path(
    #         self.full_path
    #         / self.backtest_predictions_folder
    #         / f"{self.model_filename}_prediction.feather"
    #     )
    #     self.backtesting_results_path = path_to_predictionfile

    #     file_exists = path_to_predictionfile.is_file()

    #     if file_exists:
    #         append_df = self.get_backtesting_prediction()
    #         if len(append_df) == len_backtest_df and "date" in append_df:
    #             logger.info(
    #                 f"Found backtesting prediction file at {path_to_predictionfile}"
    #             )
    #             return True
    #         else:
    #             logger.info(
    #                 "A new backtesting prediction file is required. "
    #                 "(Number of predictions is different from dataframe length or "
    #                 "old prediction file version)."
    #             )
    #             return False
    #     else:
    #         logger.info(
    #             f"Could not find backtesting prediction file at {path_to_predictionfile}"
    #         )
    #         return False

    # def get_full_models_path(self, config: Config) -> Path:
    #     """
    #     Returns default FreqAI model path
    #     :param config: Configuration dictionary
    #     """
    #     freqai_config: dict[str, Any] = config["freqai"]
    #     return Path(
    #         config["user_data_dir"] / "models" / str(freqai_config.get("identifier"))
    #     )

    # def remove_special_chars_from_feature_names(
    #     self, dataframe: pd.DataFrame
    # ) -> pd.DataFrame:
    #     """
    #     Remove all special characters from feature strings (:)
    #     :param dataframe: the dataframe that just finished indicator population. (unfiltered)
    #     :return: dataframe with cleaned feature names
    #     """

    #     spec_chars = [":"]
    #     for c in spec_chars:
    #         dataframe.columns = dataframe.columns.str.replace(c, "")

    #     return dataframe

    # def buffer_timerange(self, timerange: TimeRange):
    #     """
    #     Buffer the start and end of the timerange. This is used *after* the indicators
    #     are populated.

    #     The main example use is when predicting maxima and minima, the argrelextrema
    #     function  cannot know the maxima/minima at the edges of the timerange. To improve
    #     model accuracy, it is best to compute argrelextrema on the full timerange
    #     and then use this function to cut off the edges (buffer) by the kernel.

    #     In another case, if the targets are set to a shifted price movement, this
    #     buffer is unnecessary because the shifted candles at the end of the timerange
    #     will be NaN and FreqAI will automatically cut those off of the training
    #     dataset.
    #     """
    #     buffer = self.freqai_config["feature_parameters"]["buffer_train_data_candles"]
    #     if buffer:
    #         timerange.stopts -= buffer * timeframe_to_seconds(self.config["timeframe"])
    #         timerange.startts += buffer * timeframe_to_seconds(self.config["timeframe"])

    #     return timerange
