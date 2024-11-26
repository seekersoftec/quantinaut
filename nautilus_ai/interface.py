import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from typing import Any, Deque, Dict, List, Optional, Literal, Tuple


import datasieve.transforms as ds
import numpy as np
import pandas as pd
import psutil
from datasieve.pipeline import Pipeline
from datasieve.transforms import SKLearnWrapper
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from nautilus_trader.common.actor import Actor
from nautilus_trader.common import Environment
from nautilus_trader.model.data import Bar, BarSpecification, BarType, DataType

# from nautilus_trader.model.data.bar import Bar, BarSpecification
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.datetime import secs_to_nanos, unix_nanos_to_dt

from nautilus_ai.common.logging import Logger
from nautilus_ai.config import (
    FeatureParameters,
    INautilusAIModelConfig,
    ModelTrainingParameters,
)
from nautilus_ai.data import NautilusAIDataDrawer, NautilusAIDataKitchen
from nautilus_ai.exceptions import OperationalException
from nautilus_ai.common.utils import (
    bars_to_dataframe,
    make_bar_type,
    record_params,
    timeframe_to_seconds,
)

pd.options.mode.chained_assignment = None


class INautilusAIModel(Actor, ABC):
    """
    Base class for AI models within the Nautilus trading strategy framework.

    This class provides the foundational tools and configurations for training, prediction, and
    feature engineering. Derived classes such as `Base***PredictionModels` inherit from this class
    to implement specific AI modeling strategies.

    Attributes
    ----------
    config_info : INautilusAIModelConfig
        Configuration settings for the AI model.
    data_split_parameters : dict[str, Any]
        Parameters for splitting training and validation data.
    model_training_parameters : ModelTrainingParameters
        Training-specific parameters for the AI model.
    identifier : str
        Unique identifier for the model instance.
    retrain : bool
        Indicates whether the model requires retraining.
    first : bool
        Used for initialization tracking.
    save_backtest_models : bool
        Determines if backtest models should be saved.
    current_candle : datetime
        The current candle's timestamp for data alignment.
    scanning : bool
        Indicates if the model is in scanning mode.
    ft_params : FeatureParameters
        Parameters for feature engineering.
    corr_pairlist : list[str]
        List of pairs to include in correlation analysis.
    keras : bool
        Indicates if the model uses Keras.
    CONV_WIDTH : int
        Width parameter for convolution-based models.
    class_names : list[str]
        Used in classification models to store class names.
    continual_learning : bool
        Indicates if continual learning is enabled.
    corr_dataframes : dict[str, DataFrame]
        Cached correlation dataframes for performance optimization.
    get_corr_dataframes : bool
        Controls caching of correlation dataframes.
    metadata : dict[str, Any]
        Metadata loaded from disk.
    data_provider : Optional[DataProvider]
        Provider for accessing market data.
    activate_tensorboard : bool
        Enables TensorBoard logging if set to True.
    """

    def __init__(self, config: INautilusAIModelConfig) -> None:
        """
        Initialize the INautilusAIModel instance with the provided configuration.

        Parameters
        ----------
        config : INautilusAIModelConfig
            Configuration object for the AI model.
        """
        super().__init__(config=config)

        # Configuration and parameter initialization
        self.config_info: INautilusAIModelConfig = self.config
        self.data_split_parameters: dict[str, Any] = (
            self.config_info.data_split_parameters
        )

        self.model_training_parameters: ModelTrainingParameters = (
            ModelTrainingParameters()
            if self.config_info.model_training_parameters is None
            else self.config_info.model_training_parameters
        )
        self.identifier: str = self.config_info.identifier
        self.retrain = False
        self.first = True

        self.bar_spec: str = self.config_info.bar_spec

        self.environment = Environment(self.config_info.environment)
        self.log.debug(f"Using Environment: {self.environment}")

        # Path setup
        self.set_full_path()

        # Backtesting configurations
        self.save_backtest_models: bool = self.config_info.save_backtest_models
        if self.save_backtest_models:
            self.log.info("Backtesting module configured to save all models.")

        # Data drawer and initial candle setup
        self.data_drawer = NautilusAIDataDrawer(Path(self.full_path), self.config)
        self.current_candle: datetime = datetime.fromtimestamp(
            637887600, tz=timezone.utc
        )
        self.data_drawer.current_candle = self.current_candle

        # Feature and parameter initialization
        self.scanning = False

        self.ft_params: FeatureParameters = FeatureParameters()
        if self.config_info.feature_parameters is not None:
            self.ft_params: FeatureParameters = self.config_info.feature_parameters

        self.corr_instrument_list: List[str] = (
            self.ft_params.include_corr_instrument_list
        )

        self.keras: bool = self.config_info.keras
        if self.keras and self.ft_params.DI_threshold:
            self.ft_params.DI_threshold = 0
            self.log.warning(
                "DI threshold is not configured for Keras models yet. Deactivating."
            )

        self.CONV_WIDTH = max(1, self.config_info.conv_width)
        self.class_names: List[str] = []  # For classification subclasses
        self.pair_it = 0
        self.pair_it_train = 0
        self.total_instruments = len(self.config_info.instrument_ids_str)

        self.train_queue = self._set_train_queue()
        self.inference_time: float = 0
        self.train_time: float = 0
        self.begin_time: float = 0
        self.begin_time_train: float = 0
        # self.base_tf_seconds = timeframe_to_seconds(self.config.timeframe)
        self.continual_learning = self.config_info.continual_learning
        self.plot_features = max(0, self.ft_params.plot_feature_importances)
        self.corr_dataframes: dict[str, DataFrame] = {}
        self.get_corr_dataframes: bool = True  # Performance optimization
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # Metadata and data provider
        self.metadata: dict[str, Any] = (
            self.data_drawer.load_global_metadata_from_disk()
        )
        # self.data_provider: Optional[DataProvider] = None
        self.max_system_threads = max(int(psutil.cpu_count() * 2 - 2), 1)
        # self.can_short = True  # Updated in `start()` with strategy.can_short

        # Model and PCA checks
        self.model: Any = None
        if self.ft_params.principal_component_analysis and self.continual_learning:
            self.ft_params.principal_component_analysis = False
            self.log.warning(
                "User tried to use PCA with continual learning. Deactivating PCA."
            )

        # TensorBoard configuration
        self.activate_tensorboard: bool = self.config_info.activate_tensorboard

        # Save initial configuration
        record_params(config.dict(), self.full_path)

    #
    # Internal Actor Methods
    #

    def on_start(self) -> None:
        """
        Actions to be performed on model start.

        Entry point to the NautilusAIModel from a specific instrument, it will train a new model if
        necessary before making the prediction.

        """

        if self.config_info.instrument_ids_str is not None:
            for instrument_id_str in self.config_info.instrument_ids_str:
                instrument_id: InstrumentId = InstrumentId.from_str(instrument_id_str)
                self.instrument = self.cache.instrument(instrument_id)
                if self.instrument is None:
                    self.log.error(f"Could not find instrument for {instrument_id}")
                    self.stop()
                    return

                self.subscribe_quote_ticks(instrument_id)

                if self.bar_spec is not None:
                    self.subscribe_bars(
                        BarType.from_str(f"{instrument_id_str}-{self.bar_spec}")
                    )

        self.dataframe: DataFrame = DataFrame()
        self.metadata: dict = {
            "instrument": self.config_info.instrument_ids_str[0]
        }  # TODO: pass the proper values

        self.live = self.environment in (Environment.SANDBOX, Environment.LIVE)
        self.data_drawer.set_instrument_dict_info(self.metadata)
        
        # self.data_provider = strategy.dp
        # self.can_short = strategy.can_short

        if self.live:
            self.inference_timer("start")
            self.data_kitchen = NautilusAIDataKitchen(
                self.config_info, self.live, self.metadata["instrument"]
            )
            data_kitchen = self.start_live(
                dataframe, self.metadata, strategy, self.data_kitchen
            )
            dataframe = data_kitchen.remove_features_from_df(
                data_kitchen.return_dataframe
            )

        # For backtesting, each pair enters and then gets trained for each window along the
        # sliding window defined by "train_period_days" (training window) and "live_retrain_hours"
        # (backtest window, i.e. window immediately following the training window).
        # NautilusAI slides the window and sequentially builds the backtesting results before returning
        # the concatenated results for the full backtesting period back to the strategy.
        else:
            self.data_kitchen = NautilusAIDataKitchen(
                self.config_info, self.live, self.metadata["instrument"]
            )
            if not self.config_info.backtest_live_models:
                self.log.info(
                    f"Training {len(self.data_kitchen.training_timeranges)} timeranges"
                )
                data_kitchen = self.start_backtesting(
                    dataframe, metadata, self.data_kitchen, strategy
                )
                self.dataframe = data_kitchen.remove_features_from_df(
                    data_kitchen.return_dataframe
                )
            else:
                self.log.info("Backtesting using historic predictions (live models)")
                data_kitchen = self.start_backtesting_from_historic_predictions(
                    dataframe, metadata, self.data_kitchen
                )
                self.dataframe = data_kitchen.return_dataframe

        self.clean_up()
        if self.live:
            self.inference_timer("stop", self.metadata["instrument"])

    def on_stop(self) -> None:
        """
        Callback for subclasses to override, allowing for custom logic
        when shutting down resources upon receiving a SIGINT signal.

        This method performs the following actions:
        - Logs a shutdown message.
        - Signals all running threads to stop by setting the stop event.
        - Saves historical predictions to disk via the data drawer.
        - Optionally waits for all training threads to complete, depending on the
          `wait_for_training_iteration_on_reload` configuration.

        Raises
        ------
        None
        """
        self.log.info("Stopping NautilusAI...")

        # Signal threads to stop
        self._stop_event.set()

        # Release data provider resources
        self.data_provider = None

        # Save any unsaved historic predictions
        self.data_drawer.save_historic_predictions_to_disk()

        # Handle thread termination based on configuration
        if self.config_info.wait_for_training_iteration_on_reload:
            self.log.info("Waiting for the current training iteration to complete...")
            for thread in self._threads:
                thread.join()
            self.log.info("All threads have been successfully stopped.")
        else:
            self.log.warning(
                "Interrupting the current training iteration as "
                "'wait_for_training_iteration_on_reload' is set to False."
            )

    #
    # Methods used by internal Actor Methods
    #

    def _set_train_queue(self) -> Deque[str]:
        """
        Sets the training queue based on existing training timestamps or the provided instrument list.

        - If there are no instruments with prior training timestamps, the queue is initialized
          with the instruments listed in `instrument_ids_str` from the configuration.
        - If training timestamps exist, instruments are ordered by their most recent training timestamp,
          with untrained instruments added at the front of the queue.

        Returns:
        --------
        Deque[str]: A deque containing the instrument queue, ordered by training priority.
        """
        current_instrument_list: List[str] = self.config_info.instrument_ids_str

        # Check if the instrument dictionary is empty
        if not self.data_drawer.instrument_dict:
            self.log.info(
                f"Set fresh train queue from instrument list. Queue: {current_instrument_list}"
            )
            return deque(current_instrument_list)

        # Initialize the deque for the best training queue
        best_queue: Deque[str] = deque()

        # Sort instruments by training timestamp
        instrument_dict_sorted = sorted(
            self.data_drawer.instrument_dict.items(),
            key=lambda item: item[1]["trained_timestamp"],
        )

        # Add instruments with existing training timestamps to the queue
        for instrument, _ in instrument_dict_sorted:
            if instrument in current_instrument_list:
                best_queue.append(instrument)

        # Add instruments without training timestamps to the front of the queue
        for instrument in current_instrument_list:
            if instrument not in best_queue:
                best_queue.appendleft(instrument)

        self.log.info(
            f"Set existing queue from trained timestamps. Best approximation queue: {list(best_queue)}"
        )
        return best_queue

    def set_full_path(self) -> None:
        """
        Creates and sets the full path for the identifier.
        """
        # Ensure self.config_info.user_data_dir is treated as a Path
        user_data_path = Path(self.config_info.user_data_dir)

        # Construct the full path
        self.full_path = user_data_path / "models" / f"{self.identifier}"

        # Create the directory if it doesn't exist
        self.full_path.mkdir(parents=True, exist_ok=True)

    def clean_up(self) -> None:
        """
        Cleans up non-persistent objects to ensure they are properly
        released and eligible for garbage collection.

        This method explicitly nullifies attributes that are not intended
        to persist between coin iterations. While these objects should
        naturally be handled by the garbage collector, this explicit cleanup
        serves as a safeguard and provides clarity about their non-persistence.
        """
        self.model = None
        self.data_kitchen = None

    def start_scanning(self, *args, **kwargs) -> None:
        """
        Starts the scanning process by invoking `self._start_scanning` in a
        separate thread.

        This method creates a new thread to run the scanning process asynchronously,
        allowing the main thread to continue without blocking. The method passes any
        additional arguments (`args`) and keyword arguments (`kwargs`) to the
        `_start_scanning` method.

        Parameters:
        ----------

            *args:
                Positional arguments to be passed to the `_start_scanning` method.

            **kwargs:
                Keyword arguments to be passed to the `_start_scanning` method.
        """
        _thread = threading.Thread(
            target=self._start_scanning, args=args, kwargs=kwargs
        )
        self._threads.append(_thread)
        _thread.start()

    # Following methods which are overridden by user made prediction models.
    # See models/classification/BaseClassifierModel.py for an example.

    @abstractmethod
    def train(
        self,
        unfiltered_df: DataFrame,
        instrument: Instrument,
        data_kitchen: NautilusAIDataKitchen,
        **kwargs,
    ) -> Any:
        """
        Filters the training data and trains a model on it. This method
        relies on the data handler for storing, saving, loading, and analyzing data.

        Args:
            unfiltered_df (DataFrame): The full dataframe for the current training period.
            instrument (Instrument): Metadata for the trading instrument being used.
            data_kitchen (NautilusAIDataKitchen): A data management and analysis tool for the current instrument.
            **kwargs: Additional parameters for custom configurations.

        Returns:
            Any: The trained model, which can later be used for predictions (via `self.predict`).
        """
        pass

    @abstractmethod
    def fit(
        self,
        data_dictionary: Dict[str, Any],
        data_kitchen: NautilusAIDataKitchen,
        **kwargs,
    ) -> Any:
        """
        Fits a model to the data. This method allows for easy swapping of models
        (e.g., replacing CatBoostRegressor with LGBMRegressor) while handling
        data management internally.

        Args:
            data_dictionary (Dict[str, Any]): A dictionary containing the training and test data/labels.
            data_kitchen (NautilusAIDataKitchen): A data management and analysis tool associated with the current instrument.
            **kwargs: Additional parameters for fitting the model.

        Returns:
            Any: The fitted model.
        """
        pass

    @abstractmethod
    def predict(
        self, unfiltered_df: DataFrame, data_kitchen: NautilusAIDataKitchen, **kwargs
    ) -> Tuple[DataFrame, NDArray[np.int_]]:
        """
        Filters the prediction features from the data and generates predictions using
        the trained model.

        Args:
            unfiltered_df (DataFrame): The full dataframe for the current backtest period.
            data_kitchen (NautilusAIDataKitchen): Data management and analysis tool for the present pair.
            **kwargs: Additional parameters for prediction.

        Returns:
            tuple: A tuple containing:
                - predictions (DataFrame): The predicted values.
                - do_predict (NDArray[np.int_]): An array of 1s and 0s indicating where data was removed or uncertain.
        """
        pass
