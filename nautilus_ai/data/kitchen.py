import copy
import inspect
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from nautilus_ai.exceptions import OperationalException

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
