import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nautilus_trader.core.data import Data
from nautilus_trader.config import PositiveInt, PositiveFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ClientId

np.random.seed(100)

class FeatureSetConfig(Data):
    """Configuration for a single feature generation step."""
    def __init__(
        self,
        column_prefix: str,
        generator: str,
        feature_prefix: str,
        config: Dict[str, Any] = {},
        ts_event: int = 0,
        ts_init: int = 0,
    ):
        self.column_prefix: str = column_prefix
        self.generator: str = generator
        self.feature_prefix: str = feature_prefix
        self.config: Dict[str, Any] = config
        
        self._ts_event = ts_event
        self._ts_init = ts_init

    @property
    def ts_event(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the data event occurred.

        Returns
        -------
        int

        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the object was initialized.

        Returns
        -------
        int

        """
        return self._ts_init

class LabelSetConfig(Data):
    """Configuration for a single label generation step."""
    def __init__(
        self,
        column_prefix: str,
        generator: str,
        feature_prefix: str,
        config: Dict[str, Any] = {},
        ts_event: int = 0,
        ts_init: int = 0,
    ):
        self.column_prefix: str = column_prefix
        self.generator: str = generator
        self.feature_prefix: str = feature_prefix
        self.config: Dict[str, Any] = config
        
        self._ts_event = ts_event
        self._ts_init = ts_init
        
    @property
    def ts_event(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the data event occurred.

        Returns
        -------
        int

        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the object was initialized.

        Returns
        -------
        int

        """
        return self._ts_init
        

class AlgorithmConfig(Data):
    """Configuration for a single ML algorithm."""
    def __init__(
        self,
        name: str, 
        algo: str, 
        params: Dict[str, Any] = {},
        train: Dict[str, Any] = {},
        ts_event: int = 0,
        ts_init: int = 0,
    ):
        self.name: str = name
        self.algo: str = algo
        self.params: Dict[str, Any] = params
        self.train: Dict[str, Any] = train
        
        self._ts_event = ts_event
        self._ts_init = ts_init
        
    @property
    def ts_event(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the data event occurred.

        Returns
        -------
        int

        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the object was initialized.

        Returns
        -------
        int

        """
        return self._ts_init


class SignalSetConfig(Data):
    """Configuration for a single signal generation step."""
    def __init__(
        self,
        generator: str,
        config: Dict[str, Any] = {},
        ts_event: int = 0,
        ts_init: int = 0,
    ):
        self.generator = generator
        self.config = config
 
        self._ts_event = ts_event
        self._ts_init = ts_init
        
    @property
    def ts_event(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the data event occurred.

        Returns
        -------
        int

        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the object was initialized.

        Returns
        -------
        int

        """
        return self._ts_init


class OutputSetConfig(Data):
    """Configuration for a single output action."""
    def __init__(
        self,
        generator: str,
        config: Dict[str, Any] = {},
        ts_event: int = 0,
        ts_init: int = 0,
    ):
        self.generator = generator
        self.config = config
        
        self._ts_event = ts_event
        self._ts_init = ts_init
        
    @property
    def ts_event(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the data event occurred.

        Returns
        -------
        int

        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the object was initialized.

        Returns
        -------
        int

        """
        return self._ts_init


class RollingPredictConfig(Data):
    """Configuration for rolling prediction parameters."""
    def __init__(
        self,
        data_start: Union[int, str],
        data_end: Optional[Union[int, str]],
        prediction_start: Optional[Union[int, str]],
        prediction_size: int,
        prediction_steps: int,
        use_multiprocessing: bool,
        max_workers: int,
        ts_event: int = 0,
        ts_init: int = 0,
    ):
        self.data_start = data_start
        self.data_end = data_end
        self.prediction_start = prediction_start
        self.prediction_size = prediction_size
        self.prediction_steps = prediction_steps
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers
        
        self._ts_event = ts_event
        self._ts_init = ts_init
        
    @property
    def ts_event(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the data event occurred.

        Returns
        -------
        int

        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        UNIX timestamp (nanoseconds) when the object was initialized.

        Returns
        -------
        int

        """
        return self._ts_init



class ITBConfig(StrategyConfig, frozen=True):
    """
    Configuration for ITB instances, tailored for ML/DL/RL models.

    This configuration provides a robust set of parameters to define the behavior
    of a machine learning-based trading strategy. It covers model loading,
    feature engineering, training, and inference.

    Parameters
    ----------
    bar_types : BarType
        BarType object representing the instrument and it's timeframe.
    client_id : ClientId
        The client ID for the strategy, used for logging and identification.
 
    """
    bar_type: BarType
    client_id: ClientId = ClientId("ITB-001")
    data_folder: Path = Path("./DATA_ITB")
    model_path: Union[Path, str, None] = None
    scale_data: bool = False
    scaler_path: Union[Path, str, None] = None

    # The number of past bars used to compute features for the current bar.
    features_horizon: PositiveInt = 120

    # The look-ahead period used to define the label (e.g., future return).
    # These last few records in a dataset should not be used for training,
    # as their labels might not be correctly determined.
    label_horizon: PositiveInt = 120

    # The number of historical records to use for offline model training.
    train_length: PositiveInt = 525600

    # The number of records to keep up-to-date for online prediction.
    # This is crucial for real-time inference.
    predict_length: PositiveInt = 288

    # The number of recent records to re-compute on each iteration of an online pipeline.
    append_overlap_records: PositiveInt = 5
    
    # Data Processing
    data_sources: List[Dict[str, str]]
    feature_sets: List[FeatureSetConfig]
    label_sets: List[LabelSetConfig]
    
    # Training
    train_feature_sets: List[Dict[str, Any]]
    train_features: List[str]
    labels: List[str]
    algorithms: List[AlgorithmConfig]

    # Signal Generation & Outputs
    rvi_period: PositiveInt = 9
    rvi_threshold: PositiveFloat = 50.0
    signal_sets: List[SignalSetConfig]
    output_sets: List[OutputSetConfig]

    # Advanced Models
    rolling_predict: RollingPredictConfig
    
    