from typing import Dict, List
from decimal import Decimal
from nautilus_trader.config import ActorConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import BarType


class ModelTrainingParameters(Data):
    """
    Configuration for ``ModelTrainingParameters`` instances.

    This class encompasses parameters for configuring and optimizing machine learning
    model training. It supports general training parameters (e.g., LightGBM, XGBoost)
    and PyTorch-specific configurations, enabling flexible integration with different
    model libraries.

    Parameters
    ----------
    learning_rate : float, default 3e-4
        The learning rate used by the optimizer to update model parameters.
        For PyTorch, this value is passed directly to the optimizer.
    n_estimators : int, default 100
        Number of boosted trees to fit during training. Relevant for tree-based models
        like LightGBM or XGBoost.
    parallel_processing : dict | None
        Configuration for parallel processing, with optional keys including:
        - `n_jobs`: Number of parallel jobs to run.
        - `thread_count`: Number of threads.
        - `task_type`: Processing task type, e.g., 'cpu' or 'gpu'.

    ---PyTorch-Specific Parameters---
    These parameters focus on PyTorch model training and its associated trainer.

    model_kwargs : dict, default {}
        Additional parameters to be passed to the PyTorch model class.
    n_epochs : int | None, default 10
        The number of complete passes through the training dataset. This parameter
        defines how many times the model sees the entire training data during training.
        Overrides `n_steps`. Either `n_epochs` or `n_steps` must be set.
    n_steps : int | None, default None
        Total number of optimizer steps to perform during training. Ignored if
        `n_epochs` is set. Calculated as:
            n_epochs = n_steps / (n_obs / batch_size)
        where `n_obs` is the number of data points.
    batch_size : int, default 64
        Number of samples per batch used in the training process.

    Notes
    -----
    This class is designed to provide a unified configuration for machine learning
    model training, ensuring flexibility across different frameworks while maintaining
    compatibility with their respective parameter sets. For example:
    - Tree-based models (e.g., LightGBM, XGBoost) can use `n_estimators` and
      `parallel_processing` options.
    - PyTorch models can utilize `learning_rate`, `model_kwargs`, and trainer-specific
      parameters like `n_epochs` or `batch_size`.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_estimators: int = 100,
        parallel_processing: Dict[str, float] | None = None,
        model_kwargs: dict = {},
        n_epochs: int | None = 10,
        n_steps: int | None = None,
        batch_size: int = 64,
        ts_event: int = 0,
        ts_init: int = 0,
    ):
        self.learning_rate: float = learning_rate
        self.n_estimators: int = n_estimators
        self.parallel_processing: Dict[str, float] | None = parallel_processing

        # PyTorch-specific parameters
        self.model_kwargs: dict = model_kwargs
        self.n_epochs: int | None = n_epochs
        self.n_steps: int | None = n_steps
        self.batch_size: int = batch_size

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


class RLConfig:
    """
    Configuration for RLConfig instances, covering both general model training
    parameters and reinforcement learning-specific parameters.

    These parameters are specific to reinforcement learning and control agent behavior.

    Parameters
    ----------
    train_cycles : int
        The number of training cycles. Total training steps are calculated as `train_cycles * number of training data points`.
    max_trade_duration_candles : int
        Maximum duration (in candles) that a trade can last during training.
    model_type : str
        The RL model type, chosen from stable_baselines3 or SBcontrib libraries. Examples: 'PPO', 'A2C', 'DQN'.
    policy_type : str
        The type of policy architecture used by the RL model. Examples: 'MlpPolicy', 'CnnPolicy'.
    max_training_drawdown_pct : float, default 0.8
        The maximum drawdown percentage allowed during training.
    cpu_count : int
        Number of CPUs or threads dedicated to training. Default is total physical cores minus one.
    model_reward_parameters : float
        Parameters used in the custom reward function defined in the training environment.
    add_state_info : bool, default False
        Whether to include state information (e.g., trade duration, current profit, trade position) during training and inference.
    net_arch : dict
        Defines the network architecture for the RL model. Example: [128, 128] for two shared layers with 128 units each.
    randomize_starting_position : bool, default False
        Randomizes the starting point of each episode to prevent overfitting.
    drop_ohlc_from_features : bool, default False
        Excludes OHLC data from the feature set passed to the RL agent while still using it for the environment.
    progress_bar : bool, default False
        Displays a progress bar indicating the training status, elapsed time, and estimated time remaining.

    ---Notes---
    This class is designed to encapsulate key parameters for configuring RL training
    and models, providing flexibility for both hyperparameter tuning and environment-specific requirements.
    """

    def __init__(
        self,
        train_cycles: int,
        max_trade_duration_candles: int,
        model_type: str,
        policy_type: str,
        max_training_drawdown_pct: float = 0.8,
        cpu_count: int = 2,
        model_reward_parameters: float = 3,
        add_state_info: bool = False,
        net_arch: dict = {},
        randomize_starting_position: bool = False,
        drop_ohlc_from_features: bool = False,
        progress_bar: bool = False,
        ts_event: int = 0,
        ts_init: int = 0,
    ):

        self.train_cycles: int = train_cycles
        self.max_trade_duration_candles: int = max_trade_duration_candles
        self.model_type: str = model_type
        self.policy_type: str = policy_type
        self.max_training_drawdown_pct: float = max_training_drawdown_pct
        self.cpu_count: int = cpu_count
        self.model_reward_parameters: int = model_reward_parameters
        self.add_state_info: bool = add_state_info
        self.net_arch: dict = net_arch
        self.randomize_starting_position: bool = randomize_starting_position
        self.drop_ohlc_from_features: bool = drop_ohlc_from_features
        self.progress_bar: bool = progress_bar

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


class FeatureParameters(Data):
    """
    Configuration for FeatureParameters instances, covering both general model training
    parameters and reinforcement learning-specific parameters.

    These parameters are specific to reinforcement learning and control agent behavior.

    Parameters
    ----------
    feature_parameters : Integer, default 1
        Number of steps into the future for predictions.
    include_timeframes : List[String]
        List of timeframes to include in feature engineering.
    include_corr_instrumentlist : List[String]
        List of correlated instruments to include in feature generation.
    label_period_candles : Integer
        Number of candles used for label generation.
    include_shifted_candles : Integer
        Number of shifted candles to include as features.
    weight_factor : Float, default 0.4
        Weighting factor for feature importance.
    indicator_max_period_candles : Integer
        Maximum candles for indicator computation.
    indicator_periods_candles : List[Integer]
        Periods of candles for indicator computation.
    principal_component_analysis : Boolean, default False
        Enables PCA for dimensionality reduction.
    plot_feature_importances : Integer, default 0
        Number of top features to visualize for importance.
    DI_threshold : Float, default 0.9
        Dependency index threshold for feature inclusion.
    use_SVM_to_remove_outliers : Boolean
        Enables SVM-based outlier removal.
    svm_params : Dict
        Parameters for the SVM model.
    use_DBSCAN_to_remove_outliers : Boolean
        Enables DBSCAN-based outlier removal.
    noise_standard_deviation : Integer, default 0
        Standard deviation of noise for data augmentation.
    outlier_protection_percentage : Float, default 30
        Percentage of data protected from outlier removal.
    reverse_train_test_order : Boolean, default False
        Reverses the train-test split order.
    shuffle_after_split : Boolean, default False
        Shuffles data after train-test split.
    buffer_train_data_candles : Integer, default 0
        Additional candles used for training data buffering.

    ---Notes---
    This class is designed to encapsulate key parameters for configuring RL training
    and models, providing flexibility for both hyperparameter tuning and environment-specific requirements.
    """

    def __init__(
        self,
        include_timeframes: List[str] = [],
        include_corr_instrumentlist: List[str] = [],
        label_period_candles: int = 0,
        include_shifted_candles: int = 0,
        weight_factor: float = 0.4,
        indicator_max_period_candles: int = 0,
        indicator_periods_candles: List[int] = [2, 4],
        principal_component_analysis: bool = False,
        plot_feature_importances: int = 0,
        DI_threshold: float = 0,
        use_SVM_to_remove_outliers: bool = False,
        svm_params: dict = {},
        use_DBSCAN_to_remove_outliers: bool = False,
        noise_standard_deviation: int = 0,
        outlier_protection_percentage: float = 0.3,
        reverse_train_test_order: bool = False,
        shuffle_after_split: bool = False,
        buffer_train_data_candles: int = 0,
        ts_event: int = 0,
        ts_init: int = 0,
    ):
        self.include_timeframes: List[str] = include_timeframes
        self.include_corr_instrumentlist: List[str] = include_corr_instrumentlist
        self.label_period_candles: int = label_period_candles
        self.include_shifted_candles: int = include_shifted_candles
        self.weight_factor: float = weight_factor
        self.indicator_max_period_candles: int = indicator_max_period_candles
        self.indicator_periods_candles: List[int] = indicator_periods_candles
        self.principal_component_analysis: bool = principal_component_analysis
        self.plot_feature_importances: int = plot_feature_importances
        self.DI_threshold: float = DI_threshold
        self.use_SVM_to_remove_outliers: bool = use_SVM_to_remove_outliers
        self.svm_params: dict = svm_params
        self.use_DBSCAN_to_remove_outliers: bool = use_DBSCAN_to_remove_outliers
        self.noise_standard_deviation: int = noise_standard_deviation
        self.outlier_protection_percentage: float = outlier_protection_percentage
        self.reverse_train_test_order: bool = reverse_train_test_order
        self.shuffle_after_split: bool = shuffle_after_split
        self.buffer_train_data_candles: int = buffer_train_data_candles

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


class INautilusAIModelConfig(ActorConfig):
    """
    Configuration for ``INautilusAIModel`` instances.

    Parameters
    ----------
    instrument_ids : InstrumentId[]
        The instrument ID for the strategy.
    bar_type : BarType
        The bar type for the strategy.
    trade_size : Decimal
        The position size per trade.

    --- General Configuration Parameters ---

    train_period_days : PositiveInt, default 0
        Number of days to use for training data (width of the sliding window).
    backtest_period_days : PositiveFloat, default 7.0
        Days to infer from the trained model before retraining in backtesting.
    identifier : String
        Unique identifier for the model. Useful for saving/loading pre-trained models.
    live_retrain_hours : PositiveFloat, default 0.0
        Frequency of retraining during live runs. Default: retrains as often as possible.
    expiration_hours : PositiveInt, default 0
        Prediction validity duration. Default: models never expire.
    purge_old_models : Integer, default 2
        Number of models to retain on disk during live runs. Set 0 to keep all.
    save_backtest_models : Boolean, default False
        Save models during backtesting for reuse in subsequent runs.
    fit_live_predictions_candles : PositiveInt
        Candles used for computing target statistics from prediction data.
    continual_learning : Boolean, default False
        Enables incremental learning from the final state of the last trained model.
    write_metrics_to_disk : Boolean, default False
        Saves training and inference metrics (timings, CPU usage) to a JSON file.
    data_kitchen_thread_count : Integer, default 7
        Number of threads used in the data preprocessing pipeline.
    activate_tensorboard : Boolean, default True
        Activates TensorBoard for monitoring model performance.
    wait_for_training_iteration_on_reload : Boolean, default True
        Delays predictions until a full training iteration completes.

    --- Feature Parameters ---
    feature_parameters : ModelTrainingParameters, optional
        Parameters for training the model.

    --- Data Split Parameters ---

    train_interval_minutes : Integer, default 60
        Time interval between training sessions.
    data_split_parameters : Dict
        Parameters for custom data splitting logic.
    test_size : Float, default 0.25
        Fraction of data reserved for testing.
    shuffle : Boolean, default False
        Shuffle the dataset before splitting.

    --- Model Training Parameters ---

    model_training_parameters : FeatureParameters, optional
        The parameters used to engineer the feature set.

    --- Reinforcement Learning Parameters ---

    rl_config : RLConfig, optional
        Configuration for reinforcement learning.

    --- Additional Parameters ---

    keras : Boolean, default False
        Enables Keras compatibility for saving/loading models.
    conv_width : PositiveInt, default 2
        Width of the neural network input tensor.
    reduce_df_footprint : Boolean, default False
        Reduces RAM and disk usage by casting numeric data to smaller types.


    ---Notes---

    For more checkout => https://www.freqtrade.io/en/stable/freqai-parameter-table/
    """

    # Nautilus params
    instrument_ids_str: List[str] | None = None
    bar_spec: str | None = None
    trade_size: Decimal = Decimal("0.10")

    # General configuration parameters
    train_period_days: int = 0
    backtest_period_days: int = 7
    identifier: str = "no_id_provided"
    live_retrain_hours: int = 0
    expiration_hours: int = 0
    purge_old_models: int = 2
    save_backtest_models: bool = True
    fit_live_predictions_candles: int = 0
    continual_learning: bool = False
    write_metrics_to_disk: bool = False
    data_kitchen_thread_count: int = 0
    activate_tensorboard: bool = True
    wait_for_training_iteration_on_reload: bool = True

    # time_handler = TimeHandler(config={"timerange": "20220101-20231231"})
    # training_list, backtesting_list = time_handler.split_timerange("20220101-20231231", train_split=30, bt_split=7)
    # data_dict = time_handler.build_data_dictionary(train_df, test_df, train_labels, test_labels, train_weights, test_weights)
    timerange: str = ""  # "20220101-20231231"

    # We notice that users like to use exotic indicators where
    # they do not know the required timeperiod. Here we include a factor
    # of safety by multiplying the user considered "max" by 2.
    startup_candle_count: int = 20

    user_data_dir: str = "data"

    model_save_type: str = "joblib"

    # Feature parameters
    feature_parameters: FeatureParameters | None = None

    # Data split parameters
    train_interval_minutes: int = 60  # Interval between training sessions
    data_split_parameters: dict = {}
    test_size: float = 0.25
    shuffle: bool = False

    # Model training parameters
    model_training_parameters: ModelTrainingParameters | None = None

    # Reinforcement Learning parameters
    rl_config: RLConfig | None = None

    # Additional parameters
    keras: bool = False
    conv_width: int = 2
    reduce_df_footprint: bool = False
    extra_returns_per_train: dict = {}
