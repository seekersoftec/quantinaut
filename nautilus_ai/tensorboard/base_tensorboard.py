from pathlib import Path
from typing import Any

from xgboost.callback import TrainingCallback
from nautilus_ai.common import Logger


logger = Logger(__name__)


class BaseTensorboardLogger:
    """
    Base class for logging scalar values to TensorBoard.

    Attributes:
        logdir (Path): Path to the directory where TensorBoard logs are stored.
        activate (bool): If False, disables logging.
    """

    def __init__(self, logdir: Path, activate: bool = True):
        """
        Initializes the TensorBoard logger.

        Args:
            logdir (Path): Directory path for storing logs.
            activate (bool): Whether to activate logging. Defaults to True.
        """
        self.logdir = logdir
        self.activate = activate

        if self.activate:
            logger.info(f"TensorBoard logging activated. Logs will be saved to {logdir}")
        else:
            logger.info("TensorBoard logging is deactivated.")

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        """
        Logs a scalar value to TensorBoard.

        Args:
            tag (str): The tag or name of the scalar.
            scalar_value (Any): The value to log.
            step (int): The training step associated with this value.
        """
        if self.activate:
            logger.debug(f"Logging scalar: tag={tag}, value={scalar_value}, step={step}")
            # Actual TensorBoard logging implementation would go here.

    def close(self):
        """
        Closes the logger and releases any associated resources.
        """
        if self.activate:
            logger.info("Closing TensorBoard logger.")
            # Add TensorBoard resource cleanup logic here if needed.


class BaseTensorBoardCallback(TrainingCallback):
    """
    Base class for integrating TensorBoard logging with XGBoost training callbacks.

    Attributes:
        logdir (Path): Path to the directory where TensorBoard logs are stored.
        activate (bool): If False, disables logging.
    """

    def __init__(self, logdir: Path, activate: bool = True):
        """
        Initializes the TensorBoard callback.

        Args:
            logdir (Path): Directory path for storing logs.
            activate (bool): Whether to activate logging. Defaults to True.
        """
        self.logdir = logdir
        self.activate = activate

        if self.activate:
            logger.info(f"TensorBoard callback activated. Logs will be saved to {logdir}")
        else:
            logger.info("TensorBoard callback is deactivated.")

    def after_iteration(self, model: Any, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        """
        Callback function executed after each iteration during training.

        Args:
            model (Any): The XGBoost model being trained.
            epoch (int): The current training epoch.
            evals_log (TrainingCallback.EvalsLog): Evaluation logs containing metrics.

        Returns:
            bool: Indicates whether training should continue. False means stop training.
        """
        if self.activate:
            logger.debug(f"After iteration: epoch={epoch}, evals_log={evals_log}")
            # Add TensorBoard logging logic for evaluation metrics here.

        return False  # Modify logic if training continuation depends on specific conditions.

    def after_training(self, model: Any) -> Any:
        """
        Callback function executed after training is completed.

        Args:
            model (Any): The XGBoost model that was trained.

        Returns:
            Any: The trained model, potentially modified.
        """
        if self.activate:
            logger.info("Training completed. Finalizing TensorBoard logs.")
            # Add TensorBoard resource finalization logic here if needed.

        return model
