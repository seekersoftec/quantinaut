from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from xgboost import callback

from nautilus_ai.common import Logger
from nautilus_ai.tensorboard.base_tensorboard import (
    BaseTensorBoardCallback,
    BaseTensorboardLogger,
)

logger = Logger(__name__)


class TensorboardLogger(BaseTensorboardLogger):
    """
    TensorBoard Logger for logging scalar values during training.

    Attributes:
        activate (bool): Whether logging to TensorBoard is enabled.
        writer (SummaryWriter): TensorBoard writer instance.
    """

    def __init__(self, logdir: Path, activate: bool = True):
        """
        Initializes the TensorBoard logger.

        Args:
            logdir (Path): Directory for saving TensorBoard logs.
            activate (bool): If False, logging is disabled. Default is True.
        """
        self.activate = activate
        self.writer = SummaryWriter(f"{str(logdir)}/tensorboard") if self.activate else None

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        """
        Logs a scalar value to TensorBoard.

        Args:
            tag (str): Tag for the scalar.
            scalar_value (Any): Value to log.
            step (int): Current step of training.
        """
        if self.activate:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        """
        Closes the TensorBoard writer, ensuring all logs are flushed.
        """
        if self.activate:
            self.writer.flush()
            self.writer.close()


class TensorBoardCallback(BaseTensorBoardCallback):
    """
    TensorBoard Callback for logging metrics during XGBoost training.

    Attributes:
        activate (bool): Whether logging to TensorBoard is enabled.
        writer (SummaryWriter): TensorBoard writer instance.
    """

    def __init__(self, logdir: Path, activate: bool = True):
        """
        Initializes the TensorBoard callback.

        Args:
            logdir (Path): Directory for saving TensorBoard logs.
            activate (bool): If False, logging is disabled. Default is True.
        """
        self.activate = activate
        self.writer = SummaryWriter(f"{str(logdir)}/tensorboard") if self.activate else None

    def after_iteration(
        self, model, epoch: int, evals_log: callback.TrainingCallback.EvalsLog
    ) -> bool:
        """
        Called after each training iteration to log evaluation metrics.

        Args:
            model: XGBoost model being trained.
            epoch (int): Current training epoch.
            evals_log (callback.TrainingCallback.EvalsLog): Log of evaluation metrics.

        Returns:
            bool: Always returns False, as required by XGBoost callbacks.
        """
        if not self.activate or not evals_log:
            return False

        eval_categories = ["validation", "train"]
        for (metric_name, logs), eval_category in zip(evals_log.items(), eval_categories, strict=False):
            for metric, log_values in logs.items():
                # Get the latest score from the log
                score = log_values[-1][0] if isinstance(log_values[-1], tuple) else log_values[-1]
                self.writer.add_scalar(f"{eval_category}-{metric}", score, epoch)

        return False

    def after_training(self, model):
        """
        Called after training is complete to flush and close the TensorBoard writer.

        Args:
            model: XGBoost model after training.

        Returns:
            The trained model.
        """
        if self.activate:
            self.writer.flush()
            self.writer.close()

        return model
