from enum import Enum
from typing import Any, Type

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from nautilus_ai.rl.base_environment import BaseActions


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics and episodic summary reports to TensorBoard.

    This callback enables logging hyperparameters, training metrics, and environment-specific
    metrics during the training process.

    Attributes:
        actions (Type[Enum]): Enum class defining the action space.
        model (Any): The RL model being trained.
    """

    def __init__(self, verbose: int = 1, actions: Type[Enum] = BaseActions):
        """
        Initializes the TensorboardCallback.

        Args:
            verbose (int): Verbosity level of the callback. Default is 1.
            actions (Type[Enum]): Enum defining the action space. Default is `BaseActions`.
        """
        super().__init__(verbose)
        self.actions = actions
        self.model: Any = None

    def _on_training_start(self) -> None:
        """
        Called at the start of training. Logs hyperparameters and metrics to TensorBoard.
        """
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            # Uncomment the following lines to log additional hyperparameters:
            # "gamma": self.model.gamma,
            # "gae_lambda": self.model.gae_lambda,
            # "batch_size": self.model.batch_size,
            # "n_steps": self.model.n_steps,
        }
        metric_dict = {
            "eval/mean_reward": 0,
            "rollout/ep_rew_mean": 0,
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
            "train/explained_variance": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        """
        Called at every step during training. Logs additional metrics to TensorBoard.

        Returns:
            bool: Whether to continue training.
        """
        local_info = self.locals.get("infos", [{}])[0]

        if hasattr(self.training_env, "envs"):
            tensorboard_metrics = self.training_env.envs[0].unwrapped.tensorboard_metrics
        else:
            # Handles multi-process environments
            tensorboard_metrics = self.training_env.get_attr("tensorboard_metrics")[0]

        # Log metrics from `local_info`
        for metric, value in local_info.items():
            if metric not in ["episode", "terminal_observation"]:
                self.logger.record(f"info/{metric}", value)

        # Log custom tensorboard metrics from the environment
        for category, metrics in tensorboard_metrics.items():
            for metric, value in metrics.items():
                self.logger.record(f"{category}/{metric}", value)

        return True
