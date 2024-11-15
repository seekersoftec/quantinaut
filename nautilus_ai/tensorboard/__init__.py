# ensure users can still use a non-torch freqai version
try:
    from nautilus_ai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger

    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    from nautilus_ai.tensorboard.base_tensorboard import (
        BaseTensorBoardCallback,
        BaseTensorboardLogger,
    )

    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore

__all__ = ("TBLogger", "TBCallback")
