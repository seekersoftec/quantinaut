try:
    # Try importing TensorBoard classes if available (likely when using the Torch version)
    from nautilus_ai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger

    # Assign to the standard names for TensorBoard logger and callback
    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    # Fallback to base classes when TensorBoard-specific classes are unavailable (non-Torch version)
    from nautilus_ai.tensorboard.base_tensorboard import (
        BaseTensorBoardCallback,
        BaseTensorboardLogger,
    )

    # Use base logger and callback if the Torch-specific ones are not available
    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore

# Make the logger and callback available for import
__all__ = ("TBLogger", "TBCallback")
