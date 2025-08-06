from __future__ import annotations
from enum import Enum, IntEnum, auto, unique


@unique
class TradingDecision(IntEnum):
    """
    Decisions that a trading model can make.
    
    Attributes
    ----------
    NEUTRAL : int
        No position (hold).
    ENTER_LONG : int
        Open a long position.
    EXIT_LONG : int
        Close an existing long position.
    ENTER_SHORT : int
        Open a short position.
    EXIT_SHORT : int
        Close an existing short position.
    """
    NEUTRAL     = 0
    ENTER_LONG  = 1
    EXIT_LONG   = 2
    ENTER_SHORT = 3
    EXIT_SHORT  = 4

    def __str__(self) -> str:
        # e.g. str(TradingDecision.ENTER_LONG) -> "enter_long"
        return self.name.lower()

@unique
class SecurityType(Enum):
    STOCKS = "stocks"
    OPTIONS = "options"
    
class MLLearningType(str, Enum):
    """
    Defines the specific type of machine learning model being used.

    This enum is crucial for the SageStrategy to determine the correct internal
    logic to apply for training, prediction, and other model interactions.

    Attributes
    ----------
    CLASSIFICATION : str
        A supervised model that predicts a categorical label.
        (e.g., predicting 'buy', 'sell', or 'hold').
    REGRESSION : str
        A supervised model that predicts a continuous value.
        (e.g., predicting the next day's price).
    UNSUPERVISED : str
        A model that learns patterns from unlabeled data.
        (e.g., clustering, dimensionality reduction).
    REINFORCEMENT_LEARNING : str
        A model that learns to make decisions by taking actions in an
        environment to maximize a cumulative reward.
    """
    SUPERVISED = auto()
    UNSUPERVISED = auto()
    REINFORCEMENT = auto()
    
    def __str__(self):
        # Return a more human-readable string if needed
        return self.name.lower()

@unique
class MLTaskType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    CLUSTERING = auto()
    DIM_REDUCTION = auto()
    
    def __str__(self):
        # Return a more human-readable string if needed
        return self.name.lower()


@unique
class MLFramework(Enum):
    """
    Enum for different types of machine learning algorithms.
    """
    SCIKIT_LEARN = auto()
    TENSORFLOW = auto()
    PYTORCH = auto()
    STABLE_BASELINES3 = auto()

    def __str__(self):
        # Return a more human-readable string if needed
        return self.name.lower()
