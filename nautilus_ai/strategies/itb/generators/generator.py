import joblib
import polars as pl
from enum import Enum, auto, unique
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Type
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import DataType
from nautilus_trader.common.actor import Actor, ActorConfig
from nautilus_ai.common.data import GeneratorData


@unique
class GeneratorType(Enum):
    """
    Enum for different types of generators, grouped by pipeline phase.
    """
    PREPARATION = auto() # For merging and other data prep
    FEATURE = auto()     # For feature generation
    LABEL = auto()       # For label generation
    MODEL = auto()       # For training and prediction(includes rolling predictions)
    SIGNAL = auto()      # For signal generation

    def __str__(self) -> str:
        return self.name.lower()


class GeneratorConfig(ActorConfig, kw_only=True):
    """
    Configuration for a generator.
    """
    pass


class Generator(Actor, ABC):
    """
    Abstract base class for all data generators.

    This class provides a common interface for both online and offline data generation.
    Subclasses must implement the `generate` method, which takes a Polars DataFrame
    and returns a modified DataFrame with new columns.

    The class also provides methods for saving and loading generators, and for
    interacting with the `nautilus_trader` framework in an online setting.
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__(config=config)
        self._type: Optional[GeneratorType] = None
        self.generator_id = None

    @abstractmethod
    def generate(self, data: GeneratorData) -> Tuple[pl.DataFrame, List[str]]:
        """
        Generates and appends new data to the input DataFrame.

        Parameters
        ----------
        data : GeneratorData
            An object containing:
            - df (pl.DataFrame): The input DataFrame containing the raw or processed data.
            - kwargs (dict): Additional parameters specific to the generator's logic,
            such as a `last_rows` parameter for online updates, or generator-specific settings.

        Returns
        -------
        Tuple[pl.DataFrame, List[str]]
            A tuple containing:
            - pl.DataFrame: The modified DataFrame with new columns added.
            - List[str]: The names of the new columns generated.
        """
        pass

    def save(self, path: str) -> None:
        """
        Persist the data generator to the specified file path using joblib.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "Generator":
        """
        Load the data generator from the specified file path using joblib.
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return obj

    def set_generator_type(self, type: GeneratorType) -> None:
        """Sets the type of this generator instance."""
        self._type = type
        
    # 
    # For Nautilus Trader
    # 
    def publish_generated_data(self, data: GeneratorData):
        """Publishes Data to the system."""
        # Ensure the data has the correct ID before publishing
        PyCondition.is_true(
            data.generator_id == self.generator_id,
            "Cannot send data from a generator instance that doesn't match the data's ID."
        )
        self.publish_data(
            data_type=DataType(GeneratorData, metadata={"generator_id": f"{self.generator_id}"}),
            data=data
        )
        self.log.info(f"Published data for {self.generator_id}", color=LogColor.CYAN)
           
    def on_start(self) -> None:
        """
        Actions to perform when the generator starts.
        """
        self.publish_generated_data(GeneratorData(generator_id=self.generator_id))
        self.subscribe_data(data_type=DataType(GeneratorData))
        self.log.info(f"{self._type} generator started with ID: {self.generator_id}.")

    def on_stop(self) -> None:
        """
        Actions to perform when the generator stops.
        """
        self.unsubscribe_data(data_type=DataType(GeneratorData))
        self.log.info(f"{self._type} generator stopped.")
        
    def on_data(self, data: Data) -> None:
        """
        Actions to perform when the generator stops.
        """
        PyCondition.not_none(data, "data")

        if isinstance(data, GeneratorData) and data.generator_id == self.generator_id:
            if data.df is not None or data.model is not None:
                # Use a new task to send the message to avoid blocking on_data
                self.generate(data)
            else:
                self.log.info(f"Received empty message for {self.generator_id}", color=LogColor.YELLOW)

            

# === Registry for all generators ===
_GENERATOR_REGISTRY: Dict[str, Type['Generator']] = {}


def register_generator(name: str):
    """
    A decorator to register a generator class with the factory.

    Parameters
    ----------
    name : str
        The name used to identify the generator class in the registry.
    """
    def wrapper(cls):
        _GENERATOR_REGISTRY[name] = cls
        return cls
    return wrapper


class GeneratorFactory:
    """
    A factory class for creating generator instances from the registry.
    """
    def create(generator_type: str, config: GeneratorConfig) -> Generator:
        """
        Creates a generator instance based on its registered name.

        Parameters
        ----------
        generator_type : str
            The name of the generator as registered in the factory.
        config : GeneratorConfig
            The configuration object for the generator.

        Returns
        -------
        Generator
            An initialized instance of the specified generator class.
        """
        if generator_type not in _GENERATOR_REGISTRY:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        generator_class = _GENERATOR_REGISTRY[generator_type]
        instance = generator_class(config)
        
        # Optionally set the GeneratorType, assuming a naming convention or a method
        # to determine the type from the class or config.
        # For example, we could infer it from the name:
        try:
            generator_type_enum = GeneratorType[generator_type.upper()]
            instance.set_generator_type(generator_type_enum)
        except KeyError:
            # Handle cases where the registry name doesn't match the enum
            pass

        return instance

    @staticmethod
    def get_registry() -> Dict[str, Type[Generator]]:
        """
        Accesses a copy of the current generator registry.

        Returns
        -------
        Dict[str, Type[Generator]]
            A dictionary of all registered generator classes.
        """
        return _GENERATOR_REGISTRY.copy()
