# -*- coding: utf-8 -*-
"""
Nautilus Trader - Communication Channel Module
"""
from __future__ import annotations
import asyncio
import random
from datetime import datetime
from enum import Enum, auto, unique
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import DataType
from nautilus_trader.common.actor import Actor, ActorConfig

@unique
class ChannelType(Enum):
    """
    Enum for different types of communication channels.
    """
    TELEGRAM = auto()
    DISCORD = auto()
    SLACK = auto()
    WHATSAPP = auto()
    WEBHOOK = auto()
    PUSH_NOTIFICATIONS = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    

class ChannelData(Data):
    """
    Data class for communication channels.

    Attributes:
    -----------
    channel_id : str
        The unique identifier for the channel. This is a string in the format
        '<channel_type>-<number>' (e.g., 'telegram-123456').
    message : str | None
        The message payload to send or receive. A value of `None` indicates
        a channel discovery message rather than a notification.
    data : Dict[str, Any] | None
        Additional metadata or parameters for the message.
    """
    def __init__(
        self,
        channel_id: str,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ):
        super().__init__()
        
        self.channel_id = channel_id
        self.message = message
        self.data = data
        self.timestamp = int(datetime.now().timestamp()) if timestamp is None else timestamp 

    
class ChannelConfig(ActorConfig, kw_only=True):
    """
    Configuration for a communication channel.

    Attributes:
    -----------
    channel_type : ChannelType
        The type of the channel (e.g., Telegram, Discord).
    channel_id : str | None
        The unique identifier for the channel. If not provided, a new one
        will be generated automatically with the format '<channel_type>-<number>'.
    """
    channel_type: ChannelType
    channel_id: Optional[str] = None


class Channel(Actor, ABC):
    """
    Abstract base class for communication channels.

    This class provides a structure for sending notifications and receiving commands.
    """

    def __init__(self, config: ChannelConfig):
        super().__init__(config=config)
        self.channel_type = config.channel_type 
        self.channel_id = None
        
        # ID generation and validation are part of initialization
        if self.channel_id is None:
            if config.channel_id is None:
                random_number = random.randint(100000, 999999)
                self.channel_id = f"{self.channel_type}-{random_number}"
            else:
                # Validate that the provided ID matches the channel type
                expected_prefix = f"{self.channel_type}-"
                PyCondition.is_true(
                    config.channel_id.startswith(expected_prefix),
                    f"Provided channel_id '{config.channel_id}' does not match channel_type '{self.channel_type}'"
                )
                self.channel_id = config.channel_id

    @abstractmethod
    async def send_message(self, message: str, **kwargs: Dict[str, Any]) -> None:
        """
        Sends a notification through the channel.

        Parameters:
        -----------
        message : str
            The message to send.
        kwargs : Dict[str, Any]
            Additional parameters for the notification.
        """
        pass

    @abstractmethod
    async def start_channel(self, **kwargs: Dict[str, Any]) -> None:
        """
        Handles a command received through the channel.

        Parameters:
        -----------
        kwargs : Dict[str, Any]
            Additional parameters for the command.
        """
        pass

    @abstractmethod
    async def stop_channel(self) -> None:
        """
        Handles a command received through the channel.

        Parameters:
        -----------
        kwargs : Dict[str, Any]
            Additional parameters for the command.
        """
        pass
    
    def publish_channel_data(self, data: ChannelData):
        """Publishes ChannelData to the system."""
        # Ensure the data has the correct ID before publishing
        PyCondition.is_true(
            data.channel_id == self.channel_id,
            "Cannot send data from a channel instance that doesn't match the data's ID."
        )
        self.publish_data(
            data_type=DataType(ChannelData, metadata={"channel_id": self.channel_id}),
            data=data
        )
        self.log.info(f"Published data for {self.channel_id}", color=LogColor.CYAN)
           
    def on_start(self) -> None:
        """
        Actions to perform when the channel starts.
        """
        self.publish_channel_data(ChannelData(channel_id=self.channel_id, message=None))
        self.subscribe_data(data_type=DataType(ChannelData))
        asyncio.create_task(self.start_channel())
        self.log.info(f"{self.channel_type} channel started with ID: {self.channel_id}.")

    def on_stop(self) -> None:
        """
        Actions to perform when the channel stops.
        """
        self.unsubscribe_data(data_type=DataType(ChannelData))
        asyncio.run(self.stop_channel())
        self.log.info(f"{self.channel_type} channel stopped.")
        
    def on_reset(self) -> None:
        """
        Actions to perform when the channel stops.
        """
        self.log.info(f"{self.channel_type} channel reset.")
        
    def on_dispose(self) -> None:
        """
        Actions to perform when the channel stops.
        """
        self.log.info(f"{self.channel_type} channel stopped.")
        
    def on_data(self, data: Data) -> None:
        """
        Actions to perform when the channel stops.
        """
        PyCondition.not_none(data, "data")

        if isinstance(data, ChannelData) and data.channel_id == self.channel_id:
            if data.message is not None:
                # Use a new task to send the message to avoid blocking on_data
                asyncio.create_task(self.send_message(message=data.message, data=data.data))
            else:
                self.log.info(f"Received empty message for {self.channel_id}", color=LogColor.YELLOW)

            