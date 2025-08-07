# -*- coding: utf-8 -*-
"""
Telegram Channel
"""
from __future__ import annotations
import asyncio
import threading
from typing import Any, Dict, Optional

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.markdown import hbold

from nautilus_trader.core.correctness import PyCondition
from nautilus_ai.notifications.channel import ChannelConfig, Channel, ChannelType


class TelegramChannelConfig(ChannelConfig, kw_only=True):
    """
    Configuration for a Telegram communication channel.

    Attributes:
    -----------
    token : str
        The Telegram bot API token.
    chat_id : str
        The chat ID of the Telegram group or user to send notifications to.
    channel_type : ChannelType
        The type of the channel. Must be set to `ChannelType.TELEGRAM`.
    channel_id : str | None
        The specific channel ID. If not provided, it will be generated.
    message_prefix : str, optional
        A prefix to prepend to all messages sent via Telegram (default is an empty string).
    kwargs : dict
        Additional keyword arguments to pass to the `aiogram.Bot` constructor.
    """
    token: str
    chat_id: str
    channel_type: ChannelType = ChannelType.TELEGRAM
    channel_id: Optional[str] = None
    message_prefix: str = ""
    kwargs: dict = {}


class TelegramChannel(Channel):
    """
    A communication channel implementation for Telegram using the aiogram library.
    
        - https://gist.github.com/nafiesl/4ad622f344cd1dc3bb1ecbe468ff9f8a
    """

    def __init__(self, config: TelegramChannelConfig):
        """
        Initializes the Telegram channel.
        """
        PyCondition.not_none(config.token, "token")
        PyCondition.not_none(config.chat_id, "chat_id")
        
        PyCondition.not_empty(config.token, "token")
        PyCondition.not_empty(config.chat_id, "chat_id")
        
        super().__init__(config=config)
        # We enforce a consistent channel_id from the chat_id
        self.channel_id = f"{config.channel_type}-{config.chat_id}"
        
        self.bot = Bot(token=self.config.token, default=DefaultBotProperties(parse_mode="HTML"), **self.config.kwargs)
        self.dp = Dispatcher()
        
    async def send_message(self, message: str, **kwargs: Dict[str, Any]) -> None:
        """
        Sends a notification through the Telegram channel.

        Parameters:
        -----------
        message : str
            The message to send.
        kwargs : Dict[str, Any]
            Additional parameters for the notification.
        """
        full_message = f"{self.config.message_prefix}{message}"
        await self.bot.send_message(
            chat_id=str(self.config.chat_id), 
            text=full_message, 
            **kwargs
        )

    async def start_channel(self, **kwargs: Dict[str, Any]) -> None:
        """
        Handles a command received through the channel.
        
        This method is required by the base class, but commands are
        handled by the aiogram dispatcher's internal logic.
        """
        # Register command handlers
        self.dp.message.register(self._start_command, Command("start"))
        self.dp.message.register(self._stop_command, Command("stop"))
        self.dp.message.register(self._handle_incoming_message)

        def run_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.dp.start_polling(self.bot))

        threading.Thread(target=run_bot, daemon=True).start()

    async def stop_channel(self) -> None:
        """
        Shuts down the bot gracefully.
        """
        await self.dp.stop_polling()
        
    async def _start_command(self, message: Message):
        """Handles the /start command."""
        await message.answer(f"Hello, {hbold(message.from_user.full_name)}! \nWelcome to Nautilus AI.")
    
    async def _stop_command(self, message: Message):
        """Handles the /start command."""
        await message.answer(f"Shutting down Nautilus AI...")
    
    async def _handle_incoming_message(self, message: Message):
        """
        This method processes all incoming messages from the Telegram bot.
        It can be used to handle commands and other interactions.
        """
        self.log.info(f"Received message from {message.from_user.full_name}: {message.text}")
        # Here you could implement logic to publish a ChannelData object with the incoming message
        # self.publish_channel_data(ChannelData(channel_id=self.channel_id, message=message.text))
        await self.send_message(f"Received message from {message.from_user.full_name}: {message.text}")
        
    