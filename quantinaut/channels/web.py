# -*- coding: utf-8 -*-
"""
Web Channel

From https://github.com/karanpratapsingh/HyperTrade/
"""

from __future__ import annotations
import asyncio
import websockets
from typing import Any, Dict, Optional

from nautilus_trader.core.correctness import PyCondition
from quantinaut.channels.channel import ChannelConfig, Channel, ChannelType

# TODO: Create a web channel that uses websockets for real-time communication.
# Should use sqlite for storing messages and user data.

class WebChannelConfig(ChannelConfig, kw_only=True):
    """
    Configuration for a WebSocket-based communication channel.

    Parameters:
    -----------
    username : str
        Username for authenticating the web channel user.
    password : str
        Password for authenticating the web channel user.
    channel_type : ChannelType
        The type of the channel. Defaults to `ChannelType.WEB`.
    channel_id : Optional[str]
        Unique identifier for the channel. If not provided, it will be auto-generated.
    kwargs : dict
        Additional keyword arguments for advanced configuration.

    Example:
    --------
        config = WebChannelConfig(
            username="user1",
            password="pass123",
            channel_id="web-001",
            kwargs={"timeout": 30}
        )
    """
    username: str
    password: str
    channel_type: ChannelType = ChannelType.WEB
    channel_id: Optional[str] = None
    kwargs: dict = {}


class WebChannel(Channel):
    """
    WebChannel provides real-time communication using WebSockets.

    This class is intended for sending and receiving messages between clients and the trading system over the web. It can be extended to support features such as authentication, message persistence (e.g., with SQLite), and user management.

    Features:
    ---------
    - Real-time messaging via WebSockets
    - Designed for integration with web frontends and dashboards
    - Can be extended for persistent storage and user authentication

    References:
    -----------
    - https://github.com/karanpratapsingh/HyperTrade/

    """
    
    def __init__(self, config: WebChannelConfig):
        super().__init__(config)
        
        self._server = None  # Placeholder for WebSocket server instance
        self._loop = asyncio.get_event_loop()  # Event loop for async operations




async def echo(websocket):
    """
    Handles incoming WebSocket connections and echoes back received messages.
    """
    async for message in websocket:
        print(f"Received: {message}")
        await websocket.send(f"Echo: {message}")

async def main():
    """
    Starts the WebSocket server.
    """
    async with websockets.serve(echo, "localhost", 8090):
        print("WebSocket server started on ws://localhost:8090")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())