# -*- coding: utf-8 -*-
"""
Web Channel

Inspired by: https://github.com/karanpratapsingh/HyperTrade/
"""

from __future__ import annotations

import asyncio
from enum import Enum
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
import uvicorn
import socketio
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from socketio import AsyncServer

from nautilus_trader.core.correctness import PyCondition
from quantinaut.channels.channel import ChannelConfig, Channel, ChannelType


# ==============================
# Models
# ==============================
class Events(Enum):
    DataFrame = 'Event:DataFrame'
    GetDataFrame = 'Event:DataFrame:Get'
    GetBalance = 'Event:Balance:Get'
    GetPositions = 'Event:Positions:Get'
    GetStats = 'Event:Stats:Get'
    GetTrades = 'Event:Trades:Get'
    UpdateTradingEnabled = 'Event:Config:Update:TradingEnabled'
    GetConfigs = 'Event:Configs:Get'
    UpdateAllowedAmount = 'Event:Config:Update:AllowedAmount'
    CriticalError = 'Event:CriticalError'
    Order = 'Event:Order'
    Trade = 'Event:Trade'
    GetStrategies = 'Event:Strategies:Get'
    UpdateStrategies = 'Event:Strategies:Update'

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    uid: str
    access_token: str


# ==============================
# Configuration
# ==============================
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
    username: str = "quantinaut"
    password: str = "password"
    port: int = 8000
    jwt_secret: str = "jwt-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    channel_type: ChannelType = ChannelType.WEB
    channel_id: Optional[str] = None
    kwargs: dict = {}


# ==============================
# Utility Functions
# ==============================
def create_jwt_token(username: str, secret: str, algorithm: str, expire_minutes: int) -> str:
    expire_time = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    payload = {"sub": username, "exp": expire_time}
    return jwt.encode(payload, secret, algorithm=algorithm)


def verify_jwt_token(token: str, secret: str, algorithm: str) -> str:
    try:
        decoded = jwt.decode(token, secret, algorithms=[algorithm])
        return decoded.get("sub")
    except jwt.PyJWTError:
        return None


# ==============================
# WebChannel Implementation
# ==============================
class WebChannel(Channel):
    """
    Web-based channel for real-time communication.

    This class is intended for sending and receiving messages between clients and the trading system over the web. 
    It can be extended to support features such as authentication, message persistence (e.g., with SQLite), and user management.

    Features:
    ---------
    - Real-time messaging via Socketio
    - Designed for integration with web frontends and dashboards
    - Can be extended for persistent storage and user authentication

    """

    def __init__(self, config: WebChannelConfig):
        PyCondition.positive_int(config.port, "port")
        super().__init__(config)

        # self._loop = asyncio.get_event_loop()
        # self._server: Optional[uvicorn.Server] = None

        # FastAPI app
        self.app = FastAPI()
        self._configure_cors()

        # Socket.IO server
        self.sio = AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.socket_app = socketio.ASGIApp(self.sio, other_asgi_app=self.app)
        self.app.mount("/ws", self.socket_app)

        # Bind routes and socket events
        self._register_routes()
        self._register_socket_events()

    # ------------------------------
    # Initialization
    # ------------------------------
    def _configure_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self):
        @self.app.post("/api/login", response_model=LoginResponse)
        async def login(request: LoginRequest):
            if request.username != self.config.username or request.password != self.config.password:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

            token = create_jwt_token(
                username=request.username,
                secret=self.config.jwt_secret,
                algorithm=self.config.jwt_algorithm,
                expire_minutes=self.config.access_token_expire_minutes,
            )
            return {"uid": request.username, "access_token": token}

    def _register_socket_events(self):
        @self.sio.event
        async def connect(sid, environ, auth):
            user = None
            token = str(auth.get("token")) if auth else None
            if token == f"quantinaut:password":
                user = self.config.username
                # user = verify_jwt_token(token, self.config.jwt_secret, self.config.jwt_algorithm)
            else:
                raise ConnectionRefusedError("Missing or invalid token")
            
            
            if not user:
                # await self.sio.disconnect(sid)
                self.log.info(f'Client {sid} failed authentication.')
                raise ConnectionRefusedError("Authentication failed")
            
            # Save authenticated user in per-client session
            await self.sio.save_session(sid, {'user': user})
            self.log.info(f"User connected: {user}")

        @self.sio.event
        async def dataframe(sid, data):
            session = await self.sio.get_session(sid)
            user = session.get('user') if session else None
            msg = data.get("msg")

            if not msg or not user:
                return
            await self.sio.emit(Events.DataFrame.value, {"from": user, "msg": msg})

    # ------------------------------
    # Channel Lifecycle
    # ------------------------------
    async def send_message(self, message: str, **kwargs: Dict[str, Any]) -> None:
        pass

    async def start_channel(self, **kwargs: Dict[str, Any]) -> None:
        uvicorn.run(
            self.socket_app,
            host="0.0.0.0",
            port=self.config.port,
            reload=False,
        )

    async def stop_channel(self) -> None:
        # Implement graceful shutdown logic if needed
        self.log.info("Stopping WebChannel... Triggering graceful shutdown...")



# ==============================
# Run Server Directly
# ==============================
config = WebChannelConfig(username="quantinaut", password="password")
channel = WebChannel(config)
app = channel.socket_app 
# uvicorn quantinaut.channels.web:app --reload --port 8000
