"""WebSocket connection manager (req 2, 1.4, 1.5).

Manages up to 100 concurrent sessions. `register`/`unregister` mutate a single
dict so they are atomic under the asyncio event loop. `send` is fire-and-forget
and closes the session on send error. A keepalive loop pings idle sessions and
closes any that fail to produce activity within the ping window.

To avoid two concurrent `receive` calls on the same socket, the *endpoint* owns
receiving: it calls `touch(session_id)` on every inbound message (including
`pong`). The keepalive loop only sends pings and closes stale sessions.
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WebSocketManager:
    MAX_SESSIONS: int = 100          # req 2.7
    IDLE_TIMEOUT_S: float = 1800.0   # 30 minutes (req 2.8)
    PING_WAIT_S: float = 10.0        # req 2.9
    _KEEPALIVE_TICK_S: float = 30.0

    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}
        self._last_activity: dict[str, float] = {}
        self._ping_sent_at: dict[str, float] = {}
        self._keepalive_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._keepalive_task is None:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def stop(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None

    def count(self) -> int:
        return len(self._sessions)

    def is_active(self, session_id: str) -> bool:
        return session_id in self._sessions

    def _now(self) -> float:
        return asyncio.get_event_loop().time()

    async def register(self, session_id: str, ws) -> None:
        """Register a socket; raise RuntimeError at capacity (req 2.7)."""
        if len(self._sessions) >= self.MAX_SESSIONS:
            raise RuntimeError("WebSocket session limit reached")
        self._sessions[session_id] = ws
        self._last_activity[session_id] = self._now()
        self._ping_sent_at.pop(session_id, None)
        self._observe(+1)
        logger.info(f"[WS] Registered session {session_id} ({len(self._sessions)} active)")

    async def unregister(self, session_id: str) -> None:
        """Atomically remove a session and release resources (req 1.5)."""
        existed = self._sessions.pop(session_id, None) is not None
        self._last_activity.pop(session_id, None)
        self._ping_sent_at.pop(session_id, None)
        if existed:
            self._observe(-1)
            logger.info(f"[WS] Unregistered session {session_id}")

    def touch(self, session_id: str) -> None:
        """Record inbound activity (called by the endpoint on every message)."""
        if session_id in self._sessions:
            self._last_activity[session_id] = self._now()
            self._ping_sent_at.pop(session_id, None)

    async def send(self, session_id: str, payload: dict) -> None:
        """Send a JSON payload; close the session on error (req 2.2-2.5)."""
        ws = self._sessions.get(session_id)
        if ws is None:
            return
        try:
            await ws.send_json(payload)
        except Exception as exc:
            logger.warning(f"[WS] Send failed for {session_id}: {exc}. Closing.")
            await self.unregister(session_id)
            await self._safe_close(ws)

    async def send_error_and_close(self, session_id: str, message: str) -> None:
        """Attempt to send an error then close regardless of delivery (req 2.6)."""
        ws = self._sessions.get(session_id)
        if ws is None:
            return
        try:
            await ws.send_json({"type": "error", "message": message})
        except Exception:
            pass
        await self.unregister(session_id)
        await self._safe_close(ws)

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._KEEPALIVE_TICK_S)
                now = self._now()
                for sid in list(self._sessions.keys()):
                    pinged = self._ping_sent_at.get(sid)
                    if pinged is not None:
                        # Awaiting a pong; close if the window elapsed (req 2.9).
                        if now - pinged > self.PING_WAIT_S:
                            logger.info(f"[WS] Ping timeout for {sid}; closing.")
                            ws = self._sessions.get(sid)
                            await self.unregister(sid)
                            await self._safe_close(ws)
                        continue
                    idle = now - self._last_activity.get(sid, now)
                    if idle > self.IDLE_TIMEOUT_S:
                        self._ping_sent_at[sid] = now
                        await self.send(sid, {"type": "ping"})
        except asyncio.CancelledError:
            pass

    @staticmethod
    async def _safe_close(ws) -> None:
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass

    @staticmethod
    def _observe(delta: int) -> None:
        try:
            from .observability import observability
            observability.inc_ws_sessions(delta)
        except Exception:
            pass


# Module-level singleton
ws_manager = WebSocketManager()
