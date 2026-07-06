"""Kafka command consumer + in-memory event buffer (req 10.6, 10.7, 18.3, 18.4).

`EventBuffer` is a pure-Python bounded ring buffer (no aiokafka dependency) that
backs the Real-Time Events tool. `KafkaConsumer` subscribes to `agent.commands`,
routes `notify_user` commands to the WebSocket_Manager, and pushes every consumed
message into the buffer. `aiokafka` is imported lazily so this module always
imports when the package is absent (req 15.8).
"""
import asyncio
import json
import logging
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)

TOPIC_COMMANDS = "agent.commands"
EVENT_BUFFER_DEFAULT_SIZE = 500

try:
    from aiokafka import AIOKafkaConsumer  # type: ignore
    HAS_AIOKAFKA = True
except Exception:  # pragma: no cover
    AIOKafkaConsumer = None  # type: ignore
    HAS_AIOKAFKA = False


class EventBuffer:
    """Bounded per-topic ring buffer, newest-first (req 18.4)."""

    def __init__(self, max_size: int = EVENT_BUFFER_DEFAULT_SIZE) -> None:
        self._buffers: dict[str, deque] = {}
        self._max_size = max_size

    def push(self, topic: str, event: dict) -> None:
        buf = self._buffers.get(topic)
        if buf is None:
            buf = deque(maxlen=self._max_size)
            self._buffers[topic] = buf
        if len(buf) >= self._max_size:
            # deque(maxlen) discards the oldest automatically; log per req 18.4.
            logger.warning(f"[EventBuffer] Buffer full for topic={topic}; discarding oldest event")
        buf.appendleft(event)   # newest-first

    def get_recent(self, topic: str, limit: int = 20) -> list[dict]:
        buf = self._buffers.get(topic)
        if not buf:
            return []
        return list(buf)[: min(limit, 100)]

    def topics(self) -> list[str]:
        return list(self._buffers.keys())


class KafkaConsumer:
    """Consumes agent.commands and feeds the EventBuffer."""

    def __init__(self, bootstrap_servers: str, ws_manager, event_buffer: EventBuffer) -> None:
        if not HAS_AIOKAFKA:
            raise RuntimeError(
                "aiokafka is not installed but the kafka feature is enabled"
            )
        self._bootstrap = bootstrap_servers
        self._ws_manager = ws_manager
        self._event_buffer = event_buffer
        self._consumer: Optional[Any] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            TOPIC_COMMANDS,
            bootstrap_servers=self._bootstrap,
            group_id="agent-gateway",
            value_deserializer=lambda b: json.loads(b.decode()),
            auto_offset_reset="latest",
        )
        await self._consumer.start()
        self._task = asyncio.create_task(self._consume_loop())
        logger.info("[Kafka] Consumer started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception:
                pass

    async def _consume_loop(self) -> None:
        try:
            async for msg in self._consumer:
                try:
                    data = msg.value
                    self._event_buffer.push(msg.topic, data)
                    if isinstance(data, dict) and data.get("command_type") == "notify_user":
                        session_id = data.get("session_id", "")
                        payload = data.get("payload", {})
                        await self._ws_manager.send(session_id, payload)
                except Exception as exc:
                    logger.error(f"[Kafka] Consumer message error: {exc}")
        except asyncio.CancelledError:
            pass


# Singletons. The EventBuffer is always available (pure Python) so the
# Real-Time Events tool can read from it even before the consumer starts.
event_buffer: EventBuffer = EventBuffer(max_size=EVENT_BUFFER_DEFAULT_SIZE)
kafka_consumer: Optional[KafkaConsumer] = None
