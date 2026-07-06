"""Kafka event producer for agent lifecycle events (req 10).

Publishes to the `agent.events` topic with acks=all, linger 100 ms, and a
16 KB batch cap. Publishing is fire-and-forget: on timeout or broker error it
logs a WARNING with the correlation_id and returns without raising so the agent
request is never blocked (req 10.4, 10.5, 10.8). `aiokafka` is imported lazily
so this module always imports when the package is absent (req 15.8).
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

TOPIC_EVENTS = "agent.events"
PUBLISH_TIMEOUT_S = 5.0

try:
    from aiokafka import AIOKafkaProducer  # type: ignore
    HAS_AIOKAFKA = True
except Exception:  # pragma: no cover
    AIOKafkaProducer = None  # type: ignore
    HAS_AIOKAFKA = False


class KafkaProducer:
    """Fire-and-forget agent event producer."""

    def __init__(self, bootstrap_servers: str) -> None:
        if not HAS_AIOKAFKA:
            raise RuntimeError(
                "aiokafka is not installed but the kafka feature is enabled"
            )
        self._bootstrap = bootstrap_servers
        self._producer: Optional[Any] = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap,
            acks="all",                 # at-least-once (req 10.4)
            linger_ms=100,              # req 10.8
            max_batch_size=16384,       # 16 KB (req 10.8)
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
        )
        await self._producer.start()
        logger.info("[Kafka] Producer started")

    async def stop(self) -> None:
        if self._producer:
            try:
                await self._producer.stop()
            except Exception:
                pass

    async def publish(
        self,
        event_type: str,
        correlation_id: str,
        conversation_id: Optional[int],
        session_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Publish one event with a 5 s timeout; never raises (req 10.5)."""
        if self._producer is None:
            return
        message = {
            "event_type": event_type,
            "correlation_id": correlation_id,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        try:
            await asyncio.wait_for(
                self._producer.send_and_wait(TOPIC_EVENTS, message),
                timeout=PUBLISH_TIMEOUT_S,
            )
            try:
                from .observability import observability
                observability.inc_kafka_events()
            except Exception:
                pass
        except asyncio.TimeoutError:
            logger.warning(
                f"[Kafka] Publish timeout event={event_type} correlation_id={correlation_id} (event dropped)"
            )
        except Exception as exc:
            logger.warning(
                f"[Kafka] Publish failed event={event_type} correlation_id={correlation_id}: {exc} (event dropped)"
            )

    async def ping(self) -> bool:
        try:
            if self._producer is None:
                return False
            partitions = await asyncio.wait_for(
                self._producer.partitions_for(TOPIC_EVENTS), timeout=2.0
            )
            return partitions is not None
        except Exception:
            return False


# Singleton — assigned in main.py lifespan when the kafka flag is enabled.
kafka_producer: Optional[KafkaProducer] = None
