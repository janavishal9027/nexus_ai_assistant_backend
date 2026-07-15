"""Kafka event-driven RAG indexing (Phase 4, opt-in).

On upload we publish a ``nexus.document.uploaded`` event; a consumer worker
(``app/workers/rag_index_worker.py``) runs ingestion, emits
``nexus.document.indexed`` / ``.failed``, retries, and routes documents that keep
failing to ``nexus.document.dlq``. Everything degrades to the in-process
BackgroundTasks path when Kafka is off or unreachable, so uploads never fail.

``aiokafka`` is imported lazily so this module always imports without the package.
See docs/semantic-embedding/11-implementation-roadmap.md (Phase 4).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

TOPIC_UPLOADED = "nexus.document.uploaded"
TOPIC_INDEXED = "nexus.document.indexed"
TOPIC_FAILED = "nexus.document.failed"
TOPIC_DLQ = "nexus.document.dlq"
_PUBLISH_TIMEOUT_S = 5.0

try:
    from aiokafka import AIOKafkaProducer  # type: ignore
    HAS_AIOKAFKA = True
except Exception:  # pragma: no cover
    AIOKafkaProducer = None  # type: ignore
    HAS_AIOKAFKA = False

_producer: Optional[Any] = None
_started = False


def bootstrap_servers() -> str:
    from ..config import get_settings
    s = get_settings()
    return s.rag_kafka_bootstrap or s.kafka_bootstrap_servers


async def start() -> None:
    """Start the document-event producer if event-driven indexing is enabled."""
    global _producer, _started
    from ..config import get_settings
    if not get_settings().rag_kafka_indexing:
        return
    if not HAS_AIOKAFKA:
        raise RuntimeError("rag_kafka_indexing enabled but aiokafka is not installed")
    _producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers(),
        acks="all", linger_ms=50,
        value_serializer=lambda v: json.dumps(v, default=str).encode(),
    )
    await _producer.start()          # raises if the broker is unreachable
    _started = True
    logger.info("[RAG/Kafka] document-event producer started")


async def stop() -> None:
    global _producer, _started
    if _producer is not None:
        try:
            await _producer.stop()
        except Exception:
            pass
    _producer = None
    _started = False


def indexing_enabled() -> bool:
    """True only when the producer actually started (broker was reachable)."""
    return _started and _producer is not None


async def publish(topic: str, payload: dict[str, Any]) -> bool:
    if _producer is None:
        return False
    msg = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
    try:
        await asyncio.wait_for(_producer.send_and_wait(topic, msg), timeout=_PUBLISH_TIMEOUT_S)
        return True
    except Exception as exc:
        logger.warning(f"[RAG/Kafka] publish {topic} failed: {exc}")
        return False


async def publish_uploaded(document_id: int, owner_id: Optional[int], attempt: int = 0) -> bool:
    return await publish(TOPIC_UPLOADED, {
        "event_type": "document.uploaded",
        "document_id": document_id, "owner_id": owner_id, "attempt": attempt,
    })
