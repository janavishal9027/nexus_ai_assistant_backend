"""Kafka consumer worker: runs RAG ingestion from `nexus.document.uploaded`.

Idempotent (``ingest_document`` dedups + replaces chunks, so re-delivery is safe),
retries failed documents with a re-published event carrying an ``attempt`` count,
and routes documents that exhaust their retries to the DLQ. On success it emits
``nexus.document.indexed``.

Run modes:
  • in-process — started by the app's lifespan when ``rag_kafka_indexing`` is on
    (``start_in_process`` / ``stop_in_process``).
  • standalone — for real scale-out, run one or more copies as their own process:
        cd backend && python -m app.workers.rag_index_worker

See docs/semantic-embedding/11-implementation-roadmap.md (Phase 4).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

GROUP_ID = "nexus-rag-indexer"

_task: Optional[asyncio.Task] = None
_stop = False


async def _handle(evt: dict) -> None:
    from ..config import get_settings
    from ..database import SessionLocal
    from ..models.rag_models import Document
    from ..services import rag_events
    from ..services.rag_ingestion import ingest_document

    doc_id = evt.get("document_id")
    owner = evt.get("owner_id")
    attempt = int(evt.get("attempt", 0))
    if doc_id is None:
        return

    # ingest_document never raises — it records success/failure on the document.
    await ingest_document(doc_id, owner)

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        status = doc.status if doc else "failed"
    finally:
        db.close()

    if status == "completed":
        await rag_events.publish(rag_events.TOPIC_INDEXED, {
            "event_type": "document.indexed", "document_id": doc_id, "owner_id": owner})
        return

    retries = get_settings().rag_kafka_index_retries
    if attempt + 1 < retries:
        logger.warning(f"[RAG/Worker] doc {doc_id} failed (attempt {attempt + 1}/{retries}); retrying")
        await rag_events.publish_uploaded(doc_id, owner, attempt=attempt + 1)
    else:
        logger.error(f"[RAG/Worker] doc {doc_id} exhausted {retries} retries → DLQ")
        await rag_events.publish(rag_events.TOPIC_DLQ, {
            "event_type": "document.dlq", "document_id": doc_id, "owner_id": owner,
            "attempts": attempt + 1})


async def run_worker() -> None:
    """Consume the uploaded-document topic until stopped. Requires aiokafka + a
    reachable broker."""
    global _stop
    _stop = False
    from aiokafka import AIOKafkaConsumer  # type: ignore
    from ..services import rag_events

    consumer = AIOKafkaConsumer(
        rag_events.TOPIC_UPLOADED,
        bootstrap_servers=rag_events.bootstrap_servers(),
        group_id=GROUP_ID,
        enable_auto_commit=True,
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode()),
    )
    await consumer.start()
    await rag_events.start()      # producer for indexed / retry / DLQ events
    logger.info("[RAG/Worker] indexing worker started")
    try:
        async for msg in consumer:
            if _stop:
                break
            try:
                await _handle(msg.value or {})
            except Exception as exc:  # defensive — a handler bug shouldn't kill the loop
                logger.error(f"[RAG/Worker] handler error: {exc}")
    finally:
        await consumer.stop()
        logger.info("[RAG/Worker] indexing worker stopped")


def start_in_process() -> None:
    """Spawn the worker on the running event loop (called from lifespan)."""
    global _task
    if _task is None or _task.done():
        _task = asyncio.get_event_loop().create_task(run_worker())


async def stop_in_process() -> None:
    global _task, _stop
    _stop = True
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except Exception:
            pass
        _task = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass
