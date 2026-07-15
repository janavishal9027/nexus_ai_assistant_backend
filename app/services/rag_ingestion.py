"""Ingestion worker: turns an uploaded document into embedded, searchable chunks.

Runs as an in-process background task (enqueued via FastAPI ``BackgroundTasks``)
so no external queue is required to boot — but the whole unit of work is a
single ``ingest_document`` coroutine that a Redis/RQ/Celery worker could call
verbatim later. Progress and failures are persisted to the ``ingestion_jobs``
row so the client can poll status.

Flow (matches the spec's ingestion pipeline):
    extract text → clean → split into chunks → embed each chunk →
    insert chunks + vectors → mark document COMPLETED
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..database import SessionLocal
from ..config import get_settings
from ..models.rag_models import Document, DocumentChunk, IngestionJob, KnowledgeBase
from ..providers.embeddings import (
    embedding_provider_for_kb, resolve_embedding_provider, INPUT_PASSAGE,
)
from .rag_chunking import extract_text, clean_text, chunk_text

logger = logging.getLogger(__name__)

_EMBED_BATCH = 32          # chunks embedded per request (progress granularity)


def create_job(db, document: Document) -> IngestionJob:
    """Create the PENDING job row for a freshly uploaded document."""
    job = IngestionJob(
        document_id=document.id,
        knowledge_base_id=document.knowledge_base_id,
        owner_id=document.owner_id,
        status="pending",
        stage="Queued",
        progress=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _update_job(db, job_id: int, **fields) -> None:
    job = db.get(IngestionJob, job_id)
    if job is None:
        return
    for k, v in fields.items():
        setattr(job, k, v)
    db.commit()


async def ingest_document(document_id: int, owner_id: Optional[int]) -> None:
    """Full ingestion pipeline for one document. Never raises — failures are
    recorded on the document and its job."""
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            logger.warning(f"[Ingest] Document {document_id} vanished before ingest")
            return
        job = (
            db.query(IngestionJob)
            .filter(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.id.desc())
            .first()
        )
        job_id = job.id if job else None
        settings = get_settings()

        # 1 ─ extract ---------------------------------------------------------
        doc.status = "processing"
        db.commit()
        if job_id:
            _update_job(db, job_id, status="parsing", stage="Extracting text", progress=5)

        raw = bytes(doc.raw or b"")
        if not raw:
            raise ValueError("Uploaded file was empty.")
        text = clean_text(extract_text(doc.filename, raw))
        if not text.strip():
            raise ValueError("No readable text could be extracted from this document.")

        # 2 ─ chunk -----------------------------------------------------------
        if job_id:
            _update_job(db, job_id, status="chunking", stage="Splitting into chunks", progress=15)
        chunks = chunk_text(text, settings.rag_chunk_size, settings.rag_chunk_overlap)
        if not chunks:
            raise ValueError("Document produced no text chunks.")
        if job_id:
            _update_job(db, job_id, total_chunks=len(chunks))

        # 3 ─ embed (pinned to the KB's model once first set) -----------------
        kb = db.get(KnowledgeBase, doc.knowledge_base_id)
        provider = embedding_provider_for_kb(
            db, owner_id,
            getattr(kb, "embedding_platform", None),
            getattr(kb, "embedding_model", None),
            getattr(kb, "embedding_dim", None),
        )
        if job_id:
            _update_job(db, job_id, status="embedding",
                        stage=f"Embedding with {provider.platform}/{provider.model}", progress=25)

        vectors: list[list[float]] = []
        for i in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[i:i + _EMBED_BATCH]
            vectors.extend(await provider.embed(batch, input_type=INPUT_PASSAGE))
            done = min(i + _EMBED_BATCH, len(chunks))
            if job_id:
                _update_job(db, job_id, embedded_chunks=done,
                            progress=min(25 + int(70 * done / len(chunks)), 95))

        if len(vectors) != len(chunks):
            raise ValueError("Embedding count did not match chunk count.")

        # 4 ─ store -----------------------------------------------------------
        if job_id:
            _update_job(db, job_id, stage="Storing chunks", progress=96)
        # Idempotent re-ingest: clear any previous chunks for this document.
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        for ordinal, (chunk, vec) in enumerate(zip(chunks, vectors)):
            db.add(DocumentChunk(
                document_id=document_id,
                knowledge_base_id=doc.knowledge_base_id,
                owner_id=owner_id,
                ordinal=ordinal,
                text=chunk,
                token_count=max(1, len(chunk) // 4),
                embedding=vec,
            ))

        # Pin the KB to this embedding space on first ingest.
        if kb is not None and not kb.embedding_model:
            kb.embedding_platform = provider.platform
            kb.embedding_model = provider.model
            kb.embedding_dim = provider.dim or (len(vectors[0]) if vectors else None)

        doc.status = "completed"
        doc.chunk_count = len(chunks)
        doc.error = None
        db.commit()
        if job_id:
            _update_job(db, job_id, status="completed", stage="Done",
                        progress=100, embedded_chunks=len(chunks), error=None)
        logger.info(f"[Ingest] Document {document_id} → {len(chunks)} chunks "
                    f"({provider.platform}/{provider.model})")

    except Exception as exc:
        logger.warning(f"[Ingest] Document {document_id} failed: {exc}")
        _fail(document_id, str(exc))
    finally:
        db.close()


def _fail(document_id: int, message: str) -> None:
    """Best-effort failure marking in a fresh session (the working one may be
    poisoned by a rolled-back error)."""
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is not None:
            doc.status = "failed"
            doc.error = message[:1000]
        job = (
            db.query(IngestionJob)
            .filter(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.id.desc())
            .first()
        )
        if job is not None:
            job.status = "failed"
            job.stage = "Failed"
            job.error = message[:1000]
        db.commit()
    except Exception as exc:  # pragma: no cover
        logger.error(f"[Ingest] Could not record failure for {document_id}: {exc}")
    finally:
        db.close()


async def ingest_conversation_document(
    conversation_id: int, owner_id: Optional[int], filename: str, content: bytes,
) -> None:
    """Chunk + embed a chat-attached document into the per-conversation store
    (A.3) so it's retrievable on later turns. Best-effort; never raises."""
    db = SessionLocal()
    try:
        text = clean_text(extract_text(filename, content))
        if not text.strip():
            logger.info(f"[ConvRAG] {filename}: no extractable text; skipped")
            return
        settings = get_settings()
        chunks = chunk_text(text, settings.rag_chunk_size, settings.rag_chunk_overlap)
        if not chunks:
            return

        provider = resolve_embedding_provider(db, owner_id)
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), _EMBED_BATCH):
            vectors.extend(await provider.embed(chunks[i:i + _EMBED_BATCH],
                                                input_type=INPUT_PASSAGE))
        if len(vectors) != len(chunks):
            return

        doc = Document(
            conversation_id=conversation_id, knowledge_base_id=None, owner_id=owner_id,
            filename=filename[:512], size_bytes=len(content), status="completed",
            chunk_count=len(chunks), raw=content,
            embedding_platform=provider.platform, embedding_model=provider.model,
            embedding_dim=provider.dim or (len(vectors[0]) if vectors else None),
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        for ordinal, (chunk, vec) in enumerate(zip(chunks, vectors)):
            db.add(DocumentChunk(
                document_id=doc.id, conversation_id=conversation_id,
                knowledge_base_id=None, owner_id=owner_id, ordinal=ordinal,
                text=chunk, token_count=max(1, len(chunk) // 4), embedding=vec,
            ))
        db.commit()
        logger.info(f"[ConvRAG] {filename} → {len(chunks)} chunks in conversation "
                    f"{conversation_id} ({provider.platform}/{provider.model})")
    except Exception as e:
        logger.warning(f"[ConvRAG] ingest failed for {filename}: {e}")
        db.rollback()
    finally:
        db.close()


def enqueue_ingestion(document_id: int, owner_id: Optional[int]) -> None:
    """Schedule ingestion on the running event loop (used by BackgroundTasks,
    which may call this from a threadpool)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(ingest_document(document_id, owner_id))
            return
    except RuntimeError:
        pass
    asyncio.run(ingest_document(document_id, owner_id))
