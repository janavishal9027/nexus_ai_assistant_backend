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
import hashlib
import logging
from typing import Optional

from ..database import SessionLocal
from ..config import get_settings
from ..models.rag_models import (
    Document, DocumentChunk, IngestionJob, KnowledgeBase, ANN_DIM,
)
from ..providers.embeddings import (
    EmbeddingProvider, embedding_provider_for_kb, resolve_embedding_provider,
    INPUT_PASSAGE,
)
from .rag_chunking import (
    extract_text, clean_text, chunk_document,
    extract_pages, build_paged_text, page_for_offset,
)
from .object_store import load_document_bytes, store_document_bytes
from . import rag_cache, rag_events

logger = logging.getLogger(__name__)

_EMBED_BATCH = 32          # chunks embedded per request (progress granularity)


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


async def _embed_and_store_chunks(
    db,
    *,
    document_id: int,
    knowledge_base_id: Optional[int],
    conversation_id: Optional[int],
    owner_id: Optional[int],
    children: list[dict],
    provider: EmbeddingProvider,
    job_id: Optional[int] = None,
) -> int:
    """Embed the child chunks and persist parents + children with metadata.

    Parents (whole sections, capped) are stored with ``is_parent=True`` and NO
    embedding — they're returned to the LLM via parent-expansion, never searched.
    Children are the searchable units: embedded, linked to their parent, and
    tagged with the embedding model/version (per-chunk provenance)."""
    if not children:
        return 0
    model_tag = f"{provider.platform}/{provider.model}"
    version_tag = getattr(provider, "version", "") or ""
    child_hashes = [_sha256(c["text"]) for c in children]

    # 1 ─ embed child texts ---------------------------------------------------
    # Chunk-level dedup: reuse embeddings of unchanged chunks from a PRIOR ingest
    # of this document (persists across restarts, unlike the in-process cache) —
    # only newly-added / edited chunks are re-embedded.
    reuse: dict[str, list[float]] = {}
    if get_settings().rag_dedup_by_hash:
        for h, emb in (db.query(DocumentChunk.content_hash, DocumentChunk.embedding)
                       .filter(DocumentChunk.document_id == document_id,
                               DocumentChunk.is_parent.isnot(True),
                               DocumentChunk.embedding.isnot(None)).all()):
            if h and emb is not None and h not in reuse:
                reuse[h] = list(emb)

    vectors: list[Optional[list[float]]] = [reuse.get(h) for h in child_hashes]
    miss_idx = [i for i, v in enumerate(vectors) if v is None]
    reused = len(children) - len(miss_idx)
    if miss_idx:
        miss_texts = [children[i]["text"] for i in miss_idx]
        fresh: list[list[float]] = []
        for i in range(0, len(miss_texts), _EMBED_BATCH):
            fresh.extend(await provider.embed(miss_texts[i:i + _EMBED_BATCH],
                                              input_type=INPUT_PASSAGE))
            if job_id is not None:
                done = reused + min(i + _EMBED_BATCH, len(miss_texts))
                _update_job(db, job_id, embedded_chunks=done,
                            progress=min(25 + int(70 * done / len(children)), 95))
        if len(fresh) != len(miss_idx):
            raise ValueError("Embedding count did not match chunk count.")
        for k, i in enumerate(miss_idx):
            vectors[i] = fresh[k]
    if reused:
        logger.info(f"[Ingest] chunk-dedup reused {reused}/{len(children)} embeddings")

    # 2 ─ clear previous chunks for idempotent re-ingest ----------------------
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

    # 3 ─ insert unique parents first (to obtain their ids for the FK) --------
    parent_id_by_key: dict[int, int] = {}
    for c in children:
        pkey = c["parent_key"]
        if pkey in parent_id_by_key:
            continue
        prow = DocumentChunk(
            document_id=document_id, knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id, owner_id=owner_id,
            ordinal=pkey, text=c["parent_text"],
            token_count=max(1, len(c["parent_text"]) // 4),
            embedding=None, is_parent=True, section=c.get("section"),
            page_number=c.get("parent_page_number"),
            char_start=c.get("parent_char_start"), char_end=c.get("parent_char_end"),
            content_hash=_sha256(c["parent_text"]),
            embedding_model=model_tag, embedding_version=version_tag,
        )
        db.add(prow)
        db.flush()   # assigns prow.id without a full commit
        parent_id_by_key[pkey] = prow.id

    # 4 ─ insert children (embedded, linked to their parent) ------------------
    base_ordinal = len(parent_id_by_key)
    for idx, (c, vec) in enumerate(zip(children, vectors)):
        # Mirror into the HNSW-indexed column only when the dim matches ANN_DIM.
        ann = vec if (vec is not None and len(vec) == ANN_DIM) else None
        db.add(DocumentChunk(
            document_id=document_id, knowledge_base_id=knowledge_base_id,
            conversation_id=conversation_id, owner_id=owner_id,
            ordinal=base_ordinal + idx, text=c["text"],
            token_count=max(1, len(c["text"]) // 4), embedding=vec, embedding_ann=ann,
            is_parent=False, parent_chunk_id=parent_id_by_key.get(c["parent_key"]),
            section=c.get("section"), page_number=c.get("page_number"),
            char_start=c.get("char_start"), char_end=c.get("char_end"),
            content_hash=_sha256(c["text"]),
            embedding_model=model_tag, embedding_version=version_tag,
        ))
    return len(children)


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

        # 1 ─ extract + content-hash dedup -----------------------------------
        raw = load_document_bytes(doc)      # DB BYTEA or S3/MinIO (object-store seam)
        if not raw:
            raise ValueError("Uploaded file was empty.")
        new_hash = _sha256(raw)
        if (settings.rag_dedup_by_hash and doc.content_hash == new_hash
                and (doc.chunk_count or 0) > 0 and doc.status == "completed"):
            logger.info(f"[Ingest] Document {document_id} unchanged (hash match) — skipped")
            if job_id:
                _update_job(db, job_id, status="completed",
                            stage="Unchanged (cached)", progress=100)
            return

        doc.status = "processing"
        db.commit()
        if job_id:
            _update_job(db, job_id, status="parsing", stage="Extracting text", progress=5)
        # PDFs keep page boundaries (→ page numbers on chunks); others are flat.
        pages = extract_pages(doc.filename, raw)
        if pages:
            text, spans = build_paged_text(pages)
        else:
            text, spans = clean_text(extract_text(doc.filename, raw)), None
        if not text.strip():
            raise ValueError("No readable text could be extracted from this document.")

        # 2 ─ chunk (structure-aware parent/child) ----------------------------
        if job_id:
            _update_job(db, job_id, status="chunking", stage="Splitting into chunks", progress=15)
        children = chunk_document(
            text,
            child_size=settings.rag_child_chunk_size,
            parent_size=settings.rag_parent_chunk_size,
            overlap=settings.rag_chunk_overlap,
            structure_aware=settings.rag_structure_aware,
        )
        if not children:
            raise ValueError("Document produced no text chunks.")
        if spans:   # tag each chunk (and its parent) with a source page number
            for c in children:
                c["page_number"] = page_for_offset(c["char_start"], spans)
                c["parent_page_number"] = page_for_offset(c["parent_char_start"], spans)
        if job_id:
            _update_job(db, job_id, total_chunks=len(children))

        # 3 ─ embed + store (pinned to the KB's model once first set) ---------
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
        n = await _embed_and_store_chunks(
            db, document_id=document_id, knowledge_base_id=doc.knowledge_base_id,
            conversation_id=None, owner_id=owner_id, children=children,
            provider=provider, job_id=job_id,
        )

        # Pin the KB to this embedding space on first ingest.
        if kb is not None and not kb.embedding_model:
            kb.embedding_platform = provider.platform
            kb.embedding_model = provider.model
            kb.embedding_dim = provider.dim

        doc.status = "completed"
        doc.chunk_count = n
        doc.content_hash = new_hash
        doc.error = None
        db.commit()
        if doc.knowledge_base_id:
            rag_cache.invalidate(f"kb:{doc.knowledge_base_id}")
        if job_id:
            _update_job(db, job_id, status="completed", stage="Done",
                        progress=100, embedded_chunks=n, error=None)
        logger.info(f"[Ingest] Document {document_id} → {n} child chunks "
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
        children = chunk_document(
            text,
            child_size=settings.rag_child_chunk_size,
            parent_size=settings.rag_parent_chunk_size,
            overlap=settings.rag_chunk_overlap,
            structure_aware=settings.rag_structure_aware,
        )
        if not children:
            return

        provider = resolve_embedding_provider(db, owner_id)
        content_hash = _sha256(content)
        # Dedup: identical attachment already ingested in this conversation → skip.
        if settings.rag_dedup_by_hash:
            dup = (db.query(Document.id)
                   .filter(Document.conversation_id == conversation_id,
                           Document.content_hash == content_hash).first())
            if dup:
                logger.info(f"[ConvRAG] {filename} unchanged (hash match) — skipped")
                return

        doc = Document(
            conversation_id=conversation_id, knowledge_base_id=None, owner_id=owner_id,
            filename=filename[:512], size_bytes=len(content), status="completed",
            chunk_count=len(children), content_hash=content_hash,
            embedding_platform=provider.platform, embedding_model=provider.model,
            embedding_dim=None,
        )
        store_document_bytes(doc, content)   # DB BYTEA or S3/MinIO
        db.add(doc)
        db.commit()
        db.refresh(doc)
        n = await _embed_and_store_chunks(
            db, document_id=doc.id, knowledge_base_id=None,
            conversation_id=conversation_id, owner_id=owner_id,
            children=children, provider=provider,
        )
        doc.embedding_dim = provider.dim
        doc.chunk_count = n
        db.commit()
        rag_cache.invalidate(f"conv:{conversation_id}")
        logger.info(f"[ConvRAG] {filename} → {n} child chunks in conversation "
                    f"{conversation_id} ({provider.platform}/{provider.model})")
    except Exception as e:
        logger.warning(f"[ConvRAG] ingest failed for {filename}: {e}")
        db.rollback()
    finally:
        db.close()


async def dispatch_ingestion(background, document_id: int, owner_id: Optional[int]) -> None:
    """Route a document to ingestion: publish a Kafka ``document.uploaded`` event
    when event-driven indexing is enabled AND the broker is reachable, else run
    in-process via BackgroundTasks. Uploads never fail — any Kafka problem falls
    back to the in-process path."""
    if rag_events.indexing_enabled():
        if await rag_events.publish_uploaded(document_id, owner_id):
            return
    background.add_task(ingest_document, document_id, owner_id)


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
