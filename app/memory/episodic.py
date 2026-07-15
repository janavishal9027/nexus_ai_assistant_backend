"""Episodic memory (Part D) — durable per-user Q&A log with REAL semantic
embeddings and pgvector search.

Replaces the old fake SHA-256 encoder + Python cosine + conversation-only scope
with the same real embedding provider the RAG pipeline uses (cached + retried).
Write-through: the row is committed first, then its vector is set, so a vector
hiccup never blocks a write. Search is **owner-scoped** (per authenticated
account) and **dimension-guarded** (never compares vectors from different models).
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Optional

from ..database import SessionLocal
from ..models.db_models import MemoryChunk, HAS_PGVECTOR
from ..providers.embeddings import (
    resolve_embedding_provider, INPUT_QUERY, INPUT_PASSAGE,
)

logger = logging.getLogger(__name__)


async def _embed(owner_id: Optional[int], texts: list[str], input_type: str) -> list[list[float]]:
    """Embed with the owner's real RAG embedding provider (cached + retried)."""
    db = SessionLocal()
    try:
        provider = resolve_embedding_provider(db, owner_id)
        return await provider.embed(texts, input_type=input_type)
    finally:
        db.close()


async def store(owner_id: Optional[int], conversation_id: Optional[int],
                texts: list[str], user_id: Optional[int] = None) -> list[int]:
    """Embed + persist memory rows (write-through). Returns the new ids; never
    raises — memory issues must not interrupt a conversation."""
    clean = [str(t).strip() for t in (texts or []) if str(t or "").strip()]
    if not clean:
        return []
    try:
        vecs = await _embed(owner_id, clean, INPUT_PASSAGE)
    except Exception as exc:
        logger.warning(f"[Memory] embed(store) failed: {exc}")
        return []
    if len(vecs) != len(clean):
        return []

    def _w():
        db = SessionLocal()
        try:
            ids = []
            for t, v in zip(clean, vecs):
                c = MemoryChunk(
                    text=t, conversation_id=conversation_id, owner_id=owner_id,
                    user_id=user_id, embedding=v, embedding_dim=len(v),
                )
                db.add(c)
                db.flush()
                ids.append(c.id)
            db.commit()
            return ids
        finally:
            db.close()

    try:
        return await asyncio.to_thread(_w)
    except Exception as exc:
        logger.warning(f"[Memory] store write failed: {exc}")
        return []


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


async def search(owner_id: Optional[int], query: str,
                 conversation_id: Optional[int] = None, top_k: int = 5,
                 threshold: Optional[float] = None, scope: str = "user") -> list[dict]:
    """Owner-scoped semantic recall. ``scope='user'`` searches all of the owner's
    memories (episodic recall across conversations); ``scope='conversation'``
    restricts to one conversation. Dimension-guarded so a model change never
    mixes vector spaces."""
    from ..config import get_settings
    # Need something to scope by: an owner (per-user recall) or, for owner-less
    # legacy/tool paths, a conversation.
    if not str(query or "").strip() or (owner_id is None and conversation_id is None):
        return []
    if threshold is None:
        threshold = get_settings().memory_similarity_threshold
    try:
        qv = await _embed(owner_id, [query], INPUT_QUERY)
    except Exception as exc:
        logger.warning(f"[Memory] embed(search) failed: {exc}")
        return []
    if not qv:
        return []
    qvec = qv[0]
    dim = len(qvec)

    def _s():
        db = SessionLocal()
        try:
            base = db.query(MemoryChunk).filter(
                MemoryChunk.embedding_dim == dim,
                MemoryChunk.embedding.isnot(None),
            )
            if owner_id is not None:
                base = base.filter(MemoryChunk.owner_id == owner_id)
                if scope == "conversation" and conversation_id is not None:
                    base = base.filter(MemoryChunk.conversation_id == conversation_id)
            elif conversation_id is not None:
                # owner-less fallback (legacy / tool without request context)
                base = base.filter(MemoryChunk.conversation_id == conversation_id)
            if HAS_PGVECTOR:
                dist = MemoryChunk.embedding.cosine_distance(qvec)
                rows = base.add_columns(dist.label("d")).order_by(dist.asc()).limit(top_k * 3).all()
                out = []
                for chunk, d in rows:
                    sim = 1.0 - float(d)
                    if sim >= threshold:
                        out.append((sim, chunk))
                return out[:top_k]
            # JSON fallback — Python cosine.
            rows = base.order_by(MemoryChunk.id.desc()).limit(2000).all()
            scored = []
            for r in rows:
                emb = list(r.embedding) if r.embedding is not None else None
                if not emb:
                    continue
                sim = _cosine(qvec, emb)
                if sim >= threshold:
                    scored.append((sim, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]
        finally:
            db.close()

    try:
        scored = await asyncio.to_thread(_s)
    except Exception as exc:
        logger.warning(f"[Memory] search failed: {exc}")
        return []
    return [{
        "memory_id": r.id, "text": r.text, "conversation_id": r.conversation_id,
        "similarity": round(float(sim), 6),
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for sim, r in scored]
