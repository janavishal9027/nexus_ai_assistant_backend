"""Semantic memory (Part D Phase 2) — distilled skills / preferences / lessons
about the user, with vectors for recall.

The Reflector (skills_extractor) *upserts* skills here — a semantically
near-identical existing skill is REINFORCED (support++/confidence↑) rather than
duplicated. Recall injects the most relevant skills at turn start. Owner-scoped,
dimension-guarded, real embeddings — same machinery as episodic memory.
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from ..database import SessionLocal
from ..models.db_models import Skill, HAS_PGVECTOR
from ..providers.embeddings import (
    resolve_embedding_provider, INPUT_QUERY, INPUT_PASSAGE,
)

logger = logging.getLogger(__name__)

_RECALL_THRESHOLD = 0.3   # skills are broad — recall on a looser match than episodes


async def _embed(owner_id: Optional[int], texts: list[str], input_type: str) -> list[list[float]]:
    db = SessionLocal()
    try:
        provider = resolve_embedding_provider(db, owner_id)
        return await provider.embed(texts, input_type=input_type)
    finally:
        db.close()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _nearest_skill(db, owner_id, vec, dim):
    base = db.query(Skill).filter(
        Skill.owner_id == owner_id, Skill.embedding_dim == dim,
        Skill.embedding.isnot(None))
    if HAS_PGVECTOR:
        dist = Skill.embedding.cosine_distance(vec)
        row = base.add_columns(dist.label("d")).order_by(dist.asc()).first()
        return (1.0 - float(row[1]), row[0]) if row else None
    best = None
    for sk in base.limit(2000).all():
        emb = list(sk.embedding) if sk.embedding is not None else None
        if not emb:
            continue
        s = _cosine(vec, emb)
        if best is None or s > best[0]:
            best = (s, sk)
    return best


async def upsert_skill(owner_id: Optional[int], kind: str, content: str,
                       polarity: str = "neutral", source: str = "reflector") -> Optional[int]:
    """Insert a distilled skill, or reinforce a near-duplicate. Never raises."""
    content = str(content or "").strip()
    if not content or owner_id is None:
        return None
    from ..config import get_settings
    dedup = get_settings().memory_skill_dedup_threshold
    try:
        vecs = await _embed(owner_id, [content], INPUT_PASSAGE)
    except Exception as exc:
        logger.warning(f"[Semantic] embed(upsert) failed: {exc}")
        return None
    if not vecs:
        return None
    vec, dim = vecs[0], len(vecs[0])

    def _w():
        db = SessionLocal()
        try:
            nearest = _nearest_skill(db, owner_id, vec, dim)
            if nearest is not None and nearest[0] >= dedup:
                sk = nearest[1]
                sk.support_count = (sk.support_count or 1) + 1
                sk.confidence = min(1.0, (sk.confidence or 0.5) + 0.1)
                sk.updated_at = datetime.now(timezone.utc)
                if polarity and polarity != "neutral":
                    sk.polarity = polarity
                db.commit()
                return sk.id
            sk = Skill(owner_id=owner_id, kind=kind, content=content, polarity=polarity,
                       embedding=vec, embedding_dim=dim, confidence=0.6,
                       support_count=1, source=source)
            db.add(sk)
            db.commit()
            db.refresh(sk)
            return sk.id
        finally:
            db.close()

    try:
        return await asyncio.to_thread(_w)
    except Exception as exc:
        logger.warning(f"[Semantic] upsert write failed: {exc}")
        return None


async def search(owner_id: Optional[int], query: str, top_k: int = 3) -> list[dict]:
    """Recall the user's most relevant skills for the query. Owner-scoped."""
    if owner_id is None or not str(query or "").strip():
        return []
    try:
        qv = await _embed(owner_id, [query], INPUT_QUERY)
    except Exception as exc:
        logger.warning(f"[Semantic] embed(search) failed: {exc}")
        return []
    if not qv:
        return []
    qvec, dim = qv[0], len(qv[0])

    def _s():
        db = SessionLocal()
        try:
            base = db.query(Skill).filter(
                Skill.owner_id == owner_id, Skill.embedding_dim == dim,
                Skill.embedding.isnot(None))
            if HAS_PGVECTOR:
                dist = Skill.embedding.cosine_distance(qvec)
                rows = base.add_columns(dist.label("d")).order_by(dist.asc()).limit(top_k * 3).all()
                out = [(1.0 - float(d), sk) for sk, d in rows]
            else:
                out = []
                for sk in base.limit(2000).all():
                    emb = list(sk.embedding) if sk.embedding is not None else None
                    if emb:
                        out.append((_cosine(qvec, emb), sk))
                out.sort(key=lambda x: x[0], reverse=True)
            return [(s, sk) for s, sk in out if s >= _RECALL_THRESHOLD][:top_k]
        finally:
            db.close()

    try:
        scored = await asyncio.to_thread(_s)
    except Exception as exc:
        logger.warning(f"[Semantic] search failed: {exc}")
        return []
    return [{
        "skill_id": sk.id, "kind": sk.kind, "content": sk.content,
        "polarity": sk.polarity, "confidence": sk.confidence,
        "similarity": round(float(s), 6),
    } for s, sk in scored]


def list_skills(owner_id: Optional[int], limit: int = 100) -> list[dict]:
    """All of a user's skills, most-reinforced first (for a memory/settings view)."""
    if owner_id is None:
        return []
    db = SessionLocal()
    try:
        rows = (db.query(Skill).filter(Skill.owner_id == owner_id)
                .order_by(Skill.support_count.desc(), Skill.updated_at.desc())
                .limit(limit).all())
        return [{
            "skill_id": sk.id, "kind": sk.kind, "content": sk.content,
            "polarity": sk.polarity, "confidence": sk.confidence,
            "support_count": sk.support_count,
        } for sk in rows]
    finally:
        db.close()
