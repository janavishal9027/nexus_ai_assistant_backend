"""Memory lifecycle (Part D Phase 3) — retention, export, purge.

- **Retention** purges old EPISODIC rows (the raw Q&A log grows unbounded);
  SEMANTIC skills are distilled + durable and never expire.
- **Export** bundles a user's full memory as JSON (data portability / GDPR).
- **Purge** is the per-user "forget me" — owner-scoped and transactional.

All operations are owner-scoped; retention is a global time-based sweep.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..database import SessionLocal
from ..models.db_models import (KgEdge, MemoryChunk, MemoryEdge, MessageFeedback,
                                Skill)

logger = logging.getLogger(__name__)

SCOPES = ("all", "episodic", "skills", "feedback", "graph", "knowledge")
_SCOPES = SCOPES          # backwards-compatible alias


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def summary(owner_id: int) -> dict:
    """Row counts for a user's memory (for a settings/privacy view)."""
    db = SessionLocal()
    try:
        return {
            "episodic": db.query(MemoryChunk).filter(MemoryChunk.owner_id == owner_id).count(),
            "skills": db.query(Skill).filter(Skill.owner_id == owner_id).count(),
            "feedback": db.query(MessageFeedback).filter(MessageFeedback.owner_id == owner_id).count(),
            "graph": db.query(MemoryEdge).filter(MemoryEdge.owner_id == owner_id).count(),
            "knowledge": db.query(KgEdge).filter(KgEdge.owner_id == owner_id).count(),
        }
    finally:
        db.close()


def export_memory(owner_id: int) -> dict:
    """A user's full memory as a JSON-serializable dict — episodic + skills +
    feedback + the personal memory graph. Embeddings are excluded (not portable
    across models); the raw text they were derived from is included."""
    db = SessionLocal()
    try:
        episodes = (db.query(MemoryChunk).filter(MemoryChunk.owner_id == owner_id)
                    .order_by(MemoryChunk.id).all())
        skills = (db.query(Skill).filter(Skill.owner_id == owner_id)
                  .order_by(Skill.id).all())
        feedback = (db.query(MessageFeedback).filter(MessageFeedback.owner_id == owner_id)
                    .order_by(MessageFeedback.id).all())
        edges = (db.query(MemoryEdge).filter(MemoryEdge.owner_id == owner_id)
                 .order_by(MemoryEdge.support_count.desc()).all())
        kg = (db.query(KgEdge).filter(KgEdge.owner_id == owner_id)
              .order_by(KgEdge.support_count.desc()).all())
        return {
            "owner_id": owner_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "counts": {"episodic": len(episodes), "skills": len(skills),
                       "feedback": len(feedback), "graph": len(edges),
                       "knowledge": len(kg)},
            "episodic": [{
                "id": e.id, "text": e.text, "conversation_id": e.conversation_id,
                "created_at": _iso(e.created_at),
            } for e in episodes],
            "skills": [{
                "id": s.id, "kind": s.kind, "content": s.content, "polarity": s.polarity,
                "confidence": s.confidence, "support_count": s.support_count,
                "created_at": _iso(s.created_at), "updated_at": _iso(s.updated_at),
            } for s in skills],
            "feedback": [{
                "id": f.id, "conversation_id": f.conversation_id,
                "message_index": f.message_index, "rating": f.rating,
                "assistant_text": f.assistant_text, "created_at": _iso(f.created_at),
            } for f in feedback],
            "graph": [{
                "source": e.source, "relation": e.relation, "target": e.target,
                "source_type": e.source_type, "target_type": e.target_type,
                "support_count": e.support_count, "updated_at": _iso(e.updated_at),
            } for e in edges],
            "knowledge": [{
                "source": e.source, "relation": e.relation, "target": e.target,
                "source_type": e.source_type, "target_type": e.target_type,
                "support_count": e.support_count,
                "project_id": e.project_id, "conversation_id": e.conversation_id,
                "updated_at": _iso(e.updated_at),
            } for e in kg],
        }
    finally:
        db.close()


def purge_memory(owner_id: int, scope: str = "all") -> dict:
    """Delete a user's memory. scope ∈ SCOPES — "all" wipes every layer below,
    including the personal memory graph. Returns per-layer deleted counts.

    Raises ValueError on an unknown scope rather than coercing it to "all": this
    is irreversible, so a caller's typo must never be read as "delete it all".
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown purge scope '{scope}'. Expected one of: "
                         f"{', '.join(SCOPES)}")
    db = SessionLocal()
    try:
        counts: dict[str, int] = {}
        if scope in ("all", "episodic"):
            counts["episodic"] = db.query(MemoryChunk).filter(
                MemoryChunk.owner_id == owner_id).delete(synchronize_session=False)
        if scope in ("all", "skills"):
            counts["skills"] = db.query(Skill).filter(
                Skill.owner_id == owner_id).delete(synchronize_session=False)
        if scope in ("all", "feedback"):
            counts["feedback"] = db.query(MessageFeedback).filter(
                MessageFeedback.owner_id == owner_id).delete(synchronize_session=False)
        if scope in ("all", "graph"):
            counts["graph"] = db.query(MemoryEdge).filter(
                MemoryEdge.owner_id == owner_id).delete(synchronize_session=False)
        if scope in ("all", "knowledge"):
            counts["knowledge"] = db.query(KgEdge).filter(
                KgEdge.owner_id == owner_id).delete(synchronize_session=False)
        db.commit()
        logger.info(f"[Lifecycle] purge owner={owner_id} scope={scope} counts={counts}")
        return counts
    finally:
        db.close()


def apply_retention(retention_days: int) -> int:
    """Purge episodic memories older than ``retention_days`` (global sweep).
    Skills/feedback are kept. Returns the number of rows deleted; 0 disables."""
    if retention_days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    db = SessionLocal()
    try:
        n = db.query(MemoryChunk).filter(
            MemoryChunk.created_at < cutoff).delete(synchronize_session=False)
        db.commit()
        if n:
            logger.info(f"[Lifecycle] retention purged {n} episodic memories "
                        f"older than {retention_days}d")
        return n
    finally:
        db.close()
