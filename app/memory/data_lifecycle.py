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
from ..models.db_models import MemoryChunk, Skill, MessageFeedback

logger = logging.getLogger(__name__)

_SCOPES = ("all", "episodic", "skills", "feedback")


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
        }
    finally:
        db.close()


def export_memory(owner_id: int) -> dict:
    """A user's full memory as a JSON-serializable dict (episodic + skills + feedback)."""
    db = SessionLocal()
    try:
        episodes = (db.query(MemoryChunk).filter(MemoryChunk.owner_id == owner_id)
                    .order_by(MemoryChunk.id).all())
        skills = (db.query(Skill).filter(Skill.owner_id == owner_id)
                  .order_by(Skill.id).all())
        feedback = (db.query(MessageFeedback).filter(MessageFeedback.owner_id == owner_id)
                    .order_by(MessageFeedback.id).all())
        return {
            "owner_id": owner_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "counts": {"episodic": len(episodes), "skills": len(skills), "feedback": len(feedback)},
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
        }
    finally:
        db.close()


def purge_memory(owner_id: int, scope: str = "all") -> dict:
    """Delete a user's memory. scope ∈ {all, episodic, skills, feedback}. Returns
    per-layer deleted counts."""
    if scope not in _SCOPES:
        scope = "all"
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
