"""Per-account memory preferences — the user's privacy levers.

The `memory_*_enabled` config flags are the OPERATOR kill switch (env-only, need
a restart). This module adds the USER switch, surfaced in Settings → Memory.

The effective setting is the AND of the two: a user can always turn a layer off,
but can never force one back on that the operator disabled. A missing row means
"all on", so existing accounts need no backfill.

Every read fails open to the config default so a prefs hiccup can't silently
disable someone's memory.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..config import get_settings
from ..database import SessionLocal
from ..models.db_models import MemoryPrefs

logger = logging.getLogger(__name__)

# The user-togglable layers. Values are the MemoryPrefs column names.
_FIELDS = ("recall_enabled", "record_enabled", "reflect_enabled", "graph_enabled")


def _row(db, owner_id: int) -> Optional[MemoryPrefs]:
    return db.query(MemoryPrefs).filter(MemoryPrefs.owner_id == owner_id).first()


def get_prefs(owner_id: Optional[int]) -> dict:
    """The user's raw switches (defaults all-on when they've never set them)."""
    prefs = {f: True for f in _FIELDS}
    if owner_id is None:
        return prefs
    db = SessionLocal()
    try:
        row = _row(db, owner_id)
        if row is not None:
            for f in _FIELDS:
                prefs[f] = bool(getattr(row, f, True))
    except Exception as exc:
        logger.warning(f"[MemoryPrefs] read failed, defaulting on: {exc}")
    finally:
        db.close()
    return prefs


def set_prefs(owner_id: int, updates: dict) -> dict:
    """Update only the switches present in `updates`; returns the stored row."""
    clean = {f: bool(updates[f]) for f in _FIELDS if f in updates and updates[f] is not None}
    db = SessionLocal()
    try:
        row = _row(db, owner_id)
        if row is None:
            row = MemoryPrefs(owner_id=owner_id)
            db.add(row)
        for f, v in clean.items():
            setattr(row, f, v)
        db.commit()
        logger.info(f"[MemoryPrefs] owner={owner_id} set {clean}")
        return {f: bool(getattr(row, f, True)) for f in _FIELDS}
    except Exception as exc:
        db.rollback()
        logger.warning(f"[MemoryPrefs] write failed: {exc}")
        raise
    finally:
        db.close()


def effective(owner_id: Optional[int]) -> dict:
    """What the chat path should actually do: user switch AND operator flag.

    `record`/`recall` have no operator flag of their own (episodic memory is the
    baseline), so they follow the user switch alone.
    """
    s = get_settings()
    p = get_prefs(owner_id)
    return {
        "recall_enabled": p["recall_enabled"],
        "record_enabled": p["record_enabled"],
        # Semantic recall + reflection are separately gated by the operator.
        "semantic_recall_enabled": p["recall_enabled"] and s.memory_semantic_recall_enabled,
        "reflect_enabled": p["reflect_enabled"] and s.memory_reflect_enabled,
        "graph_enabled": p["graph_enabled"] and s.memory_graph_enabled,
    }
