"""Project brain (Part D Phase 4) — a durable, auto-maintained store of project
facts / decisions / conventions / goals, extending the static Project.instructions.

The project Reflector distils project-level knowledge from the project's
conversations (debounced, background) and dedup-reinforces it here; ``render()``
feeds it into the system prompt for every chat in that project. Owner + project
scoped, real embeddings — same machinery as semantic memory.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Optional

from ..database import SessionLocal
from ..models.db_models import ProjectBrainEntry, HAS_PGVECTOR
from ..providers.embeddings import resolve_embedding_provider, INPUT_PASSAGE

logger = logging.getLogger(__name__)

_KINDS = {"fact", "decision", "convention", "goal"}
_turn_counts: dict[int, int] = {}
_MAX_TRACK = 2000

_REFLECT_SYSTEM = (
    "You maintain a project's BRAIN — durable facts, decisions, conventions, and "
    "goals about the PROJECT ITSELF (not the user's personal preferences, not "
    "transient task detail). From the conversation, extract only stable "
    "project-level knowledge worth remembering for future chats in this project. "
    'Return ONLY a JSON array (max 5). Each: {"kind":"fact"|"decision"|'
    '"convention"|"goal","content":"<short statement>"}. Return [] if nothing durable.'
)


async def _embed(owner_id: Optional[int], texts: list[str]) -> list[list[float]]:
    db = SessionLocal()
    try:
        provider = resolve_embedding_provider(db, owner_id)
        return await provider.embed(texts, input_type=INPUT_PASSAGE)
    finally:
        db.close()


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


async def add_entry(owner_id: Optional[int], project_id: Optional[int],
                    kind: str, content: str) -> Optional[int]:
    """Insert a brain entry, or reinforce a near-duplicate. Never raises."""
    content = str(content or "").strip()
    if not content or owner_id is None or project_id is None:
        return None
    from ..config import get_settings
    dedup = get_settings().memory_brain_dedup_threshold
    try:
        vecs = await _embed(owner_id, [content])
    except Exception as exc:
        logger.warning(f"[Brain] embed failed: {exc}")
        return None
    if not vecs:
        return None
    vec, dim = vecs[0], len(vecs[0])

    def _w():
        db = SessionLocal()
        try:
            base = db.query(ProjectBrainEntry).filter(
                ProjectBrainEntry.owner_id == owner_id,
                ProjectBrainEntry.project_id == project_id,
                ProjectBrainEntry.embedding_dim == dim,
                ProjectBrainEntry.embedding.isnot(None))
            near = None
            if HAS_PGVECTOR:
                dist = ProjectBrainEntry.embedding.cosine_distance(vec)
                row = base.add_columns(dist.label("d")).order_by(dist.asc()).first()
                if row:
                    near = (1.0 - float(row[1]), row[0])
            else:
                for e in base.limit(1000).all():
                    emb = list(e.embedding) if e.embedding is not None else None
                    if emb:
                        s = _cosine(vec, emb)
                        if near is None or s > near[0]:
                            near = (s, e)
            if near is not None and near[0] >= dedup:
                e = near[1]
                e.support_count = (e.support_count or 1) + 1
                e.updated_at = datetime.now(timezone.utc)
                db.commit()
                return e.id
            e = ProjectBrainEntry(owner_id=owner_id, project_id=project_id,
                                  kind=kind, content=content, embedding=vec,
                                  embedding_dim=dim, support_count=1)
            db.add(e)
            db.commit()
            db.refresh(e)
            return e.id
        finally:
            db.close()

    try:
        return await asyncio.to_thread(_w)
    except Exception as exc:
        logger.warning(f"[Brain] write failed: {exc}")
        return None


def get_brain(owner_id: Optional[int], project_id: int, limit: int = 100) -> list[dict]:
    if owner_id is None:
        return []
    db = SessionLocal()
    try:
        rows = (db.query(ProjectBrainEntry)
                .filter(ProjectBrainEntry.owner_id == owner_id,
                        ProjectBrainEntry.project_id == project_id)
                .order_by(ProjectBrainEntry.support_count.desc(),
                          ProjectBrainEntry.updated_at.desc())
                .limit(limit).all())
        return [{"id": e.id, "kind": e.kind, "content": e.content,
                 "support_count": e.support_count,
                 "updated_at": e.updated_at.isoformat() if e.updated_at else None}
                for e in rows]
    finally:
        db.close()


def render(owner_id: Optional[int], project_id: int, limit: int = 20) -> str:
    """A '=== PROJECT BRAIN ===' block for the system prompt, grouped by kind."""
    entries = get_brain(owner_id, project_id, limit=limit)
    if not entries:
        return ""
    by_kind: dict[str, list[str]] = {}
    for e in entries:
        by_kind.setdefault(e["kind"] or "fact", []).append(e["content"])
    order = ["goal", "decision", "convention", "fact"]
    lines = []
    for k in order:
        if by_kind.get(k):
            lines.append(f"{k.capitalize()}s:")
            lines += [f"- {c}" for c in by_kind[k]]
    return "\n".join(lines)


def _project_of(conversation_id: Optional[int]) -> Optional[int]:
    if conversation_id is None:
        return None
    db = SessionLocal()
    try:
        from ..models.db_models import Conversation
        conv = db.get(Conversation, conversation_id)
        return getattr(conv, "project_id", None) if conv else None
    finally:
        db.close()


def _parse(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in (data if isinstance(data, list) else [])[:5]:
        if isinstance(it, dict):
            kind = str(it.get("kind", "")).lower().strip()
            content = str(it.get("content", "")).strip()
            if kind in _KINDS and content:
                out.append({"kind": kind, "content": content})
    return out


async def reflect(owner_id: Optional[int], project_id: Optional[int],
                  conversation_id: Optional[int]) -> int:
    """Distil project-level knowledge from a project conversation into the brain."""
    if owner_id is None or project_id is None or conversation_id is None:
        return 0

    def _load():
        db = SessionLocal()
        try:
            from ..models.db_models import Message
            msgs = (db.query(Message).filter(Message.conversation_id == conversation_id)
                    .order_by(Message.id.desc()).limit(20).all())[::-1]
            return [(m.role, m.content or "") for m in msgs if m.role in ("user", "assistant")]
        finally:
            db.close()

    try:
        transcript = await asyncio.to_thread(_load)
    except Exception:
        return 0
    if not transcript:
        return 0
    convo = "\n".join(f"{r}: {t[:500]}" for r, t in transcript[-16:])
    from ..services.fallback_router import route_chat
    from ..models.schemas import MessageDto
    db = SessionLocal()
    try:
        res = await route_chat(db, [
            MessageDto(role="system", content=_REFLECT_SYSTEM),
            MessageDto(role="user", content=f"CONVERSATION:\n{convo}\n\nJSON:"),
        ], temperature=0.0, max_tokens=400)
        content = res.content or ""
    except Exception as exc:
        logger.warning(f"[Brain] distillation failed: {exc}")
        return 0
    finally:
        db.close()
    written = 0
    for it in _parse(content):
        if await add_entry(owner_id, project_id, it["kind"], it["content"]):
            written += 1
    if written:
        logger.info(f"[Brain] project {project_id}: distilled {written} entry(ies)")
    return written


def maybe_reflect(owner_id: Optional[int], conversation_id: Optional[int]) -> None:
    """Debounced, fire-and-forget project reflection (only for project chats)."""
    if owner_id is None or conversation_id is None:
        return
    from ..config import get_settings
    s = get_settings()
    if not s.memory_project_brain_enabled:
        return
    project_id = _project_of(conversation_id)
    if project_id is None:
        return
    n = _turn_counts.get(conversation_id, 0) + 1
    if n < max(1, s.memory_project_reflect_every_turns):
        _turn_counts[conversation_id] = n
        return
    _turn_counts[conversation_id] = 0
    if len(_turn_counts) > _MAX_TRACK:
        _turn_counts.clear()
    try:
        asyncio.get_event_loop().create_task(reflect(owner_id, project_id, conversation_id))
    except RuntimeError:
        pass
