"""Content knowledge graph (Part D Phase 4).

Extracts subject–relation–object triples from conversation content (LLM,
debounced + background), stores them as edges scoped to a project / conversation,
and exposes: a graph view, entity-neighbour lookup, and a related-facts recall
block for prompt injection. Owner-scoped; deduped on (source, relation, target).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from sqlalchemy import or_, func

from ..database import SessionLocal
from ..models.db_models import KgEdge

logger = logging.getLogger(__name__)

_turn_counts: dict[int, int] = {}
_MAX_TRACK = 2000

_EXTRACT_SYSTEM = (
    "Extract a knowledge graph from the text: factual subject–relation–object "
    "triples about durable entities (people, tools, technologies, concepts, "
    "files, orgs, endpoints). Keep entity names short and canonical; skip "
    "pleasantries and transient task chatter. "
    'Return ONLY a JSON array (max 12). Each: {"source":"...","relation":"...",'
    '"target":"...","source_type":"...","target_type":"..."}. Return [] if none.'
)


def _parse(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in (data if isinstance(data, list) else [])[:12]:
        if not isinstance(it, dict):
            continue
        s = str(it.get("source", "")).strip()[:200]
        r = str(it.get("relation", "")).strip()[:120]
        t = str(it.get("target", "")).strip()[:200]
        if s and r and t:
            out.append({
                "source": s, "relation": r, "target": t,
                "source_type": str(it.get("source_type", "") or "")[:40] or None,
                "target_type": str(it.get("target_type", "") or "")[:40] or None,
            })
    return out


def _store(owner_id, project_id, conversation_id, triples: list[dict]) -> int:
    if not triples:
        return 0
    db = SessionLocal()
    try:
        added = 0
        for tr in triples:
            exists = (db.query(KgEdge.id).filter(
                KgEdge.owner_id == owner_id,
                (KgEdge.project_id == project_id) if project_id is not None
                else (KgEdge.conversation_id == conversation_id),
                func.lower(KgEdge.source) == tr["source"].lower(),
                func.lower(KgEdge.relation) == tr["relation"].lower(),
                func.lower(KgEdge.target) == tr["target"].lower()).first())
            if exists:
                continue
            db.add(KgEdge(owner_id=owner_id, project_id=project_id,
                          conversation_id=conversation_id, **tr))
            added += 1
        db.commit()
        return added
    finally:
        db.close()


async def extract(owner_id: Optional[int], project_id: Optional[int],
                  conversation_id: Optional[int], text: str) -> int:
    """LLM-extract triples from text and store the new ones. Returns count added."""
    if owner_id is None or not str(text or "").strip():
        return 0
    from ..services.fallback_router import route_chat
    from ..models.schemas import MessageDto
    db = SessionLocal()
    try:
        res = await route_chat(db, [
            MessageDto(role="system", content=_EXTRACT_SYSTEM),
            MessageDto(role="user", content=f"TEXT:\n{text[:4000]}\n\nJSON:"),
        ], temperature=0.0, max_tokens=600)
        content = res.content or ""
    except Exception as exc:
        logger.warning(f"[KG] extraction failed: {exc}")
        return 0
    finally:
        db.close()
    triples = _parse(content)
    added = await asyncio.to_thread(_store, owner_id, project_id, conversation_id, triples)
    if added:
        logger.info(f"[KG] added {added} edge(s) (project={project_id} conv={conversation_id})")
    return added


def _scoped(db, owner_id, project_id, conversation_id):
    q = db.query(KgEdge).filter(KgEdge.owner_id == owner_id)
    if project_id is not None:
        return q.filter(KgEdge.project_id == project_id)
    if conversation_id is not None:
        return q.filter(KgEdge.conversation_id == conversation_id)
    return q


def graph(owner_id: Optional[int], project_id: Optional[int] = None,
          conversation_id: Optional[int] = None, limit: int = 300) -> dict:
    """Nodes + edges for a graph view."""
    if owner_id is None:
        return {"nodes": [], "edges": []}
    db = SessionLocal()
    try:
        rows = _scoped(db, owner_id, project_id, conversation_id).limit(limit).all()
        nodes: dict[str, dict] = {}
        edges = []
        for e in rows:
            nodes.setdefault(e.source, {"id": e.source, "type": e.source_type})
            nodes.setdefault(e.target, {"id": e.target, "type": e.target_type})
            edges.append({"source": e.source, "relation": e.relation, "target": e.target})
        return {"nodes": list(nodes.values()), "edges": edges}
    finally:
        db.close()


def query(owner_id: Optional[int], text: str, project_id: Optional[int] = None,
          conversation_id: Optional[int] = None, limit: int = 12) -> list[dict]:
    """Edges whose source or target matches a term from the query (keyword)."""
    if owner_id is None or not str(text or "").strip():
        return []
    terms = [t for t in re.findall(r"\w{3,}", text.lower())][:8]
    if not terms:
        return []
    db = SessionLocal()
    try:
        q = _scoped(db, owner_id, project_id, conversation_id)
        conds = []
        for t in terms:
            like = f"%{t}%"
            conds.append(func.lower(KgEdge.source).like(like))
            conds.append(func.lower(KgEdge.target).like(like))
        rows = q.filter(or_(*conds)).limit(limit).all()
        return [{"source": e.source, "relation": e.relation, "target": e.target}
                for e in rows]
    finally:
        db.close()


def render(owner_id: Optional[int], query_text: str, project_id: Optional[int] = None,
           conversation_id: Optional[int] = None, limit: int = 10) -> str:
    """A '=== RELATED FACTS ===' block for the system prompt."""
    edges = query(owner_id, query_text, project_id=project_id,
                  conversation_id=conversation_id, limit=limit)
    if not edges:
        return ""
    return "\n".join(f"- {e['source']} {e['relation']} {e['target']}" for e in edges)


async def maybe_extract(owner_id: Optional[int], conversation_id: Optional[int]) -> None:
    """Debounced, fire-and-forget triple extraction from the recent turns."""
    if owner_id is None or conversation_id is None:
        return
    from ..config import get_settings
    s = get_settings()
    if not s.memory_kg_enabled:
        return
    n = _turn_counts.get(conversation_id, 0) + 1
    if n < max(1, s.memory_kg_every_turns):
        _turn_counts[conversation_id] = n
        return
    _turn_counts[conversation_id] = 0
    if len(_turn_counts) > _MAX_TRACK:
        _turn_counts.clear()

    async def _run():
        def _load():
            db = SessionLocal()
            try:
                from ..models.db_models import Message, Conversation
                conv = db.get(Conversation, conversation_id)
                pid = getattr(conv, "project_id", None) if conv else None
                msgs = (db.query(Message).filter(Message.conversation_id == conversation_id)
                        .order_by(Message.id.desc()).limit(8).all())[::-1]
                text = "\n".join(f"{m.role}: {m.content or ''}" for m in msgs)
                return pid, text
            finally:
                db.close()
        try:
            pid, text = await asyncio.to_thread(_load)
            await extract(owner_id, pid, conversation_id, text)
        except Exception as exc:
            logger.warning(f"[KG] maybe_extract failed: {exc}")

    try:
        asyncio.get_event_loop().create_task(_run())
    except RuntimeError:
        pass
