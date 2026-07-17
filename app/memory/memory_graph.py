"""Personal memory graph (Part D Phase 5).

A per-USER entity graph distilled from episodic memory across ALL conversations:
the people/orgs the user knows and the tools/tech they work with, as subject–
relation–object triples. Unlike the content knowledge graph (rag/knowledge_graph,
scoped to a project/conversation), this is owner-scoped and cross-conversation —
"what I know about *you*", relationally.

- **Reinforce + decay**: a repeated fact bumps its support_count (recalled first);
  facts not seen for ``memory_graph_decay_days`` slowly fade and are dropped.
- **Query-relevant recall**: ``render()`` injects only the graph facts whose
  entities match the current message (low token cost).
- Extraction is LLM-based, debounced, and runs in the background — never blocks a
  turn. Owner-scoped; deduped on lowercased (source, relation, target).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import or_, func

from ..database import SessionLocal
from ..models.db_models import MemoryEdge

logger = logging.getLogger(__name__)

_turn_counts: dict[int, int] = {}
_MAX_TRACK = 2000

_EXTRACT_SYSTEM = (
    "From the conversation, extract a small PERSONAL knowledge graph about the "
    "USER and their world — ONLY: (a) people & organizations they mention "
    "(colleagues, contacts, companies, teams, employers) and (b) tools & "
    "technologies they use or work with (languages, frameworks, libraries, "
    "services, platforms). Express each as a subject-relation-object triple; the "
    "subject is usually 'User' (the person you're talking to) or a named person/"
    "org. Use short, canonical names. Skip transient task chatter, opinions, code "
    "detail, and anything that isn't a durable fact about the user's people or "
    "tech. source_type/target_type MUST each be one of: person, org, tool, tech, "
    "user. Return ONLY a JSON array (max 10). Each: {\"source\":\"...\","
    "\"relation\":\"...\",\"target\":\"...\",\"source_type\":\"...\","
    "\"target_type\":\"...\"}. Return [] if nothing durable."
)

# Type normalization → the four we keep (+ the user). Anything that normalizes
# outside this set (topic, preference, goal, concept, file…) drops the triple, so
# the graph stays focused on people/orgs + tools/tech.
_KEEP = {"person", "org", "tool", "tech", "user"}
_TYPE_ALIASES = {
    "person": "person", "people": "person", "individual": "person",
    "contact": "person", "colleague": "person", "human": "person",
    "org": "org", "organization": "org", "organisation": "org",
    "company": "org", "team": "org", "employer": "org", "group": "org",
    "tool": "tool", "framework": "tool", "library": "tool", "service": "tool",
    "platform": "tool", "software": "tool", "app": "tool", "ide": "tool",
    "database": "tool", "db": "tool", "sdk": "tool",
    "tech": "tech", "technology": "tech", "language": "tech",
    "protocol": "tech", "standard": "tech", "concept_tech": "tech",
    "user": "user", "self": "user", "me": "user",
}


def _norm_type(t: Optional[str]) -> Optional[str]:
    key = (t or "").lower().strip()
    return _TYPE_ALIASES.get(key, key or None)


def _parse(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in (data if isinstance(data, list) else [])[:10]:
        if not isinstance(it, dict):
            continue
        s = str(it.get("source", "")).strip()[:200]
        r = str(it.get("relation", "")).strip()[:120]
        t = str(it.get("target", "")).strip()[:200]
        if not (s and r and t):
            continue
        st = _norm_type(it.get("source_type"))
        tt = _norm_type(it.get("target_type"))
        # Keep only people/orgs + tools/tech (both endpoints must be in-scope or
        # untyped). Drops topic/preference/goal/etc. that the model may emit.
        if (st is not None and st not in _KEEP) or (tt is not None and tt not in _KEEP):
            continue
        out.append({"source": s, "relation": r, "target": t,
                    "source_type": st, "target_type": tt})
    return out


def _store(owner_id: int, triples: list[dict]) -> int:
    """Insert new edges; reinforce (support_count↑) existing ones. Returns #added."""
    if not triples:
        return 0
    db = SessionLocal()
    try:
        added = 0
        for tr in triples:
            row = (db.query(MemoryEdge).filter(
                MemoryEdge.owner_id == owner_id,
                func.lower(MemoryEdge.source) == tr["source"].lower(),
                func.lower(MemoryEdge.relation) == tr["relation"].lower(),
                func.lower(MemoryEdge.target) == tr["target"].lower()).first())
            if row:
                row.support_count = (row.support_count or 1) + 1
                row.updated_at = datetime.now(timezone.utc)
                # Backfill the vector on edges stored before embeddings existed.
                if row.embedding is None and tr.get("embedding"):
                    row.embedding = tr["embedding"]
                    row.embedding_dim = tr["embedding_dim"]
            else:
                db.add(MemoryEdge(owner_id=owner_id, support_count=1, **tr))
                added += 1
        db.commit()
        return added
    finally:
        db.close()


async def extract(owner_id: Optional[int], text: str) -> int:
    """LLM-extract personal triples from text and store/reinforce them."""
    if owner_id is None or not str(text or "").strip():
        return 0
    from ..services.fallback_router import route_chat
    from ..models.schemas import MessageDto
    db = SessionLocal()
    try:
        res = await route_chat(db, [
            MessageDto(role="system", content=_EXTRACT_SYSTEM),
            MessageDto(role="user", content=f"CONVERSATION:\n{text[:4000]}\n\nJSON:"),
        ], temperature=0.0, max_tokens=500)
        content = res.content or ""
    except Exception as exc:
        logger.warning(f"[MemGraph] extraction failed: {exc}")
        return 0
    finally:
        db.close()
    triples = _parse(content)
    # Embed before the (sync) write so recall can match on meaning.
    from . import graph_recall
    triples = await graph_recall.embed_edges(owner_id, triples)
    added = await asyncio.to_thread(_store, owner_id, triples)
    if added:
        logger.info(f"[MemGraph] owner={owner_id}: added {added} personal edge(s)")
    return added


# ── Read / recall ───────────────────────────────────────────────────────────

def graph(owner_id: Optional[int], limit: int = 300) -> dict:
    """Nodes + edges of the user's personal graph (strongest first)."""
    if owner_id is None:
        return {"nodes": [], "edges": []}
    db = SessionLocal()
    try:
        rows = (db.query(MemoryEdge).filter(MemoryEdge.owner_id == owner_id)
                .order_by(MemoryEdge.support_count.desc(),
                          MemoryEdge.updated_at.desc()).limit(limit).all())
        nodes: dict[str, dict] = {}
        edges = []
        for e in rows:
            nodes.setdefault(e.source, {"id": e.source, "type": e.source_type})
            nodes.setdefault(e.target, {"id": e.target, "type": e.target_type})
            edges.append({"id": e.id, "source": e.source, "relation": e.relation,
                          "target": e.target, "support": e.support_count})
        return {"nodes": list(nodes.values()), "edges": edges}
    finally:
        db.close()


async def query(owner_id: Optional[int], text: str, limit: int = 10) -> list[dict]:
    """Edges relevant to the query — semantic when embeddings are available,
    keyword otherwise. See graph_recall."""
    if owner_id is None or not str(text or "").strip():
        return []
    from . import graph_recall
    rows = await graph_recall.search(
        MemoryEdge, owner_id, [MemoryEdge.owner_id == owner_id], text, limit=limit)
    return [{"id": e.id, "source": e.source, "relation": e.relation,
             "target": e.target, "support": e.support_count} for e in rows]


def neighbors(owner_id: Optional[int], entity: str, limit: int = 50) -> list[dict]:
    """All edges touching a specific entity (exact, case-insensitive)."""
    if owner_id is None or not str(entity or "").strip():
        return []
    e = entity.lower().strip()
    db = SessionLocal()
    try:
        rows = (db.query(MemoryEdge).filter(
                    MemoryEdge.owner_id == owner_id,
                    or_(func.lower(MemoryEdge.source) == e,
                        func.lower(MemoryEdge.target) == e))
                .order_by(MemoryEdge.support_count.desc()).limit(limit).all())
        return [{"id": r.id, "source": r.source, "relation": r.relation,
                 "target": r.target, "support": r.support_count} for r in rows]
    finally:
        db.close()


async def render(owner_id: Optional[int], query_text: str, limit: int = 8) -> str:
    """A recall block of the graph facts relevant to the current message."""
    edges = await query(owner_id, query_text, limit=limit)
    if not edges:
        return ""
    return "\n".join(f"- {e['source']} {e['relation']} {e['target']}" for e in edges)


# ── Lifecycle ────────────────────────────────────────────────────────────────

def summary(owner_id: int) -> int:
    db = SessionLocal()
    try:
        return db.query(MemoryEdge).filter(MemoryEdge.owner_id == owner_id).count()
    finally:
        db.close()


def purge(owner_id: int) -> int:
    db = SessionLocal()
    try:
        n = (db.query(MemoryEdge).filter(MemoryEdge.owner_id == owner_id)
             .delete(synchronize_session=False))
        db.commit()
        return n
    finally:
        db.close()


def delete_edge(owner_id: int, edge_id: int) -> bool:
    """Forget ONE edge — "that's wrong about me" without nuking the graph.

    Filtered on owner_id as well as id, so a guessed id can't delete another
    account's edge. False when nothing matched."""
    db = SessionLocal()
    try:
        n = (db.query(MemoryEdge).filter(MemoryEdge.owner_id == owner_id,
                                         MemoryEdge.id == edge_id)
             .delete(synchronize_session=False))
        db.commit()
        if n:
            logger.info(f"[MemGraph] owner={owner_id} deleted edge id={edge_id}")
        return bool(n)
    finally:
        db.close()


def decay(days: Optional[int] = None) -> int:
    """Slow decay (global sweep): edges not reinforced within ``days`` fade a
    support point; the weakest are dropped. 0/None disables. Returns #removed.

    We must keep ``updated_at`` OLD when fading a stronger edge (so it keeps
    decaying on later sweeps) — a bulk Query.update() would otherwise fire the
    column's onupdate and reset staleness, so we set updated_at to itself."""
    from ..config import get_settings
    days = days if days is not None else get_settings().memory_graph_decay_days
    if not days or days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    db = SessionLocal()
    try:
        # Stale + already weak → drop outright (avoids a decrement/delete race).
        removed = db.query(MemoryEdge).filter(
            MemoryEdge.updated_at < cutoff,
            MemoryEdge.support_count <= 1).delete(synchronize_session=False)
        # Stale + stronger → fade one point, preserving updated_at.
        db.query(MemoryEdge).filter(
            MemoryEdge.updated_at < cutoff,
            MemoryEdge.support_count > 1).update(
            {MemoryEdge.support_count: MemoryEdge.support_count - 1,
             MemoryEdge.updated_at: MemoryEdge.updated_at},
            synchronize_session=False)
        db.commit()
        if removed:
            logger.info(f"[MemGraph] decay dropped {removed} stale edge(s)")
        return removed
    finally:
        db.close()


# ── Trigger ──────────────────────────────────────────────────────────────────

async def maybe_extract(owner_id: Optional[int], conversation_id: Optional[int]) -> None:
    """Debounced, fire-and-forget personal-graph extraction from the recent turns."""
    if owner_id is None or conversation_id is None:
        return
    from ..config import get_settings
    s = get_settings()
    if not s.memory_graph_enabled:
        return
    n = _turn_counts.get(conversation_id, 0) + 1
    if n < max(1, s.memory_graph_every_turns):
        _turn_counts[conversation_id] = n
        return
    _turn_counts[conversation_id] = 0
    if len(_turn_counts) > _MAX_TRACK:
        _turn_counts.clear()

    async def _run():
        def _load():
            db = SessionLocal()
            try:
                from ..models.db_models import Message
                msgs = (db.query(Message)
                        .filter(Message.conversation_id == conversation_id)
                        .order_by(Message.id.desc()).limit(8).all())[::-1]
                return "\n".join(f"{m.role}: {m.content or ''}" for m in msgs)
            finally:
                db.close()
        try:
            text = await asyncio.to_thread(_load)
            await extract(owner_id, text)
        except Exception as exc:
            logger.warning(f"[MemGraph] maybe_extract failed: {exc}")

    try:
        asyncio.get_event_loop().create_task(_run())
    except RuntimeError:
        pass
