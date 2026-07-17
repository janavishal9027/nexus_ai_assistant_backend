"""Shared graph-edge recall: embedding-based, with a keyword fallback.

Both graphs — the personal one (MemoryEdge) and the content one (KgEdge) —
recall edges the same way, so the logic lives here once. Each edge is embedded
as "source relation target" at write time; recall embeds the query and ranks by
cosine similarity, so "who do I work with" can match `User works_at Acme` even
though they share no substring.

Falls back to keyword LIKE when there's no embedding provider, when the query
can't be embedded, when no stored edge has a matching dimension, or when the
semantic pass finds nothing. The fallback searches the RELATION column too, not
just source/target — "what do I use" should be able to match on `uses`.

Never raises: recall problems must not interrupt a turn.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Optional

from sqlalchemy import func, or_

from ..database import SessionLocal
from ..models.db_models import HAS_PGVECTOR
from ..providers.embeddings import (INPUT_PASSAGE, INPUT_QUERY,
                                    resolve_embedding_provider)

logger = logging.getLogger(__name__)

def _threshold() -> float:
    """Cosine floor for an edge to count as relevant. Config-driven because it
    is embedding-model dependent — see memory_graph_recall_threshold."""
    from ..config import get_settings
    try:
        return float(get_settings().memory_graph_recall_threshold)
    except Exception:
        return 0.65


def edge_text(source: str, relation: str, target: str) -> str:
    """The string an edge is embedded as."""
    return f"{source} {relation} {target}"


async def embed(owner_id: Optional[int], texts: list[str],
                input_type: str) -> list[list[float]]:
    """Embed with the owner's configured provider. [] on any failure."""
    if not texts:
        return []

    def _resolve():
        db = SessionLocal()
        try:
            return resolve_embedding_provider(db, owner_id)
        finally:
            db.close()
    try:
        provider = await asyncio.to_thread(_resolve)
        return await provider.embed(texts, input_type=input_type)
    except Exception as exc:
        logger.warning(f"[GraphRecall] embed failed: {exc}")
        return []


async def embed_edges(owner_id: Optional[int], triples: list[dict]) -> list[dict]:
    """Attach `embedding` + `embedding_dim` to each triple, in place-ish.

    Returns the triples either way — an edge without a vector is still worth
    storing; it just recalls via keyword until something re-embeds it.
    """
    if not triples:
        return triples
    texts = [edge_text(t["source"], t["relation"], t["target"]) for t in triples]
    vecs = await embed(owner_id, texts, INPUT_PASSAGE)
    if len(vecs) != len(triples):
        return triples
    for t, v in zip(triples, vecs):
        if v:
            t["embedding"] = v
            t["embedding_dim"] = len(v)
    return triples


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def keyword_filter(model, text: str):
    """A LIKE condition across source, relation AND target. None if no terms."""
    terms = re.findall(r"\w{3,}", (text or "").lower())[:10]
    if not terms:
        return None
    conds = []
    for t in terms:
        like = f"%{t}%"
        conds.append(func.lower(model.source).like(like))
        conds.append(func.lower(model.relation).like(like))
        conds.append(func.lower(model.target).like(like))
    return or_(*conds)


async def search(model, owner_id: Optional[int], base_filters: list, text: str,
                 limit: int = 10) -> list:
    """Edges matching `text`, best first. Semantic when possible, else keyword.

    `base_filters` are SQLAlchemy conditions already scoping the query (owner,
    project, …); `owner_id` additionally selects whose embedding provider to
    use. Returns ORM rows.
    """
    if not str(text or "").strip():
        return []
    qv = await embed(owner_id, [text], INPUT_QUERY)
    qvec = qv[0] if qv else None

    def _run():
        db = SessionLocal()
        try:
            rows = _semantic(db, model, base_filters, qvec, limit) if qvec else []
            if rows:
                return rows
            return _keyword(db, model, base_filters, text, limit)
        finally:
            db.close()

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        logger.warning(f"[GraphRecall] search failed: {exc}")
        return []


def _semantic(db, model, base_filters, qvec, limit):
    dim = len(qvec)
    threshold = _threshold()
    base = db.query(model).filter(*base_filters).filter(
        model.embedding_dim == dim, model.embedding.isnot(None))
    if HAS_PGVECTOR:
        dist = model.embedding.cosine_distance(qvec)
        rows = (base.add_columns(dist.label("d")).order_by(dist.asc())
                .limit(limit * 3).all())
        keep = [(1.0 - float(d), r) for r, d in rows if (1.0 - float(d)) >= threshold]
    else:
        cand = base.order_by(model.id.desc()).limit(2000).all()
        keep = []
        for r in cand:
            emb = list(r.embedding) if r.embedding is not None else None
            if not emb:
                continue
            sim = _cosine(qvec, emb)
            if sim >= threshold:
                keep.append((sim, r))
    # Similarity first, then reinforcement — a strongly-supported edge wins ties.
    keep.sort(key=lambda x: (x[0], x[1].support_count or 1), reverse=True)
    return [r for _, r in keep[:limit]]


def _keyword(db, model, base_filters, text, limit):
    cond = keyword_filter(model, text)
    if cond is None:
        return []
    return (db.query(model).filter(*base_filters).filter(cond)
            .order_by(model.support_count.desc(), model.updated_at.desc())
            .limit(limit).all())
