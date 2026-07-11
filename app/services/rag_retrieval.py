"""Hybrid retrieval + grounded-prompt construction for the RAG pipeline.

Recommended sequence (from the spec):

    semantic top-N   +   keyword top-N
                 ↓
        reciprocal rank fusion
                 ↓
              top-K
                 ↓
       optional cross-encoder rerank
                 ↓
            final top 5–8

Vector and keyword scores are NOT directly comparable, so the two ranked lists
are merged with Reciprocal Rank Fusion (position-based) rather than by mixing
raw scores.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Optional

from sqlalchemy import func, text as sql_text
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models.schemas import MessageDto
from ..models.rag_models import DocumentChunk, Document, KnowledgeBase, HAS_PGVECTOR
from ..providers.embeddings import embedding_provider_for_kb, INPUT_QUERY
from ..providers.reranker import resolve_reranker

logger = logging.getLogger(__name__)


GROUNDED_SYSTEM = """You are a knowledge-base assistant. Answer the user's \
question using ONLY the numbered context sources below. Cite every claim inline \
with bracketed source numbers like [1] or [2][3]. If the answer is not contained \
in the sources, say you don't have enough information in the knowledge base — do \
not use outside knowledge or invent facts. Be concise and accurate.

Context sources:
{context}"""


# ─── Individual retrievers ──────────────────────────────────────────────────

def semantic_search(db: Session, kb_id: int, query_vec: list[float], limit: int) -> list[DocumentChunk]:
    """Vector nearest-neighbours within a KB, best-first. Uses pgvector's
    cosine distance operator when available, else a Python fallback."""
    if not query_vec:
        return []
    if HAS_PGVECTOR:
        distance = DocumentChunk.embedding.cosine_distance(query_vec)
        rows = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.knowledge_base_id == kb_id)
            .filter(DocumentChunk.embedding.isnot(None))
            .order_by(distance.asc())
            .limit(limit)
            .all()
        )
        return rows
    return _semantic_python(db, kb_id, query_vec, limit)


def _semantic_python(db: Session, kb_id: int, query_vec: list[float], limit: int) -> list[DocumentChunk]:
    """Brute-force cosine in Python for the JSON-column (no-pgvector) mode."""
    rows = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.knowledge_base_id == kb_id)
        .limit(5000)
        .all()
    )
    qn = math.sqrt(sum(v * v for v in query_vec)) or 1.0
    scored = []
    for r in rows:
        emb = list(r.embedding) if r.embedding is not None else None
        if not emb or len(emb) != len(query_vec):
            continue
        dot = sum(a * b for a, b in zip(query_vec, emb))
        rn = math.sqrt(sum(v * v for v in emb)) or 1.0
        scored.append((dot / (qn * rn), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def keyword_search(db: Session, kb_id: int, query: str, limit: int) -> list[DocumentChunk]:
    """Postgres full-text keyword match within a KB, ranked by ts_rank. Backed
    by the functional GIN index on to_tsvector('english', text)."""
    query = (query or "").strip()
    if not query:
        return []
    tsv = func.to_tsvector("english", DocumentChunk.text)
    tsq = func.plainto_tsquery("english", query)
    try:
        rows = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.knowledge_base_id == kb_id)
            .filter(tsv.op("@@")(tsq))
            .order_by(func.ts_rank(tsv, tsq).desc())
            .limit(limit)
            .all()
        )
        return rows
    except Exception as exc:  # pragma: no cover - FTS should always be present
        logger.warning(f"[RAG] Keyword search failed ({exc}); using ILIKE fallback")
        db.rollback()
        like = f"%{query}%"
        return (
            db.query(DocumentChunk)
            .filter(DocumentChunk.knowledge_base_id == kb_id)
            .filter(DocumentChunk.text.ilike(like))
            .limit(limit)
            .all()
        )


# ─── Fusion ─────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[DocumentChunk]], k: int, top_k: int,
) -> list[tuple[DocumentChunk, float]]:
    """Merge position-ranked lists: score(d) = Σ 1/(k + rank_d). Returns the
    top_k chunks best-first with their fused score."""
    scores: dict[int, float] = {}
    by_id: dict[int, DocumentChunk] = {}
    for lst in ranked_lists:
        for rank, chunk in enumerate(lst):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[chunk.id] = chunk
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(by_id[cid], score) for cid, score in ordered[:top_k]]


# ─── Orchestration ──────────────────────────────────────────────────────────

async def retrieve(db: Session, kb: KnowledgeBase, query: str, owner_id: Optional[int]) -> list[dict]:
    """Run the full hybrid pipeline and return the final ranked chunks as dicts
    carrying their source document metadata (for citations)."""
    settings = get_settings()
    provider = embedding_provider_for_kb(
        db, owner_id, kb.embedding_platform, kb.embedding_model, kb.embedding_dim,
    )
    query_vec = await provider.embed_one(query, input_type=INPUT_QUERY)

    # Run both searches on ONE worker thread (sequentially) so the sync DB work
    # stays off the event loop without sharing the Session across threads (a
    # SQLAlchemy Session is not safe for concurrent use).
    def _search_both():
        return (
            semantic_search(db, kb.id, query_vec, settings.rag_semantic_top_n),
            keyword_search(db, kb.id, query, settings.rag_keyword_top_n),
        )

    sem, kw = await asyncio.to_thread(_search_both)

    fused = reciprocal_rank_fusion([sem, kw], settings.rag_rrf_k, settings.rag_fusion_top_k)
    if not fused:
        return []

    # Optional rerank (no-op by default) → final top 5–8.
    reranker = resolve_reranker(db, owner_id)
    docs_text = [c.text for c, _ in fused]
    order = await reranker.rerank(query, docs_text, settings.rag_final_top_k)
    final = [(fused[i][0], fused[i][1]) for i in order if 0 <= i < len(fused)]

    # Resolve source filenames in one query.
    doc_ids = {c.document_id for c, _ in final}
    names = dict(
        db.query(Document.id, Document.filename).filter(Document.id.in_(doc_ids)).all()
    ) if doc_ids else {}

    results = []
    for idx, (chunk, score) in enumerate(final, 1):
        results.append({
            "index": idx,
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_name": names.get(chunk.document_id, "document"),
            "ordinal": chunk.ordinal,
            "text": chunk.text,
            "score": round(float(score), 6),
        })
    return results


def build_grounded_messages(
    query: str, chunks: list[dict], history: Optional[list[MessageDto]] = None,
) -> list[MessageDto]:
    """Construct the LLM message list: a grounded system prompt with numbered
    sources, optional prior turns, then the user's question."""
    context = "\n\n".join(
        f"[{c['index']}] (source: {c['document_name']})\n{c['text']}" for c in chunks
    ) or "(no relevant sources found)"
    messages = [MessageDto(role="system", content=GROUNDED_SYSTEM.format(context=context))]
    if history:
        # Keep the last few turns for follow-up context; drop any system prompts.
        for m in history[-6:]:
            if m.role in ("user", "assistant"):
                messages.append(MessageDto(role=m.role, content=m.content))
    messages.append(MessageDto(role="user", content=query))
    return messages
