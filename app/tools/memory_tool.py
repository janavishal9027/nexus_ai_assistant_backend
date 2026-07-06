"""Memory Service Tool — memory_store / memory_search / memory_delete /
memory_store_batch (req 8, 13).

Embeddings use a deterministic local encoder by default so the store→search
round-trip (Property 5) holds without any external embedding API; swap `_embed`
for a real provider (e.g. OpenAI text-embedding-3-small) in production. Cosine
similarity is computed in Python so search works whether the embedding column is
backed by pgvector or the JSON fallback.
"""
import asyncio
import hashlib
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from ..services.tool_registry import tool_registry
from ..services import request_context
from ..services.audit import write_audit_log
from ..models.db_models import EMBEDDING_DIM

logger = logging.getLogger(__name__)

TOP_K_DEFAULT = 5
TOP_K_MAX = 20        # req 8.4
BATCH_MAX = 50        # req 8.11
_SEARCH_CANDIDATE_LIMIT = 2000
_EMBED_FAIL_MSG = "Embedding generation failed"


def _threshold() -> float:
    from ..config import get_settings
    return get_settings().memory_similarity_threshold


def _embed(text: str) -> list[float]:
    """Deterministic pseudo-embedding: identical text → identical unit vector."""
    if text is None or not str(text).strip():
        raise ValueError(_EMBED_FAIL_MSG)
    vec = [0.0] * EMBEDDING_DIM
    tokens = str(text).lower().split() or [str(text).lower()]
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        for i, b in enumerate(digest):
            vec[(i * 7 + (hash(tok) & 0xFF)) % EMBEDDING_DIM] += (b - 127.5) / 127.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@tool_registry.tool(
    name="memory_store",
    description="Store a conversation memory chunk with its embedding.",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "conversation_id": {"type": "integer"},
            "user_id": {"type": "integer"},
        },
        "required": ["text", "conversation_id"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=15.0,
)
async def memory_store(text: str, conversation_id: int, user_id: Optional[int] = None) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import MemoryChunk
    from ..services.observability import observability

    try:
        embedding = _embed(text)
    except Exception:
        raise ValueError(_EMBED_FAIL_MSG)

    def _store():
        db = SessionLocal()
        try:
            chunk = MemoryChunk(
                text=text, embedding=embedding,
                conversation_id=conversation_id, user_id=user_id,
            )
            db.add(chunk)
            db.commit()
            db.refresh(chunk)
            return chunk.id
        finally:
            db.close()

    memory_id = await asyncio.to_thread(_store)
    observability.inc_memory_stored()
    return {"memory_id": memory_id, "source": "live", "fetched_at": _now_iso()}


@tool_registry.tool(
    name="memory_search",
    description="Semantic search over stored memory chunks by cosine similarity.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "conversation_id": {"type": "integer"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=15.0,
)
async def memory_search(query: str, conversation_id: Optional[int] = None, top_k: int = TOP_K_DEFAULT) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import MemoryChunk
    from ..services.observability import observability

    top_k = min(max(int(top_k), 1), TOP_K_MAX)
    try:
        q_emb = _embed(query)
    except Exception:
        raise ValueError(_EMBED_FAIL_MSG)
    threshold = _threshold()

    def _search():
        db = SessionLocal()
        try:
            q = db.query(MemoryChunk)
            if conversation_id is not None:
                q = q.filter(MemoryChunk.conversation_id == conversation_id)
            rows = q.order_by(MemoryChunk.id.desc()).limit(_SEARCH_CANDIDATE_LIMIT).all()
            scored = []
            for r in rows:
                emb = list(r.embedding) if r.embedding is not None else None
                if not emb:
                    continue
                sim = _cosine(q_emb, emb)
                if sim >= threshold:
                    scored.append((sim, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "memory_id": r.id,
                    "text": r.text,
                    "conversation_id": r.conversation_id,
                    "similarity": round(sim, 6),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for sim, r in scored[:top_k]
            ]
        finally:
            db.close()

    chunks = await asyncio.to_thread(_search)
    observability.inc_memory_search()
    return {"chunks": chunks, "count": len(chunks), "source": "live", "fetched_at": _now_iso()}


@tool_registry.tool(
    name="memory_delete",
    description="Delete a memory chunk by memory_id (owner-scoped).",
    input_schema={
        "type": "object",
        "properties": {"memory_id": {"type": "integer"}, "user_id": {"type": "integer"}},
        "required": ["memory_id"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def memory_delete(memory_id: int, user_id: Optional[int] = None) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import MemoryChunk

    acting = request_context.get_acting_user_id()
    if acting is None:
        acting = user_id

    def _delete():
        db = SessionLocal()
        try:
            row = db.query(MemoryChunk).filter(MemoryChunk.id == memory_id).first()
            if row is None:
                raise ValueError(f"Memory {memory_id} not found")
            if row.user_id is not None and acting is not None and row.user_id != acting:
                raise ValueError("Access denied")
            db.delete(row)
            db.commit()
            write_audit_log("memory_delete", memory_id, "success", acting_user_id=acting)
            return True
        finally:
            db.close()

    await asyncio.to_thread(_delete)
    return {"deleted": True, "memory_id": memory_id}


@tool_registry.tool(
    name="memory_store_batch",
    description="Store up to 50 memory chunks in a single transaction.",
    input_schema={
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "object"}},
            "user_id": {"type": "integer"},
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=20.0,
)
async def memory_store_batch(items: list[dict], user_id: Optional[int] = None) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import MemoryChunk
    from ..services.observability import observability

    items = list(items or [])[:BATCH_MAX]
    try:
        prepared = [
            (it.get("text", ""), it.get("conversation_id"), _embed(it.get("text", "")))
            for it in items if str(it.get("text", "")).strip()
        ]
    except Exception:
        raise ValueError(_EMBED_FAIL_MSG)

    def _store():
        db = SessionLocal()
        try:
            chunks = [
                MemoryChunk(text=t, embedding=emb, conversation_id=cid, user_id=user_id)
                for (t, cid, emb) in prepared
            ]
            db.add_all(chunks)
            db.commit()
            for c in chunks:
                db.refresh(c)
            return [c.id for c in chunks]
        finally:
            db.close()

    ids = await asyncio.to_thread(_store)
    observability.inc_memory_stored(len(ids))
    return {"stored": len(ids), "memory_ids": ids, "source": "live", "fetched_at": _now_iso()}
