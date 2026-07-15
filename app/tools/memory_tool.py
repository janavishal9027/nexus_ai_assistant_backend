"""Memory Service Tool — memory_store / memory_search / memory_delete /
memory_store_batch (req 8, 13).

Part D: these now delegate to ``app/memory/episodic.py`` — REAL semantic
embeddings (the RAG provider), pgvector search, and **owner scoping** (the
authenticated account, read from request_context). The old fake SHA-256 encoder
and Python-only cosine are gone.
"""
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from ..services.tool_registry import tool_registry
from ..services import request_context
from ..services.audit import write_audit_log
from ..memory import episodic

logger = logging.getLogger(__name__)

TOP_K_DEFAULT = 5
TOP_K_MAX = 20        # req 8.4
BATCH_MAX = 50        # req 8.11


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
    from ..services.observability import observability
    owner = request_context.get_owner_id()
    ids = await episodic.store(owner, conversation_id, [text], user_id=user_id)
    if ids:
        observability.inc_memory_stored()
    return {"memory_id": (ids[0] if ids else None), "source": "live", "fetched_at": _now_iso()}


@tool_registry.tool(
    name="memory_search",
    description="Semantic search over the user's stored memory by cosine similarity.",
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
async def memory_search(query: str, conversation_id: Optional[int] = None,
                        top_k: int = TOP_K_DEFAULT) -> dict:
    from ..services.observability import observability
    top_k = min(max(int(top_k), 1), TOP_K_MAX)
    owner = request_context.get_owner_id()
    chunks = await episodic.search(owner, query, conversation_id=conversation_id,
                                   top_k=top_k, scope="user")
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

    owner = request_context.get_owner_id()
    acting = request_context.get_acting_user_id() or user_id

    def _delete():
        db = SessionLocal()
        try:
            row = db.query(MemoryChunk).filter(MemoryChunk.id == memory_id).first()
            if row is None:
                raise ValueError(f"Memory {memory_id} not found")
            if owner is not None and row.owner_id is not None and row.owner_id != owner:
                raise ValueError("Access denied")
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
    description="Store up to 50 memory chunks in a single call.",
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
    from ..services.observability import observability
    owner = request_context.get_owner_id()
    groups: dict = defaultdict(list)
    for it in list(items or [])[:BATCH_MAX]:
        t = str(it.get("text", "")).strip()
        if t:
            groups[it.get("conversation_id")].append(t)
    all_ids: list[int] = []
    for cid, texts in groups.items():
        all_ids.extend(await episodic.store(owner, cid, texts, user_id=user_id))
    if all_ids:
        observability.inc_memory_stored(len(all_ids))
    return {"stored": len(all_ids), "memory_ids": all_ids, "source": "live", "fetched_at": _now_iso()}
