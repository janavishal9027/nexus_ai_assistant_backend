"""Query understanding + rewriting for RAG retrieval.

A follow-up like "how does it work?" embeds poorly on its own. Using the recent
conversation we rewrite it into a standalone query ("how does JWT refresh token
rotation work?") that produces a far better query embedding and keyword match.
The user's *intent* is preserved — we only resolve references ("it", "that") and
restore implied context; we never answer the question or invent specifics.

Gated by ``settings.rag_query_rewrite`` and only fired for short/referential
follow-ups, so most queries skip the extra LLM call.
See docs/semantic-embedding/02-retrieval-pipeline.md.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..config import get_settings
from ..models.schemas import MessageDto

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "You rewrite the user's latest message into a single standalone search query "
    "for a document-retrieval system. Resolve pronouns and references (it, that, "
    "this, those, they) using the conversation, and restore implied context, but "
    "do NOT change the user's intent, invent specifics, or answer the question. "
    "Preserve technical tokens exactly (e.g. C++, .NET, /api/auth/refresh, "
    "gemini-embedding-001). Output ONLY the rewritten query — no quotes, no preamble."
)

_REFERENTIAL = (
    " it", "it ", "its ", "that", "this", "those", "these", "them", "they",
    "he ", "she ", "the same", "above", "previous", "earlier",
)


def _needs_rewrite(query: str) -> bool:
    q = query.strip().lower()
    if len(q.split()) < 6:
        return True
    return any(tok in q for tok in _REFERENTIAL)


async def rewrite_query(
    db: Session,
    query: str,
    history: Optional[list[MessageDto]],
    owner_id: Optional[int],
) -> str:
    """Return a standalone query (rewritten only when it helps). Never raises —
    falls back to the original query on any error."""
    q = (query or "").strip()
    if not q:
        return q
    settings = get_settings()
    if not getattr(settings, "rag_query_rewrite", True) or not history:
        return q
    recent = [m for m in history if m.role in ("user", "assistant")][-6:]
    if not recent or not _needs_rewrite(q):
        return q
    try:
        from .fallback_router import route_chat
        convo = "\n".join(f"{m.role}: {m.content}" for m in recent)
        messages = [
            MessageDto(role="system", content=_REWRITE_SYSTEM),
            MessageDto(role="user", content=(
                f"Conversation:\n{convo}\n\nLatest message: {q}\n\nStandalone query:"
            )),
        ]
        result = await route_chat(db, messages, temperature=0.0, max_tokens=80)
        first_line = (result.content or "").strip().strip('"').splitlines()
        rewritten = first_line[0].strip() if first_line else ""
        if rewritten and 0 < len(rewritten) <= 400:
            if rewritten.lower() != q.lower():
                logger.info(f"[RAG] Query rewrite: {q!r} -> {rewritten!r}")
            return rewritten
    except Exception as exc:
        logger.warning(f"[RAG] Query rewrite failed ({exc}); using original query")
    return q
