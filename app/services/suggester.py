"""The Suggester — a side agent that proposes what the user might ask next
(chat-module A.2 · rides on the `done` envelope as follow-up suggestions).

Runs as a post-turn call (like the Clarifier's pre-turn gate) so the streaming
paths stay untouched: once a turn is saved, the client asks for follow-ups and
renders them as tappable chips under the answer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from sqlalchemy.orm import Session

from ..models.schemas import MessageDto
from ..models.db_models import Conversation, Message
from .fallback_router import route_chat

logger = logging.getLogger(__name__)

_SYSTEM = """You propose what the user is most likely to ask NEXT in this conversation.
Given the recent messages, suggest 3-4 short, specific follow-up questions or requests.

Rules:
- Phrase each the way the USER would type it (a question or an imperative).
- Keep each under 60 characters.
- Make them specific to THIS conversation's topic — never generic filler.
- No numbering, no surrounding quotes.

Return STRICT JSON and nothing else:
{"suggestions": ["...", "...", "..."]}"""


def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except Exception:
        return None


async def suggest_followups(
    db: Session,
    conversation_id: int,
    owner_id: Optional[int] = None,
    model: Optional[str] = None,
    max_n: int = 4,
) -> list[str]:
    """Return up to ``max_n`` follow-up suggestions for a conversation, or []."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None or (conv.owner_id is not None and conv.owner_id != owner_id):
        return []

    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(6)
        .all()
    )
    rows = list(reversed(rows))
    if not rows:
        return []

    convo = "\n".join(f"{m.role}: {(m.content or '')[:600]}" for m in rows)
    prompt = [
        MessageDto(role="system", content=_SYSTEM),
        MessageDto(role="user", content=f"Conversation:\n{convo}\n\nSuggest the follow-ups."),
    ]

    try:
        result = await route_chat(db, prompt, requested_model=model,
                                  temperature=0.4, max_tokens=220)
        data = _extract_json(result.content)
    except Exception as e:
        logger.warning(f"[Suggester] failed: {e}")
        return []

    items = (data or {}).get("suggestions") or []
    out: list[str] = []
    for s in items:
        s = str(s).strip().strip('"').lstrip("-•*0123456789. ").strip()
        if s and len(s) <= 120 and s not in out:
            out.append(s)
    return out[:max_n]
