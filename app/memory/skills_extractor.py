"""The Reflector (Part D Phase 2) — turns completed episodes + feedback into
durable skills.

Runs debounced in the background (once every ``memory_reflect_every_turns`` stored
turns per conversation). It reads the conversation transcript + the user's 👍/👎
feedback, asks the LLM to distil STABLE facts about the *user* (preferences,
skills, lessons — "responds well to X" / "this approach isn't landing"), and
upserts each into semantic memory (dedup-reinforced). Never raises.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from ..database import SessionLocal
from . import semantic

logger = logging.getLogger(__name__)

_KINDS = {"preference", "skill", "lesson"}
_POLARITY = {"positive", "negative", "neutral"}

_REFLECT_SYSTEM = (
    "You are a reflection engine. From a user's conversation and their feedback, "
    "distil DURABLE facts about the USER to personalise future help — their "
    "preferences, skills, tools, and what works vs what to avoid. Ignore transient "
    "task details and the topic itself. Upvoted answers show what lands ('responds "
    "well to X'); downvoted answers show what to avoid. "
    'Return ONLY a JSON array (max 5). Each item: {"kind":"preference"|"skill"|'
    '"lesson","content":"<short statement about the user, e.g. \'Prefers concise, '
    'code-first answers\'>","polarity":"positive"|"negative"}. Return [] if nothing durable.'
)

# conversation_id -> stored turns since the last reflection (debounce state)
_turn_counts: dict[int, int] = {}
_MAX_TRACK = 2000


def _parse_skills(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in (data if isinstance(data, list) else [])[:5]:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind", "")).lower().strip()
        content = str(it.get("content", "")).strip()
        polarity = str(it.get("polarity", "neutral")).lower().strip()
        if kind in _KINDS and content:
            out.append({"kind": kind, "content": content,
                        "polarity": polarity if polarity in _POLARITY else "neutral"})
    return out


async def reflect(owner_id: Optional[int], conversation_id: Optional[int]) -> int:
    """Distil this conversation into skills. Returns how many skills were written."""
    if owner_id is None or conversation_id is None:
        return 0

    def _load():
        db = SessionLocal()
        try:
            from ..models.db_models import Message, MessageFeedback
            msgs = (db.query(Message)
                    .filter(Message.conversation_id == conversation_id)
                    .order_by(Message.id.desc()).limit(24).all())[::-1]
            fbs = (db.query(MessageFeedback)
                   .filter(MessageFeedback.conversation_id == conversation_id,
                           MessageFeedback.owner_id == owner_id).all())
            transcript = [(m.role, m.content or "") for m in msgs
                          if m.role in ("user", "assistant")]
            feedback = [(f.rating, (f.assistant_text or "")) for f in fbs]
            return transcript, feedback
        finally:
            db.close()

    try:
        transcript, feedback = await asyncio.to_thread(_load)
    except Exception as exc:
        logger.warning(f"[Reflector] load failed: {exc}")
        return 0
    if not transcript:
        return 0

    convo = "\n".join(f"{role}: {text[:500]}" for role, text in transcript[-16:])
    fb_lines = "\n".join(
        f"- {'UPVOTED (responds well to this)' if r > 0 else 'DOWNVOTED (did not land)'}: {t[:200]}"
        for r, t in feedback) or "(no explicit feedback)"

    from ..services.fallback_router import route_chat
    from ..models.schemas import MessageDto
    messages = [
        MessageDto(role="system", content=_REFLECT_SYSTEM),
        MessageDto(role="user",
                   content=f"CONVERSATION:\n{convo}\n\nFEEDBACK:\n{fb_lines}\n\nJSON:"),
    ]
    db = SessionLocal()
    try:
        result = await route_chat(db, messages, temperature=0.0, max_tokens=400)
        content = result.content or ""
    except Exception as exc:
        logger.warning(f"[Reflector] LLM distillation failed: {exc}")
        return 0
    finally:
        db.close()

    skills = _parse_skills(content)
    written = 0
    for sk in skills:
        if await semantic.upsert_skill(owner_id, sk["kind"], sk["content"], sk["polarity"]):
            written += 1
    if written:
        logger.info(f"[Reflector] conversation {conversation_id}: distilled {written} skill(s)")
    return written


def maybe_reflect(owner_id: Optional[int], conversation_id: Optional[int]) -> None:
    """Debounced, fire-and-forget: reflect once per N stored turns per conversation."""
    if owner_id is None or conversation_id is None:
        return
    from ..config import get_settings
    s = get_settings()
    if not s.memory_reflect_enabled:
        return
    n = _turn_counts.get(conversation_id, 0) + 1
    if n < max(1, s.memory_reflect_every_turns):
        _turn_counts[conversation_id] = n
        return
    _turn_counts[conversation_id] = 0
    if len(_turn_counts) > _MAX_TRACK:      # keep the debounce map bounded
        _turn_counts.clear()
    try:
        asyncio.get_event_loop().create_task(reflect(owner_id, conversation_id))
    except RuntimeError:
        pass
