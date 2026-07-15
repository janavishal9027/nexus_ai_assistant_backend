"""Document decision / triage (chat-module A.4).

Backend-authoritative: given the assistant's answer, decide whether the user is
likely to want it as a downloadable document and which format(s) apply — the
client never guesses. Content-based (a table ⇒ Excel/CSV, several code blocks ⇒
zip, a long structured report ⇒ Word/PDF), so it's instant and deterministic.

Returns {"document": bool, "format": <primary|None>, "formats": [<applicable>]}.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..models.db_models import Conversation, Message
from .document_export import _parse_blocks

logger = logging.getLogger(__name__)

_PROSE = ["markdown", "word", "pdf", "text"]


def classify_content(text: str) -> dict:
    text = text or ""
    blocks = _parse_blocks(text)
    has_table = any(b["type"] == "table" for b in blocks)
    code_blocks = [b for b in blocks if b["type"] == "code" and b["text"].strip()]
    headings = [b for b in blocks if b["type"] == "heading"]
    length = len(text)

    document = (
        length > 500 or has_table or len(code_blocks) >= 1 or len(headings) >= 2
    )
    if not document:
        return {"document": False, "format": None, "formats": []}

    formats: list[str] = []
    if has_table:
        formats += ["excel", "csv"]
    if code_blocks:
        formats += ["zip"]
    formats += _PROSE
    if len(headings) >= 2 or length > 1500:
        formats += ["powerpoint"]

    # Primary suggestion by dominant content shape.
    if has_table and length < 1500:
        primary = "excel"
    elif len(code_blocks) >= 2:
        primary = "zip"
    elif len(headings) >= 3:
        primary = "powerpoint"
    elif length > 1200:
        primary = "word"
    else:
        primary = "markdown"

    seen: set[str] = set()
    formats = [f for f in formats if not (f in seen or seen.add(f))]
    return {"document": True, "format": primary, "formats": formats}


def decide_document(db: Session, conversation_id: int, owner_id: Optional[int]) -> dict:
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None or (conv.owner_id is not None and conv.owner_id != owner_id):
        return {"document": False, "format": None, "formats": []}
    msg = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id, Message.role == "assistant")
        .order_by(Message.created_at.desc())
        .first()
    )
    if msg is None or not msg.content:
        return {"document": False, "format": None, "formats": []}
    return classify_content(msg.content)
