"""Dynamic new-chat conversation starters.

Instead of hardcoded chips, an LLM generates 3 fresh, specific starters each time
a new chat opens: (1) a current technology / coding task, (2) a globally-relevant
new idea to brainstorm, (3) a useful writing / productivity task. Fails open to a
small, varied default set so the empty state always has chips.
"""
from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from ..models.schemas import MessageDto

logger = logging.getLogger(__name__)

_CATS = ("code", "idea", "write")

_SYSTEM = (
    "Generate exactly 3 fresh, specific conversation-starter suggestions for a NEW "
    "chat with a general AI assistant. Return ONLY a JSON array of 3 objects, each: "
    '{"category":"code"|"idea"|"write","label":"<2-4 word chip label>","prompt":'
    '"<the full first message the user would send>"}.\n'
    "- Item 1 — category 'code': the best / most current technology or coding task "
    "(a trending language, framework, tool, or technique).\n"
    "- Item 2 — category 'idea': a fresh, creative idea to brainstorm or generate "
    "that is relevant to the world right now (globally trending themes).\n"
    "- Item 3 — category 'write': a genuinely useful writing or productivity task.\n"
    "Labels SHORT (2-4 words); prompts specific + actionable. Make them feel new and "
    "different each time — avoid clichs like 'write a python script that generates a "
    "random password'."
)

# Used only when the LLM is unavailable — a few sets so even the fallback varies.
_FALLBACK = [
    [
        {"category": "code", "label": "Build a REST API", "prompt": "Show me how to build a small REST API with one CRUD endpoint, with runnable code."},
        {"category": "idea", "label": "Startup ideas", "prompt": "Brainstorm 5 startup ideas that solve a real problem people have right now."},
        {"category": "write", "label": "Polish my email", "prompt": "Help me rewrite an email so it sounds clear, warm, and professional."},
    ],
    [
        {"category": "code", "label": "Explain async/await", "prompt": "Explain async/await with a simple, practical code example I can run."},
        {"category": "idea", "label": "Weekend project", "prompt": "Suggest 3 fun weekend coding projects I could actually finish, with a plan for each."},
        {"category": "write", "label": "Draft a bio", "prompt": "Help me write a short, punchy professional bio for my profile."},
    ],
    [
        {"category": "code", "label": "Optimize this code", "prompt": "Show me common ways to make a slow function faster, with before/after examples."},
        {"category": "idea", "label": "Content ideas", "prompt": "Give me 10 fresh content ideas trending globally that I could post about this week."},
        {"category": "write", "label": "Summarize text", "prompt": "Help me summarize a long piece of text into clear, concise bullet points."},
    ],
]


def _parse(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in (data if isinstance(data, list) else [])[:3]:
        if not isinstance(it, dict):
            continue
        cat = str(it.get("category", "idea")).lower().strip()
        if cat not in _CATS:
            cat = "idea"
        label = str(it.get("label", "")).strip()[:40]
        prompt = str(it.get("prompt", "")).strip()[:400]
        if label and prompt:
            out.append({"category": cat, "label": label, "prompt": prompt})
    return out


async def generate_starters(db: Session, owner_id: Optional[int] = None,
                            model: Optional[str] = None) -> list[dict]:
    """3 fresh starters via the LLM; a varied default set on any failure."""
    from .fallback_router import route_chat
    today = datetime.now().strftime("%B %Y")
    try:
        res = await route_chat(db, [
            MessageDto(role="system", content=f"{_SYSTEM}\n\nToday's date: {today}."),
            MessageDto(role="user", content="Return the JSON array now."),
        ], requested_model=model, temperature=0.9, max_tokens=320)
        starters = _parse(res.content or "")
    except Exception as exc:
        logger.warning(f"[Starters] generation failed: {exc}")
        starters = []
    if len(starters) < 3:
        starters = random.choice(_FALLBACK)
    return starters[:3]
