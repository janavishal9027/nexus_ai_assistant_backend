"""The Clarifier — the sole decision-maker for whether a turn needs a clarifying
question before it is answered (chat-module spec A.2).

Runs as a fast pre-flight gate BEFORE the answer stream. Decision:
  • trivial (greeting/ack/too-short) → skip (no LLM call)
  • a REQUIRED slot is missing with no sensible default → BLOCK: return a
    structured AskUserQuestion so the UI asks first instead of answering in a
    default.
  • everything else → skip (answer directly).

This keeps the streaming paths untouched: the client calls /api/chat/clarify,
and only proceeds to stream once it has an answer (or there was nothing to ask).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from sqlalchemy.orm import Session

from ..models.schemas import MessageDto
from .fallback_router import route_chat

logger = logging.getLogger(__name__)

# Instant skip: greetings / acknowledgements never need clarification.
_TRIVIAL = {
    "hi", "hello", "hey", "yo", "sup", "hiya", "hola", "thanks", "thank you",
    "thx", "ty", "ok", "okay", "k", "cool", "great", "nice", "good", "yes",
    "no", "yep", "nope", "sure", "got it", "gotcha", "morning", "gm",
}

_SYSTEM = """You are the Clarifier. You run BEFORE an AI assistant answers, and you
decide whether the assistant should ask the user ONE clarifying question first.

Read the message and judge the user's INTENT. If the intent is underspecified — a
key detail is missing and the best answer would differ a LOT depending on it — ask
ONE targeted question with concrete options. This applies to ANY domain, not just
code: writing, design, planning, recommendations, analysis, math, cooking, travel…

Ask when the request is genuinely ambiguous, for example:
- "write a program to reverse a string" → which programming language?
- "translate this" → which target language?
- "design a logo" → what style / for what brand?
- "plan a trip" → destination, dates, budget?
- "recommend a laptop" → budget and main use?
- "write an essay" → topic, length, tone?
- "make it better" → better in what way?

Do NOT ask when:
- it's a greeting, thanks, or small talk.
- the request already has enough to give a useful, targeted answer.
- it's a clear factual question with a well-known answer ("capital of France",
  "who wrote Hamlet", "what is 2+2") — just answer it.
- one obvious default clearly satisfies it, or the user wants your judgement
  ("you decide", "anything", "surprise me").

Do NOT manufacture ambiguity. Assume the common, present-day interpretation. If a
reasonable person would confidently know what is being asked, answer it — only ask
when you are genuinely unsure what the user wants.

When you ask, give 2-4 concrete, specific options tailored to the request.

Respond with STRICT JSON and nothing else:
{"clarify": false}
OR
{"clarify": true, "question": {"header": "<label, <=14 chars>", "question": "<one specific question>", "multi_select": false, "options": [{"label": "<short choice>", "description": "<what it means>"}]}}"""


def _looks_trivial(text: str) -> bool:
    t = text.lower().strip(" \t\n!?.,…")
    return not t or len(t) < 3 or t in _TRIVIAL


# ── Deterministic fast-path: code request with no language named ────────────
# The canonical "required missing slot" case (spec A.2). The LLM clarifier is
# unreliable here, so decide it with rules: it must NOT depend on a model's mood.

# A named programming language / language-bound framework in the message.
_LANG_RE = re.compile(
    r"(?<![a-z0-9+#])("
    r"python|py|javascript|js|typescript|ts|java|c\+\+|cpp|c#|csharp|c|golang|go|"
    r"rust|ruby|php|swift|kotlin|dart|scala|matlab|sql|bash|shell|powershell|"
    r"html|css|perl|haskell|lua|assembly|fortran|cobol|elixir|erlang|clojure|"
    r"objective-?c|visual\s*basic|vb\.net|node(?:\.js)?|react|flutter|pandas|numpy|r"
    r")(?![a-z0-9+#])",
    re.I,
)

# A request to produce code (verb + code noun, or "<code noun> to/that/for …").
_CODE_RE = re.compile(
    r"\b(write|create|build|make|generate|implement|develop|code|give\s+me|show\s+me|"
    r"need|design)\b.{0,40}\b(program|function|script|code|algorithm|method|class|"
    r"snippet|app|api|regex|query|command)\b",
    re.I | re.S,
)
_CODE_RE2 = re.compile(
    r"\b(program|function|script|code|algorithm|method|regex|query)\b\s+"
    r"(to|that|which|for)\b",
    re.I,
)

_LANGUAGE_QUESTION = {
    "clarify": True,
    "question": {
        "header": "Language",
        "question": "Which programming language should I use?",
        "multi_select": False,
        "options": [
            {"label": "Python", "description": "Clear and concise; good default"},
            {"label": "JavaScript", "description": "Web / Node.js"},
            {"label": "Java", "description": "Typed; enterprise / Android"},
            {"label": "C++", "description": "Systems / performance"},
        ],
    },
}


def _needs_language(text: str) -> bool:
    """A code request that doesn't name a language → must ask which one."""
    if _LANG_RE.search(text):
        return False
    return bool(_CODE_RE.search(text) or _CODE_RE2.search(text))


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


async def assess_clarification(
    db: Session,
    message: str,
    history: Optional[list[MessageDto]] = None,
    owner_id: Optional[int] = None,
    model: Optional[str] = None,
) -> dict:
    """Return {"clarify": False} or {"clarify": True, "question": {...}}."""
    text = (message or "").strip()
    if _looks_trivial(text):
        return {"clarify": False}

    # Deterministic fast-path (spec's canonical case): a code request with no
    # language named ALWAYS blocks to ask which language — no LLM, no latency.
    if _needs_language(text):
        logger.info("[Clarifier] code request without a language → ask language")
        return _LANGUAGE_QUESTION

    # A little recent context helps (e.g. a follow-up that's clear given history).
    context = ""
    if history:
        tail = [m for m in history if m.role in ("user", "assistant")][-4:]
        context = "\n".join(f"{m.role}: {m.content[:300]}" for m in tail)

    user_block = (f"Recent conversation:\n{context}\n\n" if context else "") + \
        f"New user message:\n{text}"
    msgs = [
        MessageDto(role="system", content=_SYSTEM),
        MessageDto(role="user", content=user_block),
    ]

    try:
        result = await route_chat(db, msgs, requested_model=model,
                                  temperature=0.0, max_tokens=400)
        data = _extract_json(result.content)
    except Exception as e:
        logger.warning(f"[Clarifier] assessment failed, skipping: {e}")
        return {"clarify": False}

    if not data or not data.get("clarify"):
        return {"clarify": False}

    q = data.get("question") or {}
    options = [
        {"label": str(o.get("label", "")).strip()[:60],
         "description": str(o.get("description", "")).strip()[:180]}
        for o in (q.get("options") or []) if str(o.get("label", "")).strip()
    ][:4]

    # A valid blocking question needs a prompt and at least two real options.
    if not str(q.get("question", "")).strip() or len(options) < 2:
        return {"clarify": False}

    return {
        "clarify": True,
        "question": {
            "header": str(q.get("header") or "Clarify").strip()[:16] or "Clarify",
            "question": str(q["question"]).strip()[:300],
            "multi_select": bool(q.get("multi_select", False)),
            "options": options,
        },
    }
