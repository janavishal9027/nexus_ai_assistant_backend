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
decide whether the assistant should ask the user one or more clarifying questions
first.

Read the message and judge the user's INTENT. Identify EVERY key detail that is
underspecified — where the best answer would differ a LOT depending on the choice —
and ask a SHORT question for each. A single request often has SEVERAL missing
details; ask about each one (at most 4, most important first). This applies to ANY
domain: code, writing, design, planning, documents, analysis, math, cooking, travel…

Examples (note how one request can need several questions):
- "write a program to remove spaces from a string, and give me a document" →
  (1) which programming language? (2) remove which spaces — all spaces, only
  leading/trailing, or all whitespace? (3) which document format — Word, PDF,
  Markdown?
- "translate this and format it nicely" → target language? + output format?
- "design a logo" → what style? + for what brand/industry?
- "plan a trip" → destination? + dates? + budget?
- "write an essay" → topic? + length? + tone?

Do NOT ask when:
- it's a greeting, thanks, or small talk.
- the request already has enough to give a useful, targeted answer.
- it's a clear factual question with a well-known answer ("capital of France",
  "who wrote Hamlet", "what is 2+2") — just answer it.
- one obvious default clearly satisfies it, or the user wants your judgement
  ("you decide", "anything", "surprise me").

Do NOT manufacture ambiguity. Assume the common, present-day interpretation, and ask
ONLY about details that genuinely change the answer. If a reasonable person would
confidently know what is being asked, answer it.

Each question: give 2-4 concrete, specific options tailored to the request.

Respond with STRICT JSON and nothing else:
{"clarify": false}
OR
{"clarify": true, "questions": [
  {"header": "<label, <=14 chars>", "question": "<one specific question>", "multi_select": false, "options": [{"label": "<short choice>", "description": "<what it means>"}]}
]}"""


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

_LANGUAGE_Q = {
    "header": "Language",
    "question": "Which programming language should I use?",
    "multi_select": False,
    "options": [
        {"label": "Python", "description": "Clear and concise; good default"},
        {"label": "JavaScript", "description": "Web / Node.js"},
        {"label": "Java", "description": "Typed; enterprise / Android"},
        {"label": "C++", "description": "Systems / performance"},
    ],
}


def _needs_language(text: str) -> bool:
    """A code request that doesn't name a language → must ask which one."""
    if _LANG_RE.search(text):
        return False
    return bool(_CODE_RE.search(text) or _CODE_RE2.search(text))


def _is_language_question(q: dict) -> bool:
    h = str(q.get("header", "")).lower()
    qt = str(q.get("question", "")).lower()
    return "language" in h or "programming language" in qt


# ── Deterministic fast-path #2: a document is wanted but no format named ─────
# Like the language rule, this must NOT depend on the (flaky) LLM: when the user
# asks for a downloadable document/file without naming a format, always ask which.

# A specific export format IS named → no need to ask.
_FORMAT_NAMED_RE = re.compile(
    r"\.docx|\bdocx\b|word document|word format|word file|\bpdf\b|\.pdf|"
    r"\bxlsx\b|\bexcel\b|spreadsheet|\bcsv\b|power ?point|\bpptx\b|slide deck|"
    r"\bslides\b|markdown|\.md\b|\.txt|text file|plain text|\bzip\b",
    re.I,
)
# A request to PRODUCE a document / file (not "analyse this document").
_DOC_WANT_RE = re.compile(
    r"\b(give|make|create|generate|write|produce|prepare|want|need|provide|send|"
    r"build|draft)\b.{0,30}\b(document|\bdoc\b|file|report|write[- ]?up|"
    r"hand[- ]?out|resume|cv|cover letter|letter|essay|paper|pdf)\b"
    r"|\bdownloadable\b|\bexport\b|\bas a (document|file|report|pdf)\b"
    r"|\bdownload\b.{0,20}\b(document|file|copy|version)\b",
    re.I | re.S,
)

_DOC_FORMAT_Q = {
    "header": "Doc format",
    "question": "Which document format would you like to download?",
    "multi_select": False,
    "options": [
        {"label": "Word (.docx)", "description": "Editable Word document"},
        {"label": "PDF", "description": "Portable, print-ready"},
        {"label": "Markdown", "description": "Plain .md with formatting"},
        {"label": "Text", "description": "Plain .txt"},
    ],
}


def _needs_doc_format(text: str) -> bool:
    """A request for a downloadable document that doesn't name a format."""
    if _FORMAT_NAMED_RE.search(text):
        return False
    return bool(_DOC_WANT_RE.search(text))


def _is_format_question(q: dict) -> bool:
    h = str(q.get("header", "")).lower()
    qt = str(q.get("question", "")).lower()
    return "format" in h or "format" in qt


def _clean_question(q: dict) -> Optional[dict]:
    """Validate + normalize one question; None if it has no prompt or <2 options."""
    if not isinstance(q, dict):
        return None
    options = [
        {"label": str(o.get("label", "")).strip()[:60],
         "description": str(o.get("description", "")).strip()[:180]}
        for o in (q.get("options") or []) if str(o.get("label", "")).strip()
    ][:4]
    if not str(q.get("question", "")).strip() or len(options) < 2:
        return None
    return {
        "header": str(q.get("header") or "Clarify").strip()[:16] or "Clarify",
        "question": str(q["question"]).strip()[:300],
        "multi_select": bool(q.get("multi_select", False)),
        "options": options,
    }


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
    """Return {"clarify": bool, "questions": [{...}, ...]} — a request can be
    ambiguous in several ways, so this may carry more than one question."""
    text = (message or "").strip()
    if _looks_trivial(text):
        return {"clarify": False, "questions": []}

    needs_lang = _needs_language(text)

    # LLM assessment: find every underspecified detail (up to 4 questions).
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

    questions: list[dict] = []
    try:
        result = await route_chat(db, msgs, requested_model=model,
                                  temperature=0.0, max_tokens=600)
        data = _extract_json(result.content)
        if data and data.get("clarify"):
            # Accept the new "questions" list, or a single legacy "question".
            raw = data.get("questions")
            if not raw and data.get("question"):
                raw = [data["question"]]
            for rq in (raw or [])[:4]:
                cq = _clean_question(rq)
                if cq:
                    questions.append(cq)
    except Exception as e:
        logger.warning(f"[Clarifier] assessment failed: {e}")

    # Guarantee the language question for a code request that named no language —
    # the LLM is unreliable here, so this must not depend on the model's mood.
    if needs_lang and not any(_is_language_question(q) for q in questions):
        logger.info("[Clarifier] code request without a language → ask language")
        questions.insert(0, _LANGUAGE_Q)

    # Guarantee the document-format question when a downloadable document is
    # wanted but no format was named — also deterministic, so multi-question
    # clarification doesn't silently collapse to one when the LLM call fails.
    if _needs_doc_format(text) and not any(_is_format_question(q) for q in questions):
        logger.info("[Clarifier] document wanted without a format → ask format")
        questions.append(_DOC_FORMAT_Q)

    questions = questions[:4]
    return {"clarify": bool(questions), "questions": questions}
