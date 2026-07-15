"""Working memory (Part D) — the in-session ring buffer + scratch pad.

Ephemeral, in-process, bounded. Holds the current conversation's recent turns (so
the Reflector can distil them into episodic/semantic memory at session end — a
later phase) plus a small scratch dict the agent can jot notes into during a turn.
Not persisted; LRU-evicted across conversations and cleared on demand.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Any, Optional

_MAX_TURNS = 40           # recent entries kept per conversation
_MAX_CONVERSATIONS = 500  # LRU cap across all live conversations

_lock = threading.Lock()
# conversation_id -> {"turns": deque, "scratch": dict, "ts": monotonic}
_store: "OrderedDict[int, dict]" = OrderedDict()


def _bucket(conversation_id: int) -> dict:
    b = _store.get(conversation_id)
    if b is None:
        b = {"turns": deque(maxlen=_MAX_TURNS), "scratch": {}, "ts": time.monotonic()}
        _store[conversation_id] = b
    _store.move_to_end(conversation_id)
    while len(_store) > _MAX_CONVERSATIONS:
        _store.popitem(last=False)
    return b


def remember(conversation_id: Optional[int], role: str, text: str) -> None:
    """Append a turn to the conversation's ring buffer."""
    if conversation_id is None or not str(text or "").strip():
        return
    with _lock:
        _bucket(conversation_id)["turns"].append(
            {"role": role, "text": text, "ts": time.time()})


def recent(conversation_id: Optional[int], n: int = 10) -> list[dict]:
    if conversation_id is None:
        return []
    with _lock:
        b = _store.get(conversation_id)
        return list(b["turns"])[-n:] if b else []


def scratch_set(conversation_id: int, key: str, value: Any) -> None:
    with _lock:
        _bucket(conversation_id)["scratch"][key] = value


def scratch_get(conversation_id: int, key: str, default: Any = None) -> Any:
    with _lock:
        b = _store.get(conversation_id)
        return b["scratch"].get(key, default) if b else default


def snapshot(conversation_id: int) -> dict:
    """All working state for a conversation — consumed by the Reflector."""
    with _lock:
        b = _store.get(conversation_id)
        if not b:
            return {"turns": [], "scratch": {}}
        return {"turns": list(b["turns"]), "scratch": dict(b["scratch"])}


def clear(conversation_id: int) -> None:
    """Drop a conversation's working memory (session end)."""
    with _lock:
        _store.pop(conversation_id, None)
