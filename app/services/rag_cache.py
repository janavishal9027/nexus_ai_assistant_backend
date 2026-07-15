"""Semantic retrieval cache (Phase 3).

Reuses the retrieved *chunk set* (not a generated answer) for a repeated or
semantically-equivalent query within a scope — a knowledge base or a
conversation. Only the SELECTION of chunks is cached; the LLM still re-grounds a
fresh answer each turn, so answers never go stale. The scope is invalidated
whenever a document is (re)ingested into it. In-process, bounded per scope, TTL'd.

    scope keys:  "kb:<id>"  |  "conv:<id>"

See docs/semantic-embedding/11-implementation-roadmap.md (Phase 3).
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict, defaultdict
from typing import Any, Optional

# scope -> OrderedDict[query_text, (ts_monotonic, query_vec, payload)]
_CACHE: "dict[str, OrderedDict[str, tuple]]" = defaultdict(OrderedDict)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def get(scope: str, query_text: str, query_vec: list[float],
        threshold: float, ttl_s: int) -> Optional[Any]:
    """Return a cached payload for an equivalent query, or None. Exact-text match
    first, then the nearest cached query by cosine similarity ≥ threshold."""
    store = _CACHE.get(scope)
    if not store:
        return None
    now = time.monotonic()
    exact = store.get(query_text)
    if exact and now - exact[0] <= ttl_s:
        store.move_to_end(query_text)
        return exact[2]
    best_payload = None
    best_sim = threshold
    for key, (ts, vec, payload) in list(store.items()):
        if now - ts > ttl_s:
            del store[key]
            continue
        sim = _cosine(query_vec, vec)
        if sim >= best_sim:
            best_sim = sim
            best_payload = payload
    return best_payload


def put(scope: str, query_text: str, query_vec: list[float],
        payload: Any, cap: int) -> None:
    store = _CACHE[scope]
    store[query_text] = (time.monotonic(), list(query_vec), payload)
    store.move_to_end(query_text)
    while len(store) > max(1, cap):
        store.popitem(last=False)


def invalidate(scope: str) -> None:
    """Drop a scope's cache — called when its documents change."""
    _CACHE.pop(scope, None)
