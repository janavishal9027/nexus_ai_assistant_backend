"""Shared test helpers for the agent-orchestration suite."""

import hashlib
import math

_DIM = 256


def _embed(text: str) -> list[float]:
    """Self-contained deterministic pseudo-embedder for the store→search property
    test (Property 5). Identical text → identical unit vector. Kept in the test
    suite so the property doesn't depend on the production memory internals (which
    now use the real RAG embedding provider)."""
    vec = [0.0] * _DIM
    for tok in (str(text).lower().split() or [str(text).lower()]):
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        for i, b in enumerate(digest):
            vec[(i * 7 + (int.from_bytes(digest[:1], "big"))) % _DIM] += (b - 127.5) / 127.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class MockMemoryTool:
    """In-memory memory tool using a self-contained deterministic embedder +
    cosine, so the store→search round-trip can be tested without a database
    (Property 5)."""

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.threshold = similarity_threshold
        self._rows: list[tuple] = []   # (id, text, conversation_id, embedding)
        self._next_id = 1

    async def store(self, text: str, conversation_id: int, user_id=None) -> dict:
        emb = _embed(text)
        mid = self._next_id
        self._next_id += 1
        self._rows.append((mid, text, conversation_id, emb))
        return {"status": "success", "memory_id": mid}

    async def search(self, query: str, conversation_id=None, top_k: int = 5) -> dict:
        q = _embed(query)
        scored = []
        for mid, text, conv, emb in self._rows:
            if conversation_id is not None and conv != conversation_id:
                continue
            sim = _cosine(q, emb)
            if sim >= self.threshold:
                scored.append((sim, mid, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return {
            "status": "success",
            "chunks": [{"memory_id": m, "text": t, "similarity": s} for s, m, t in scored[:top_k]],
        }


def create_mock_memory_tool(similarity_threshold: float = 0.7) -> MockMemoryTool:
    return MockMemoryTool(similarity_threshold=similarity_threshold)
