"""Shared test helpers for the agent-orchestration suite."""


class MockMemoryTool:
    """In-memory memory tool using the real deterministic embedder + cosine, so
    the store→search round-trip can be tested without a database (Property 5)."""

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.threshold = similarity_threshold
        self._rows: list[tuple] = []   # (id, text, conversation_id, embedding)
        self._next_id = 1

    async def store(self, text: str, conversation_id: int, user_id=None) -> dict:
        from app.tools.memory_tool import _embed
        emb = _embed(text)
        mid = self._next_id
        self._next_id += 1
        self._rows.append((mid, text, conversation_id, emb))
        return {"status": "success", "memory_id": mid}

    async def search(self, query: str, conversation_id=None, top_k: int = 5) -> dict:
        from app.tools.memory_tool import _embed, _cosine
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
