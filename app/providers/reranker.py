"""Reranker provider abstraction (optional cross-encoder stage).

The retrieval pipeline calls ``RerankerProvider.rerank`` after RRF fusion. The
default ``NoOpReranker`` keeps the fused order unchanged, so hybrid retrieval
works with no extra key. A real cross-encoder (Cohere Rerank, Jina Reranker, a
local model, …) can be added later by implementing this interface and returning
it from ``resolve_reranker`` — no change to the pipeline is required.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy.orm import Session


class RerankerProvider(ABC):
    name: str = "none"

    @property
    def enabled(self) -> bool:
        return False

    @abstractmethod
    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Return document indices ordered best-first, truncated to ``top_k``."""
        ...


class NoOpReranker(RerankerProvider):
    """Identity reranker: preserves the incoming (RRF) order."""

    name = "none"

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        return list(range(len(documents)))[:top_k]


def resolve_reranker(db: Session, owner_id: Optional[int]) -> RerankerProvider:
    # Reranking is intentionally disabled for now (see module docstring). Wire a
    # concrete provider here once a rerank key is configured.
    return NoOpReranker()
