"""Reranker provider abstraction (cross-encoder stage after RRF fusion).

The retrieval pipeline retrieves a wide candidate set (dense ∥ sparse → RRF) then
asks a reranker to reorder it by deep query↔passage relevance and keep the best
few. Unlike the bi-encoder embedding model (which encodes query and document
*separately*), a reranker scores the (query, passage) *pair together*, which is
far more accurate — but slower, so it only runs on the tens of fused candidates.

Backends, tried in ``settings.rag_rerank_preference`` order (first available wins):
  cohere / jina / voyage  →  a real hosted cross-encoder (needs that API key)
  llm                     →  rank with the user's own chat model (keyless)
  heuristic               →  cheap lexical-overlap reorder (keyless, always works)
  (none)                  →  NoOpReranker — preserve the fused order

See docs/semantic-embedding/06-hybrid-search-reranking.md.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0


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


def _identity(n: int, top_k: int) -> list[int]:
    return list(range(n))[:top_k]


def _parse_indices(text: str, n: int, top_k: int) -> list[int]:
    """Parse an LLM's comma/space/newline separated index ranking; dedup, keep
    in-range, then append any indices it omitted so nothing is silently dropped."""
    seen: list[int] = []
    for tok in re.findall(r"\d+", text or ""):
        i = int(tok)
        if 0 <= i < n and i not in seen:
            seen.append(i)
    for i in range(n):
        if i not in seen:
            seen.append(i)
    return seen[:top_k]


class HeuristicReranker(RerankerProvider):
    """Keyless reranker: reorder by query-term overlap (a cheap lexical signal
    the bi-encoder can miss on exact tokens). Better than identity, near-free."""

    name = "heuristic"

    @property
    def enabled(self) -> bool:
        return True

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        terms = {t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 2}
        if not terms:
            return _identity(len(documents), top_k)
        scored = []
        for i, d in enumerate(documents):
            dl = (d or "").lower()
            hits = sum(1 for t in terms if t in dl)
            scored.append((hits, -i, i))   # overlap desc, then original order
        scored.sort(reverse=True)
        return [i for _, _, i in scored][:top_k]


class LLMReranker(RerankerProvider):
    """Keyless cross-encoder using the user's own chat model. One call ranks the
    fused candidates. Robust: falls back to the incoming order on any error."""

    name = "llm"

    def __init__(self, db: Session, owner_id: Optional[int]) -> None:
        self._db = db
        self._owner_id = owner_id

    @property
    def enabled(self) -> bool:
        return True

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        if not documents:
            return []
        if len(documents) == 1:
            return [0]
        from ..services.fallback_router import route_chat
        from ..models.schemas import MessageDto
        listing = "\n".join(f"[{i}] {(d or '')[:400]}" for i, d in enumerate(documents))
        system = (
            "You rank passages by how well each helps answer the user's query. "
            "Return ONLY a comma-separated list of passage numbers, best first "
            "(e.g. '3,0,5,1'). No prose, no explanation."
        )
        user = f"Query: {query}\n\nPassages:\n{listing}\n\nRanking (best first):"
        try:
            res = await route_chat(
                self._db,
                [MessageDto(role="system", content=system),
                 MessageDto(role="user", content=user)],
                temperature=0.0, max_tokens=120,
            )
            order = _parse_indices(res.content or "", len(documents), top_k)
            if order:
                return order
        except Exception as exc:
            logger.warning(f"[RAG] LLM rerank failed ({exc}); keeping fused order")
        return _identity(len(documents), top_k)


class _HostedReranker(RerankerProvider):
    """Cohere/Jina/Voyage-style hosted rerank endpoints (OpenAI-ish shape)."""

    def __init__(self, name: str, url: str, api_key: str, model: str,
                 auth_header: str = "Authorization", auth_prefix: str = "Bearer ") -> None:
        self.name = name
        self._url = url
        self._api_key = api_key
        self._model = model
        self._auth_header = auth_header
        self._auth_prefix = auth_prefix

    @property
    def enabled(self) -> bool:
        return True

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        if not documents:
            return []
        headers = {self._auth_header: f"{self._auth_prefix}{self._api_key}",
                   "Content-Type": "application/json"}
        body = {"model": self._model, "query": query,
                "documents": documents, "top_n": min(top_k, len(documents))}
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(self._url, json=body, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")
            results = resp.json().get("results", [])
            order = [r["index"] for r in results if isinstance(r.get("index"), int)]
            if order:
                return order[:top_k]
        except Exception as exc:
            logger.warning(f"[RAG] {self.name} rerank failed ({exc}); keeping fused order")
        return _identity(len(documents), top_k)


_HOSTED = {
    "cohere": ("https://api.cohere.com/v1/rerank", "rerank-english-v3.0"),
    "jina": ("https://api.jina.ai/v1/rerank", "jina-reranker-v2-base-multilingual"),
    "voyage": ("https://api.voyageai.com/v1/rerank", "rerank-2"),
}


def resolve_reranker(db: Session, owner_id: Optional[int]) -> RerankerProvider:
    """Pick a reranker per ``settings.rag_rerank_preference``; first available
    wins. Disabled → NoOp (preserve fused order)."""
    from ..config import get_settings
    from .embeddings import _find_key
    s = get_settings()
    if not getattr(s, "rag_rerank_enabled", False):
        return NoOpReranker()
    for backend in [p.strip() for p in s.rag_rerank_preference.split(",") if p.strip()]:
        if backend in _HOSTED:
            key = _find_key(db, backend, owner_id)
            if key:
                url, model = _HOSTED[backend]
                logger.info(f"[RAG] Reranker: {backend}/{model}")
                return _HostedReranker(backend, url, key, model)
        elif backend == "llm":
            logger.info("[RAG] Reranker: llm (user chat model)")
            return LLMReranker(db, owner_id)
        elif backend == "heuristic":
            return HeuristicReranker()
    return NoOpReranker()
