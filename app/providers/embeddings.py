"""Embedding provider abstraction for the RAG pipeline.

The retrieval/ingestion code depends ONLY on the ``EmbeddingProvider`` interface
(``embed(texts) -> list[vector]``), never on a concrete SDK — so a provider can
be swapped without touching the pipeline (spec: "Create interfaces so providers
can be replaced without changing the RAG pipeline").

``resolve_embedding_provider`` auto-detects which provider to use from the keys
the user has stored, following ``settings.rag_embedding_preference``. A keyless
deterministic ``HashEmbedding`` is the last-resort fallback so ingestion still
completes without any embedding key (retrieval quality is poor in that mode and
the job status flags it).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

import httpx
from sqlalchemy import or_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# input_type hints for asymmetric embedding models (query vs stored passage).
INPUT_QUERY = "query"
INPUT_PASSAGE = "passage"

_HTTP_TIMEOUT = 60.0
_BATCH = 64                 # texts per request (keeps payloads well-bounded)
_HASH_DIM = 768

# In-process embedding cache: key = sha256(model|input_type|text) → vector.
# Identical content (re-ingest, repeated query) reuses the vector instead of
# re-calling the provider. Bounded LRU; cleared on restart (see
# docs/semantic-embedding/04-embedding-service.md).
_EMBED_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()


def _cache_get(key: str) -> Optional[list[float]]:
    vec = _EMBED_CACHE.get(key)
    if vec is not None:
        _EMBED_CACHE.move_to_end(key)
    return vec


def _cache_put(key: str, vec: list[float], cap: int) -> None:
    _EMBED_CACHE[key] = vec
    _EMBED_CACHE.move_to_end(key)
    while len(_EMBED_CACHE) > max(0, cap):
        _EMBED_CACHE.popitem(last=False)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class EmbeddingProvider(ABC):
    """Turns text into vectors. ``platform``/``model`` identify what produced a
    vector so a KB can stay pinned to one embedding space."""

    platform: str = "unknown"
    model: str = "unknown"
    version: str = ""          # optional model-revision tag (provenance)

    def __init__(self) -> None:
        self._dim: Optional[int] = None

    @property
    def dim(self) -> Optional[int]:
        """Vector dimension, known once at least one embed has run."""
        return self._dim

    @property
    def is_fallback(self) -> bool:
        return False

    @abstractmethod
    async def _embed_batch(self, texts: list[str], input_type: Optional[str]) -> list[list[float]]:
        ...

    def _cache_key(self, text: str, input_type: Optional[str]) -> str:
        raw = f"{self.platform}/{self.model}@{self.version}|{input_type or ''}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _embed_with_retry(self, texts: list[str], input_type: Optional[str]) -> list[list[float]]:
        """One batch with exponential backoff on transient provider errors."""
        from ..config import get_settings
        retries = max(1, get_settings().rag_embed_max_retries)
        delay = 0.5
        last: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return await self._embed_batch(texts, input_type)
            except Exception as exc:  # network / 429 / 5xx
                last = exc
                if attempt + 1 >= retries:
                    break
                logger.warning(f"[Embed] {self.platform}/{self.model} attempt "
                               f"{attempt + 1}/{retries} failed ({exc}); retrying")
                await asyncio.sleep(delay)
                delay *= 2
        raise last if last else RuntimeError("embedding failed")

    async def embed(self, texts: list[str], input_type: Optional[str] = None) -> list[list[float]]:
        """Embed a list of texts. Batches large inputs, retries transient
        failures with backoff, serves repeats from an in-process cache, and
        optionally L2-normalizes. Order is preserved."""
        if not texts:
            return []
        from ..config import get_settings
        s = get_settings()
        normalize = bool(getattr(s, "rag_embed_normalize", False))
        cap = int(getattr(s, "rag_embed_cache_size", 4096))

        keys = [self._cache_key(t, input_type) for t in texts]
        results: list[Optional[list[float]]] = [
            (_cache_get(k) if cap > 0 else None) for k in keys
        ]
        missing = [i for i, r in enumerate(results) if r is None]

        if missing:
            fresh: list[list[float]] = []
            miss_texts = [texts[i] for i in missing]
            for i in range(0, len(miss_texts), _BATCH):
                fresh.extend(await self._embed_with_retry(miss_texts[i:i + _BATCH], input_type))
            if len(fresh) != len(missing):
                raise RuntimeError("embedding count mismatch")
            for j, i in enumerate(missing):
                vec = _l2_normalize(fresh[j]) if normalize else fresh[j]
                results[i] = vec
                if cap > 0:
                    _cache_put(keys[i], vec, cap)

        out = [r for r in results if r is not None]
        if out and self._dim is None:
            self._dim = len(out[0])
        return out

    async def embed_one(self, text: str, input_type: Optional[str] = None) -> list[float]:
        vecs = await self.embed([text], input_type)
        return vecs[0] if vecs else []


class OpenAICompatEmbedding(EmbeddingProvider):
    """Any provider exposing an OpenAI-style ``POST /embeddings`` endpoint
    (Mistral, OpenAI, Vercel AI Gateway, NVIDIA NIM, …)."""

    def __init__(self, platform: str, base_url: str, model: str, api_key: str,
                 send_input_type: bool = False, extra_headers: Optional[dict] = None) -> None:
        super().__init__()
        self.platform = platform
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._send_input_type = send_input_type
        self._extra_headers = extra_headers or {}

    async def _embed_batch(self, texts: list[str], input_type: Optional[str]) -> list[list[float]]:
        body: dict = {"model": self.model, "input": texts}
        if self._send_input_type and input_type:
            body["input_type"] = input_type
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(f"{self._base_url}/embeddings", json=body, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"{self.platform} embeddings error {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json().get("data", [])
        # Sort by index so the returned order matches the input order.
        data = sorted(data, key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]


class GeminiEmbedding(EmbeddingProvider):
    """Google Generative Language embeddings (text-embedding-004)."""

    _URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        super().__init__()
        self.platform = "google"
        self.model = model
        self._api_key = api_key

    async def _embed_batch(self, texts: list[str], input_type: Optional[str]) -> list[list[float]]:
        task = "RETRIEVAL_QUERY" if input_type == INPUT_QUERY else "RETRIEVAL_DOCUMENT"
        model_path = f"models/{self.model}"
        requests = [{
            "model": model_path,
            "content": {"parts": [{"text": t}]},
            "taskType": task,
        } for t in texts]
        url = self._URL.format(model=self.model)
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(url, params={"key": self._api_key},
                                     json={"requests": requests})
        if resp.status_code != 200:
            raise RuntimeError(f"google embeddings error {resp.status_code}: {resp.text[:300]}")
        return [e["values"] for e in resp.json().get("embeddings", [])]


class HashEmbedding(EmbeddingProvider):
    """Deterministic keyless fallback: identical text → identical unit vector.
    Poor semantic quality — used only when no embedding key is configured so
    ingestion still succeeds and the pipeline stays exercised."""

    def __init__(self, dim: int = _HASH_DIM) -> None:
        super().__init__()
        self.platform = "hash"
        self.model = f"local-hash-{dim}"
        self._dim = dim

    @property
    def is_fallback(self) -> bool:
        return True

    async def _embed_batch(self, texts: list[str], input_type: Optional[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        dim = self._dim or _HASH_DIM
        vec = [0.0] * dim
        tokens = (str(text).lower().split() or [str(text).lower()])
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            anchor = int.from_bytes(digest[:2], "big")
            for i, b in enumerate(digest):
                vec[(i * 7 + anchor) % dim] += (b - 127.5) / 127.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ─── Auto-detection ─────────────────────────────────────────────────────────

# platform → (base_url, model, needs input_type). Google handled separately.
_OPENAI_COMPAT_EMBED: dict[str, tuple[str, str, bool]] = {
    "mistral": ("https://api.mistral.ai/v1", "mistral-embed", False),
    "openai": ("https://api.openai.com/v1", "text-embedding-3-small", False),
    "vercel": ("https://ai-gateway.vercel.sh/v1", "openai/text-embedding-3-small", False),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "nvidia/nv-embedqa-e5-v5", True),
}


def _find_key(db: Session, platform: str, owner_id: Optional[int]) -> Optional[str]:
    """First enabled, non-errored key for a platform, scoped to the owner
    (their own keys plus shared/global NULL-owner keys)."""
    from ..models.db_models import ApiKey
    q = db.query(ApiKey).filter(
        ApiKey.platform == platform,
        ApiKey.enabled == True,          # noqa: E712
        ApiKey.status != "error",
    )
    if owner_id is not None:
        q = q.filter(or_(ApiKey.owner_id == owner_id, ApiKey.owner_id.is_(None)))
    row = q.first()
    return row.api_key if row else None


def resolve_embedding_provider(db: Session, owner_id: Optional[int]) -> EmbeddingProvider:
    """Pick an embedding provider from the keys the user holds, following
    ``settings.rag_embedding_preference``. Falls back to a keyless local
    encoder if none match."""
    from ..config import get_settings
    preference = [p.strip() for p in get_settings().rag_embedding_preference.split(",") if p.strip()]

    for platform in preference:
        if platform == "hash":
            break
        if platform == "google":
            key = _find_key(db, "google", owner_id)
            if key:
                logger.info("[RAG] Embeddings: google/text-embedding-004")
                return GeminiEmbedding(api_key=key)
            continue
        spec = _OPENAI_COMPAT_EMBED.get(platform)
        if not spec:
            continue
        key = _find_key(db, platform, owner_id)
        if key:
            base_url, model, needs_input_type = spec
            logger.info(f"[RAG] Embeddings: {platform}/{model}")
            return OpenAICompatEmbedding(
                platform=platform, base_url=base_url, model=model,
                api_key=key, send_input_type=needs_input_type,
            )

    logger.warning(
        "[RAG] No embedding-capable key found; using the keyless local hash "
        "encoder (low retrieval quality — add a Mistral key for real embeddings)"
    )
    return HashEmbedding()


def embedding_provider_for_kb(
    db: Session, owner_id: Optional[int],
    platform: Optional[str], model: Optional[str], dim: Optional[int],
) -> EmbeddingProvider:
    """Build the exact embedding provider a KB is pinned to (set at first
    ingest) so queries and later documents embed into the same vector space.
    Falls back to fresh auto-detection if the pinned key is no longer present."""
    if not platform:
        return resolve_embedding_provider(db, owner_id)
    if platform == "hash":
        return HashEmbedding(dim=dim or _HASH_DIM)
    if platform == "google":
        key = _find_key(db, "google", owner_id)
        if key:
            return GeminiEmbedding(api_key=key, model=model or "text-embedding-004")
    else:
        spec = _OPENAI_COMPAT_EMBED.get(platform)
        key = _find_key(db, platform, owner_id)
        if spec and key:
            base_url, default_model, needs_input_type = spec
            return OpenAICompatEmbedding(
                platform=platform, base_url=base_url, model=model or default_model,
                api_key=key, send_input_type=needs_input_type,
            )
    logger.warning(
        f"[RAG] KB is pinned to '{platform}' but no key is available; "
        f"falling back to auto-detection (retrieval quality may suffer)"
    )
    return resolve_embedding_provider(db, owner_id)
