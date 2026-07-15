"""Object storage for uploaded document bytes (Phase 4).

Two backends behind one seam so the rest of the app never cares where bytes live:

  "db"  (default) — bytes stay in ``documents.raw`` (BYTEA). Zero extra infra.
  "s3"            — bytes go to MinIO/S3 at ``documents.storage_key``; ``raw`` is
                    NULL. Offloads large blobs out of Postgres.

Selected by ``settings.rag_object_store``. The S3 client (boto3, S3-compatible so
it also drives MinIO via an endpoint URL) is imported lazily — if "s3" is
configured without boto3 we fail fast with a clear message rather than silently
losing data. See docs/semantic-embedding/11-implementation-roadmap.md (Phase 4).
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class ObjectStore(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...
    @abstractmethod
    def get(self, key: str) -> bytes: ...
    @abstractmethod
    def delete(self, key: str) -> None: ...


class S3ObjectStore(ObjectStore):
    """MinIO / S3-compatible object store via boto3."""

    def __init__(self) -> None:
        from ..config import get_settings
        s = get_settings()
        try:
            import boto3  # type: ignore
            from botocore.config import Config as _BotoConfig  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "rag_object_store='s3' but boto3 is not installed "
                "(pip install boto3)"
            ) from exc
        self._bucket = s.rag_s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=(s.rag_s3_endpoint or None),
            aws_access_key_id=(s.rag_s3_access_key or None),
            aws_secret_access_key=(s.rag_s3_secret_key or None),
            region_name=s.rag_s3_region,
            config=_BotoConfig(signature_version="s3v4"),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except Exception:
            try:
                self._client.create_bucket(Bucket=self._bucket)
                logger.info(f"[ObjectStore] Created bucket '{self._bucket}'")
            except Exception as exc:  # pragma: no cover
                logger.warning(f"[ObjectStore] Could not ensure bucket: {exc}")

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read()

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)


_store: Optional[ObjectStore] = None
_resolved = False


def get_object_store() -> Optional[ObjectStore]:
    """The active object store, or None for the in-DB (BYTEA) default. Built once."""
    global _store, _resolved
    if _resolved:
        return _store
    from ..config import get_settings
    mode = (get_settings().rag_object_store or "db").lower()
    if mode == "s3":
        _store = S3ObjectStore()
        logger.info("[ObjectStore] Using S3/MinIO backend")
    else:
        _store = None
    _resolved = True
    return _store


# ── Document-level helpers (used by the upload / ingest / delete paths) ──────

def store_document_bytes(document, content: bytes) -> None:
    """Persist a document's bytes to the active backend, setting either
    ``document.raw`` (db) or ``document.storage_key`` (s3)."""
    store = get_object_store()
    if store is None:
        document.raw = content
        document.storage_key = None
    else:
        key = f"documents/{uuid.uuid4().hex}/{(document.filename or 'file')[:120]}"
        store.put(key, content)
        document.storage_key = key
        document.raw = None


def load_document_bytes(document) -> bytes:
    """Read a document's bytes from wherever they live."""
    key = getattr(document, "storage_key", None)
    if key:
        store = get_object_store()
        if store is not None:
            return store.get(key)
    return bytes(document.raw or b"")


def delete_document_bytes(document) -> None:
    """Best-effort remove a document's S3 object (no-op in db mode)."""
    key = getattr(document, "storage_key", None)
    if not key:
        return
    store = get_object_store()
    if store is None:
        return
    try:
        store.delete(key)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[ObjectStore] delete failed for {key}: {exc}")
