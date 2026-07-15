"""Lightweight per-retrieval tracing for the RAG pipeline.

One ``RagTrace`` threads a correlation id through a single retrieval and records
per-stage latencies + counts, then emits ONE structured summary line. It never
logs secrets, document text, or vectors — only ids, timings and counts. Reuses
the app-wide correlation id when the request already has one (agent path).

    trace = RagTrace("kb")
    with trace.stage("rewrite"):   ...
    with trace.stage("embed"):     ...
    trace.count("candidates", 30); trace.count("final", 6)
    trace.done()   # → [RAGTRACE] cid=… scope=kb total_ms=… embed_ms=… final=6

See docs/semantic-embedding/09-evaluation-observability.md.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("rag.trace")

# Optional OpenTelemetry tracer (Phase 4). Off unless init_otel() succeeds.
_tracer = None


def init_otel(endpoint: str) -> None:
    """Wire an OTLP span exporter so each retrieval also emits an OTel span.
    Raises if opentelemetry isn't installed (caller logs + continues)."""
    global _tracer
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("nexus.rag")


def _current_cid() -> str:
    # Prefer the app's request correlation id if the agent path set one.
    try:
        from .request_context import get_correlation_id  # type: ignore
        cid = get_correlation_id()
        if cid and cid != "unknown":
            return cid
    except Exception:
        pass
    return uuid.uuid4().hex[:12]


class RagTrace:
    def __init__(self, scope: str, correlation_id: Optional[str] = None) -> None:
        self.scope = scope
        self.cid = correlation_id or _current_cid()
        self._stages: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._meta: dict[str, str] = {}
        self._t0 = time.monotonic()

    @contextmanager
    def stage(self, name: str):
        t = time.monotonic()
        try:
            yield
        finally:
            self._stages[name] = round((time.monotonic() - t) * 1000, 1)

    def count(self, name: str, n: int) -> None:
        self._counts[name] = int(n)

    def note(self, key: str, value) -> None:
        self._meta[key] = str(value)[:80]

    def done(self) -> None:
        total = round((time.monotonic() - self._t0) * 1000, 1)
        parts = [f"cid={self.cid}", f"scope={self.scope}", f"total_ms={total}"]
        parts += [f"{k}_ms={v}" for k, v in self._stages.items()]
        parts += [f"{k}={v}" for k, v in self._counts.items()]
        parts += [f"{k}={v}" for k, v in self._meta.items()]
        try:
            logger.info("[RAGTRACE] " + " ".join(parts))
        except Exception:
            pass
        if _tracer is not None:
            try:
                span = _tracer.start_span(f"rag.{self.scope}")
                span.set_attribute("rag.correlation_id", self.cid)
                span.set_attribute("rag.total_ms", total)
                for k, v in self._stages.items():
                    span.set_attribute(f"rag.stage.{k}_ms", v)
                for k, v in self._counts.items():
                    span.set_attribute(f"rag.count.{k}", v)
                for k, v in self._meta.items():
                    span.set_attribute(f"rag.{k}", v)
                span.end()
            except Exception:
                pass
