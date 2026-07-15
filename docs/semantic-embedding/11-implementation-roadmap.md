# Implementation Roadmap — Nexus AI semantic embedding

> Phased plan. **Phase 1 shipped in this pass**; later phases are the scale-out
> tier. Cross-references: `10-gap-analysis.md` for status per capability.

---

## Phase 1 — quality core (SHIPPED ✅)

All on the existing FastAPI + Postgres/pgvector stack (no new infra). Verified by
`ast` parse of every file, full `app.main` import, DDL applied to the live DB, and
an end-to-end ingest→dedup→retrieve run.

1. **Schema** — new `document_chunks` columns (`content_hash, section, page_number,
   char_start, char_end, parent_chunk_id, is_parent, embedding_model,
   embedding_version`) + `documents.content_hash`, via idempotent `ALTER TABLE …
   IF NOT EXISTS` in `main.py::_ensure_rag_schema` (Postgres) and the SQLAlchemy
   models (SQLite/dev).
2. **Content-hash dedup** — `rag_ingestion.py` skips re-embedding an unchanged
   upload (`documents.content_hash`); conversation attachments dedup by hash too.
3. **Structure-aware + parent/child chunking** — `rag_chunking.py::chunk_document`
   detects heading structure, builds section paths, and emits small **child**
   chunks (searched) inside larger **parent** chunks (returned to the LLM).
4. **Embedding hardening** — retries+backoff, LRU cache, optional L2-normalize,
   per-chunk model/version tags (`providers/embeddings.py`).
5. **Query rewriting** — `services/rag_query.py` turns referential follow-ups into
   standalone queries (gated; only fires when it helps).
6. **Real reranker** — `providers/reranker.py`: hosted cross-encoder
   (Cohere/Jina/Voyage if key) → keyless LLM reranker → heuristic → NoOp.
7. **Retrieval upgrade** — wide candidate set (`rag_candidate_top_k=30`) → rerank →
   `_expand_parents` (small-to-big) + dedup, in both KB and conversation paths.
8. **Evaluation** — `scripts/rag_eval.py` (Recall@K/Precision@K/MRR/nDCG/HitRate) +
   `scripts/rag_eval.sample.jsonl`.
9. **Observability** — `services/rag_observability.py::RagTrace` logs one redacted
   structured line per retrieval (correlation id + per-stage ms + counts).

**Config knobs** (all in `config.py`, override via env): `rag_child_chunk_size`,
`rag_parent_chunk_size`, `rag_structure_aware`, `rag_candidate_top_k`,
`rag_rerank_enabled`, `rag_rerank_preference`, `rag_query_rewrite`,
`rag_embed_max_retries`, `rag_embed_cache_size`, `rag_embed_normalize`,
`rag_dedup_by_hash`.

> **Apply it:** restart the backend so `_ensure_rag_schema` runs
> (`cd backend && "C:/Python314/python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8080`).
> Existing chunks keep working (new columns are nullable / `is_parent` NULL treated
> as a child); **re-ingest** a document to get section metadata + parent/child.

---

## Phase 2 — precision & safety (SHIPPED ✅)

- **`owner_id` in the retrieval WHERE** — `rag_retrieval.py::_scope_owner` filters
  `owner_id` (plus shared/NULL rows) in every search: `semantic_search`,
  `keyword_search`, `_semantic_search_conv`, `_keyword_search_conv`, and the
  Python fallback; `owner_id` passed from both orchestrators. Verified: owner B
  gets 0 results from owner A's KB.
- **PDF page numbers** — `rag_chunking.py::extract_pages`/`build_paged_text`
  preserve page boundaries; `page_for_offset` tags each chunk (and parent) with
  `page_number`; citations render `file.pdf · p.4 · Section` via `_source_label`.
  Verified: a 2-page PDF produced chunks on pages 1 and 2.
- **RAG-answer grading** — `scripts/rag_eval.py --grade` generates a grounded
  answer then LLM-judges groundedness / citation-correctness / relevance (0–1),
  alongside Recall@K / MRR / nDCG.

Still open here: **chunk-level dedup** (reuse `content_hash` to skip re-embedding
unchanged chunks on partial edits) — deferred to Phase 3.

## Phase 3 — scale (SHIPPED ✅: ANN index, chunk-dedup, semantic cache)

- **HNSW ANN index** ✅ — a fixed `Vector(ANN_DIM)` mirror column
  (`document_chunks.embedding_ann`, `ANN_DIM` = env `RAG_ANN_DIM`, default 1024 =
  mistral-embed) carries an **HNSW** (`vector_cosine_ops`) index
  (`main.py::_ensure_ann_index`, own transaction, best-effort). Chunks whose
  vector matches `ANN_DIM` are mirrored at ingest; `semantic_search` /
  `_semantic_search_conv` use the ANN column when `len(query_vec)==ANN_DIM` and
  fall back to the exact `<=>` scan on the unbounded column otherwise (and on any
  ANN error). Verified: column + HNSW index exist; ANN branch executes; other-dim
  / failure paths fall back cleanly.
- **Chunk-level dedup** ✅ — `rag_ingestion.py::_embed_and_store_chunks` reuses the
  embeddings of unchanged chunks (by `content_hash`) from a prior ingest of the
  same document, re-embedding only new/edited chunks (persists across restarts,
  unlike the in-process embedding cache). Verified: **reused 5/10** after editing
  one section.
- **Semantic retrieval cache** ✅ — `services/rag_cache.py` reuses the retrieved
  chunk-set for a repeated / semantically-equivalent query per scope (`kb:<id>` /
  `conv:<id>`); invalidated whenever a document is (re)ingested; only the chunk
  SELECTION is cached (the LLM still re-grounds a fresh answer, so answers never
  go stale). Config: `rag_semantic_cache_*`. Verified: 2nd identical retrieve is a
  cache hit.

**Deferred — multi-tenancy / Postgres RLS.** This needs an organizations/teams
data model the app does not have yet (today's isolation is single-level
`owner_id`, now enforced in-SQL by `_scope_owner`). Adding `org_id`/`tenant_id` +
RLS is a **product-level** change (accounts → orgs, membership, sharing) that
should follow the product introducing organizations — it is not a RAG-layer
change. Path when ready: add `org_id` to `knowledge_bases`/`documents`/
`document_chunks`, backfill from `owner_id`, add it to `_scope_owner`, then enable
RLS policies keyed on a `SET app.current_org` GUC. See `08-security-multitenancy.md`.

## Phase 4 — event-driven platform (SHIPPED ✅ — opt-in, default off)

All three backends are wired behind config flags and **default off**; the app
runs the in-process pipeline (BackgroundTasks + in-DB BYTEA) untouched until you
enable them. Infra to run them locally: `backend/docker-compose.phase4.yml`
(Redpanda + MinIO + Jaeger).

- **Object storage** ✅ — `services/object_store.py`: `rag_object_store="db"`
  (BYTEA, default) or `"s3"` (MinIO/S3 via boto3). One seam
  (`store_/load_/delete_document_bytes`) used by upload, ingest, reingest and
  delete; `documents.storage_key` holds the S3 key, `raw` is NULL. Fail-fast at
  boot if `"s3"` is misconfigured. Verified: db path store→load round-trips.
- **Kafka event-driven indexing** ✅ — `services/rag_events.py` publishes
  `nexus.document.uploaded`; `app/workers/rag_index_worker.py` consumes it, runs
  the (idempotent) `ingest_document`, emits `nexus.document.indexed`, retries via a
  re-published event carrying an `attempt` count, and routes exhausted docs to
  `nexus.document.dlq`. `rag_ingestion.dispatch_ingestion` publishes when the
  broker is up, else falls back to BackgroundTasks — **uploads never fail**. Run
  the worker in-process (lifespan) or standalone
  (`python -m app.workers.rag_index_worker`). Enable: `RAG_KAFKA_INDEXING=true`.
- **Distributed tracing** ✅ — `rag_observability.init_otel` exports each
  `RagTrace` as an OTLP span (correlation id + per-stage timings + counts) to a
  collector (`RAG_OTEL_ENABLED=true`, `RAG_OTEL_ENDPOINT`); the structured log
  line is always emitted regardless.

> **Scope note:** code + infra are complete and default-off; the Kafka/MinIO/OTel
> **round-trips require the compose stack running** — here they were verified by
> import, config wiring and the graceful-fallback path, not against live brokers.

---

## Status: Phases 1–4 shipped

The semantic-embedding subsystem now spans quality (P1), precision & safety (P2),
single-node scale (P3), and an opt-in event-driven platform (P4). The only item
consciously deferred is **org/tenant Postgres RLS** (P3 note) — it needs a
product-level organizations model, not a RAG-layer change.

---

## Rollback / safety notes

- Every Phase-1 column is nullable and additive; the DDL is idempotent. No data
  migration is required and old chunks remain searchable.
- Reranking and query-rewrite are **config-gated** — set `rag_rerank_enabled=false`
  / `rag_query_rewrite=false` to revert to pure hybrid+RRF with zero extra LLM calls.
- The embedding cache is in-process and self-bounding; it clears on restart.
