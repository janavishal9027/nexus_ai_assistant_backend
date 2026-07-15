# Gap Analysis — target vs current — Nexus AI

> Maps the target semantic-embedding architecture (`00-ARCHITECTURE.md`) onto the
> real code, with `file:line` anchors. Status legend:
> ✅ done · 🟩 **done this pass** · 🟨 partial · ⬜ deferred (scale-out tier).

---

## 1. Indexing pipeline

| Capability | Status | Where |
|---|---|---|
| Multi-format extraction (PDF/DOCX/40+ text/code) | ✅ | `services/rag_chunking.py::extract_text` |
| Clean/normalize (NFC, whitespace, control-strip) | ✅ | `rag_chunking.py::clean_text` |
| In-process async ingestion + job/progress | ✅ | `services/rag_ingestion.py`, `ingestion_jobs` |
| **Content-hash dedup** (skip unchanged re-ingest) | 🟩 | `rag_ingestion.py::ingest_document` (`_sha256`, `documents.content_hash`) |
| **Chunk-level dedup** (reuse unchanged chunk embeddings) (Phase 3) | 🟩 | `_embed_and_store_chunks` reuses by `content_hash`; re-embeds only edited chunks |
| **Structure-aware chunking** (heading paths) | 🟩 | `rag_chunking.py::chunk_document`, `_split_sections` |
| **Parent/child chunking** (small-to-big) | 🟩 | `rag_chunking.py::chunk_document`; `rag_ingestion.py::_embed_and_store_chunks` |
| **Per-chunk metadata** (section/offsets/hash/model) | 🟩 | `DocumentChunk` new columns (`models/rag_models.py`) |
| **Page-number capture from PDF** (Phase 2) | 🟩 | `rag_chunking.py::extract_pages`/`build_paged_text`/`page_for_offset` → `page_number` on chunks + citations |
| Event-driven (Kafka) indexing + DLQ | ⬜ | scale-out tier; worker is lift-and-shift ready |
| MinIO/S3 object storage | ⬜ | `documents.raw` BYTEA is the seam |

## 2. Embedding service

| Capability | Status | Where |
|---|---|---|
| Provider abstraction, per-user-key auto-select | ✅ | `providers/embeddings.py::resolve_embedding_provider` |
| Asymmetric query/passage input types | ✅ | `INPUT_QUERY`/`INPUT_PASSAGE` |
| Order-preserving batching | ✅ | `EmbeddingProvider.embed` |
| **Retries + exponential backoff** | 🟩 | `EmbeddingProvider._embed_with_retry` |
| **In-process embedding cache (LRU)** | 🟩 | `_EMBED_CACHE`, `_cache_get/_cache_put` |
| **Optional L2 normalization** | 🟩 | `_l2_normalize`, `rag_embed_normalize` |
| **Per-chunk model/version provenance** | 🟩 | `DocumentChunk.embedding_model/embedding_version` |
| KB embedding-space pinning | ✅ | `rag_ingestion.py` (KB pin), `embedding_provider_for_kb` |
| Distributed cache (Redis) / cost tracking | ⬜ | Redis exists but flag-gated off |

## 3. Vector storage

| Capability | Status | Where |
|---|---|---|
| pgvector cosine `<=>` + JSON fallback | ✅ | `rag_models.py`, `rag_retrieval.py::semantic_search` |
| Postgres FTS (GIN) keyword search | ✅ | `main.py::_ensure_rag_schema`, `keyword_search` |
| **Owner-scoped isolation in SQL** (Phase 2) | 🟩 | `rag_retrieval.py::_scope_owner` filters `owner_id` in every search (defense-in-depth) |
| **HNSW ANN on RAG chunks** (Phase 3) | 🟩 | fixed `Vector(ANN_DIM)` mirror `embedding_ann` + HNSW index; exact-scan fallback for other dims / errors |
| Fixed-dim per-model tables | ⬜ | mirror column covers the primary dim; per-model tables still deferred |

## 4. Retrieval

| Capability | Status | Where |
|---|---|---|
| Hybrid dense∥sparse + RRF (k=60) | ✅ | `rag_retrieval.py::reciprocal_rank_fusion` |
| **Wide candidate set → rerank → final** | 🟩 | `rag_candidate_top_k=30` → rerank → `rag_final_top_k=6` |
| **Query rewriting (standalone follow-ups)** | 🟩 | `services/rag_query.py::rewrite_query`, wired in `retrieve`/`retrieve_conversation_context` |
| **Real reranker** (hosted / LLM / heuristic) | 🟩 | `providers/reranker.py` (was `NoOpReranker`) |
| **Parent expansion + dedup in context** | 🟩 | `rag_retrieval.py::_expand_parents` |
| Parents excluded from search | 🟩 | `is_parent` filter on keyword search; parents have no embedding |
| Grounded, cite-from-sources prompt | ✅ | `GROUNDED_SYSTEM`, `build_grounded_messages` |
| Token-budget compression (conv path) | ✅ | `retrieve_conversation_context` |
| **Semantic retrieval cache** (Phase 3) | 🟩 | `services/rag_cache.py`, per-scope, similarity-gated, invalidated on ingest |
| Intent classification / routing | 🟨 | agent has tool routing; RAG path uses rewrite only |
| MMR diversity | ⬜ | dedup by parent instead |

## 5. Security / multi-tenancy

| Capability | Status | Where |
|---|---|---|
| JWT owner scoping on all KB routes | ✅ | `routes/knowledge.py` (`get_current_account`) |
| Prompt-injection defense (evidence-not-instructions) | ✅ | `GROUNDED_SYSTEM` |
| **`owner_id` in the retrieval WHERE (defense-in-depth)** (Phase 2) | 🟩 | `rag_retrieval.py::_scope_owner`, passed from both orchestrators |
| org/tenant columns, Postgres RLS | ⬜ | single-level `owner_id` today |

## 6. Evaluation & observability

| Capability | Status | Where |
|---|---|---|
| **Retrieval eval (Recall@K/MRR/nDCG/Hit)** | 🟩 | `scripts/rag_eval.py` + `scripts/rag_eval.sample.jsonl` |
| **Per-retrieval trace (correlation id + stage ms)** | 🟩 | `services/rag_observability.py::RagTrace`, wired in retrieval |
| **RAG-answer grading (groundedness/citation/relevance)** (Phase 2) | 🟩 | `scripts/rag_eval.py --grade` (generate + LLM judge) |
| Prometheus RAG metrics endpoint | ⬜ | agent metrics exist; no RAG-specific series yet |

---

## 7. Summary

**Phase 1 (🟩):** content-hash dedup · structure-aware + parent/child chunking with
per-chunk metadata · embedding retries + cache + normalization + versioning · query
rewriting · a real reranker (hosted cross-encoder / keyless LLM / heuristic) ·
wide-candidate → rerank → parent-expand retrieval · retrieval evaluation harness ·
per-retrieval observability trace.

**Phase 2 (🟩):** `owner_id` defense-in-depth filter in every RAG search · PDF page
numbers threaded onto chunks + richer citations (`file · p.4 · section`) · LLM-graded
answer correctness (`rag_eval.py --grade`: groundedness / citation / relevance).

**Phase 3 (🟩):** HNSW ANN index via a fixed-dim `embedding_ann` mirror column
(exact-scan fallback) · chunk-level dedup (reuse unchanged chunk embeddings) ·
semantic retrieval cache (per-scope, invalidated on ingest).

**Still open (⬜):** Postgres RLS + org/tenant columns (**deferred — needs a
product-level organizations model**, see roadmap Phase 3 note) · per-model fixed-dim
tables · Kafka/MinIO event platform (roadmap Phase 4).
