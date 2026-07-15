# Semantic Embedding Architecture — Nexus AI

> Master architecture for the semantic‑embedding / RAG subsystem of **Nexus AI**
> (Flutter client + FastAPI backend + PostgreSQL/pgvector).
> This file is the single source of truth; the `NN-*.md` analysis files drill into
> each stage, `10-gap-analysis.md` maps target → current code, and
> `11-implementation-roadmap.md` sequences the work.

---

## 0. What a semantic embedding is (and why it matters here)

A **semantic embedding** is a numeric vector that encodes *meaning, intent and
context* rather than surface tokens. Two texts that mean the same thing land near
each other in vector space even when they share no words:

```
"How can I reset my password?"      ─┐
"I forgot my login credentials."     ├─ near each other in vector space
"Where do I change my account pw?"  ─┘
"Chocolate cake recipe"              ── far away
```

Similarity is measured with **cosine similarity** (`1.0` identical → `0.0`
unrelated). Nexus AI compares vectors with pgvector's cosine distance operator
`<=>` (`similarity = 1 - distance`).

All semantic embeddings are embeddings; not all embeddings are semantic. Nexus AI
uses semantic text embeddings for: document RAG (knowledge bases), per‑conversation
attachment RAG, and (separately) conversation **memory** retrieval.

---

## 1. Stack context (what we build on)

| Concern | Nexus AI choice | Notes |
|---|---|---|
| Client | Flutter (Windows/Android/iOS/Web) | uploads via multipart; SSE for grounded chat |
| API | FastAPI (uvicorn :8080) | JWT, owner‑scoped |
| Relational + vector store | PostgreSQL 18 + **pgvector 0.8.x** | one database, no separate vector DB |
| Object storage | `documents.raw BYTEA` (in‑DB today) | seam for MinIO/S3 later |
| Async work | FastAPI `BackgroundTasks` / `asyncio.create_task` | no Celery/Kafka in the RAG path |
| Embeddings | provider abstraction over user API keys | Mistral/OpenAI/Vercel/NVIDIA/Google + keyless hash fallback |
| Keyword search | Postgres FTS (`to_tsvector` + GIN) | hybrid with dense |
| Cache | Redis present but flag‑gated/off | RAG path is currently cache‑free |

**Design principle — two tiers.** Everything below is written as a *target*
architecture (the "ideal, scale‑out" system, including Kafka/MinIO/HNSW/RLS) and a
*pragmatic Nexus AI adaptation* (what actually ships on the single‑Postgres stack).
The roadmap only implements the pragmatic tier; the ideal tier is documented so the
system can grow into it without a rewrite.

---

## 2. The two pipelines

A semantic‑embedding system is two pipelines that meet at the vector store.

```
                    INDEXING (write path)                RETRIEVAL (read path)
  Source ─▶ Ingest ─▶ Parse/Clean ─▶ Dedup ─▶ Chunk ─▶   Query ─▶ Understand/Rewrite ─▶
  ─▶ Embed ─▶ Store (pgvector + FTS)                     Embed ─▶ Hybrid search ─▶ Rerank ─▶
                                    │                    Context build ─▶ LLM ─▶ Grounded answer
                                    └──────────  vector store  ──────────┘
```

### 2.1 Indexing pipeline (detail → `01-indexing-pipeline.md`, `03-chunking.md`)

```
Data sources (PDF/DOCX/MD/code/chat/…)
   │
   ▼  UploadFile (multipart)                    routes/knowledge.py
Ingestion service ── create Document(raw=bytes), IngestionJob, enqueue
   │                                            services/rag_ingestion.py
   ▼
Parse + clean ── extract_text() + clean_text()  services/rag_chunking.py
   │             (preserve structure/headings; keep code blocks)
   ▼
Dedup ── SHA‑256(normalized) → skip unchanged   [NEW]
   │
   ▼
Chunk ── structure‑aware → recursive → parent/child, with metadata   [ENHANCED]
   │     (section path, page, char offsets, token count)
   ▼
Embed ── batch → vector, with retries + cache + versioning   providers/embeddings.py [HARDENED]
   │
   ▼
Store ── document_chunks(embedding pgvector, text, metadata)   models/rag_models.py
         + GIN FTS index    (+ HNSW when fixed‑dim, target tier)
```

### 2.2 Retrieval pipeline (detail → `02-retrieval-pipeline.md`, `06-hybrid-search-reranking.md`)

```
Flutter query
   │  POST /api/kb/{id}/chat/stream  |  agent turn w/ conversation docs
   ▼
Query understanding ── normalize, detect intent, resolve "it"/"that",
   │                    rewrite to a standalone query               [NEW]
   ▼
Query embedding ── same model/space as the stored chunks
   │
   ▼
Hybrid retrieval ── dense (pgvector <=>) + sparse (FTS ts_rank)
   │               ⨁ Reciprocal Rank Fusion (k=60)        services/rag_retrieval.py
   │               + access‑control filter (owner_id/scope) in SQL
   ▼
Rerank ── cross‑encoder / LLM judge on top‑N candidates   providers/reranker.py [REAL, was NoOp]
   │
   ▼
Context build ── parent expansion, content dedup, order, token budget   [ENHANCED]
   │
   ▼
LLM generation ── grounded, cite‑only‑from‑sources prompt
   │
   ▼
Flutter ── streamed answer + Sources/citation chips
```

---

## 3. Component responsibilities (map to code)

| Stage | Target responsibility | Nexus AI file(s) |
|---|---|---|
| Ingestion | accept content, create job, kick off async processing | `routes/knowledge.py`, `services/rag_ingestion.py` |
| Parse/clean | extract structured text, normalize, keep headings/code | `services/rag_chunking.py` (`extract_text`, `clean_text`) |
| Dedup | content‑hash skip; chunk‑level idempotency | `rag_ingestion.py` (+ `documents.content_hash`) **[NEW]** |
| Chunk | structure‑aware → recursive → parent/child + metadata | `services/rag_chunking.py` **[ENHANCED]** |
| Embed | batch, retry, cache, normalize, version | `providers/embeddings.py` **[HARDENED]** |
| Store | pgvector + FTS + metadata, scoped by owner | `models/rag_models.py`, `main.py` DDL |
| Query understanding | normalize, intent, rewrite/standalone | `services/rag_query.py` **[NEW]** |
| Hybrid search | dense + sparse + RRF + ACL filter | `services/rag_retrieval.py` |
| Rerank | reorder top‑N by deep relevance | `providers/reranker.py` **[REAL]** |
| Context build | parent expansion, dedup, order, budget, cite | `services/rag_retrieval.py` **[ENHANCED]** |
| Generate | grounded prompt, streaming, citations | `routes/knowledge.py`, `services/agent.py` |
| Evaluate | Recall@K / MRR / nDCG on a labeled set | `scripts/rag_eval.py` **[NEW]** |
| Observe | correlation id + per‑stage metrics/logs | `services/rag_observability.py` **[NEW]** |

---

## 4. Data model (target vs current) — see `05-vector-storage.md`

Core table `document_chunks` today stores: `id, document_id, knowledge_base_id,
conversation_id, owner_id, ordinal, text, token_count, embedding, created_at`.

Target adds per‑chunk provenance + structure + linkage:

```
document_chunks
  … existing …
  content_hash            sha256(normalized chunk text)      [dedup, cache key]
  section                 heading path e.g. "Auth ▸ Refresh" [structure metadata]
  page_number             source page (PDF)                  [citation]
  char_start, char_end    offsets into cleaned document      [highlighting]
  parent_chunk_id         → document_chunks.id               [small‑to‑big retrieval]
  is_parent               bool                               [parent vs child]
  embedding_model         e.g. "mistral/mistral-embed"       [versioning]
  embedding_version       e.g. "2026-06"                     [migration safety]
documents
  content_hash            sha256(raw)                        [skip re‑ingest]
```

`embedding` remains an **unbounded pgvector `Vector()`** so any auto‑detected model
dimension works without migration (cost: no ANN index — exact `<=>` scan, fine at
personal‑KB scale). The scale‑out tier fixes the dimension per model and adds an
**HNSW** index (`05-vector-storage.md`).

---

## 5. Cross‑cutting concerns

- **Security / multi‑tenancy** (`08-security-multitenancy.md`): every query filters
  by `owner_id` **inside the SQL** — authorization at retrieval, never after.
  Retrieved document text is **untrusted evidence**, never instructions
  (prompt‑injection defense in the grounded system prompt).
- **Embedding compatibility**: document and query vectors must come from the *same*
  model/version. KBs pin their model on first ingest; the query path rebuilds that
  exact provider. Never mix vector spaces.
- **Evaluation** (`09-evaluation-observability.md`): a labeled `{query,
  relevant_chunk_ids}` set drives Recall@K, MRR, nDCG; RAG answers add groundedness
  and citation‑correctness.
- **Observability**: a `correlation_id` threads Flutter → FastAPI → embed → search →
  rerank → LLM; per‑stage latency + similarity + cache‑hit metrics are logged
  (never keys/tokens/PII/full embeddings).

---

## 6. Request lifecycle (end‑to‑end, grounded chat)

```
1. Flutter POST /api/kb/{id}/chat/stream {query, history, model}
2. FastAPI: authn/authz, validate, mint correlation_id
3. rag_query.rewrite(query, history)            → standalone query (+ intent)
4. embeddings.resolve(owner).embed([q], QUERY)  → query_vec (cached)
5. rag_retrieval: dense(<=>) ∥ sparse(FTS)       → RRF(k=60) → top‑N (owner‑scoped)
6. reranker.rerank(q, candidates)                → top‑K by relevance
7. context builder: parent‑expand, dedup, order, token budget → context + sources
8. build_grounded_messages(q, context, history)  → LLM (stream)
9. SSE: first event = sources + conversationId; then content deltas; final done
10. Flutter renders streamed answer + Sources chips; metrics/logs flushed
```

---

## 7. Non‑goals / explicitly deferred (scale‑out tier)

Documented but **not** implemented on the single‑Postgres stack (see roadmap §"Later"):

- Kafka event‑driven indexing (`document.uploaded → … → document.indexed`) — the
  in‑process `BackgroundTasks` worker is written to be lift‑and‑shift compatible.
- MinIO/S3 object storage — `documents.raw` BYTEA is the single seam to replace.
- HNSW/IVFFlat ANN on RAG chunks — requires fixed‑dim columns (per‑model tables).
- Postgres Row‑Level Security / org/tenant columns — today's isolation is `owner_id`.
- Distributed semantic answer cache — Redis exists but is flag‑gated off.

---

## 8. File index

| Doc | Contents |
|---|---|
| `00-ARCHITECTURE.md` | this file — the whole system |
| `01-indexing-pipeline.md` | sources → ingest → parse → dedup → chunk → embed → store |
| `02-retrieval-pipeline.md` | query → understand → embed → hybrid → rerank → context → LLM |
| `03-chunking.md` | fixed/recursive/semantic/structure‑aware/parent‑child |
| `04-embedding-service.md` | model selection, batching, retries, cache, normalize, versioning |
| `05-vector-storage.md` | pgvector, unbounded vs fixed dim, HNSW/IVFFlat, FTS, metadata |
| `06-hybrid-search-reranking.md` | dense+sparse, RRF, cross‑encoder rerank |
| `07-context-building-rag.md` | dedup, parent expansion, budget, grounded prompt, citations |
| `08-security-multitenancy.md` | ACL‑in‑SQL, isolation, prompt‑injection defense |
| `09-evaluation-observability.md` | Recall@K/MRR/nDCG, tracing, metrics |
| `10-gap-analysis.md` | target → current code, file:line, what's missing |
| `11-implementation-roadmap.md` | phased plan + status |
