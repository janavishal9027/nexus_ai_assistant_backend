# Evaluation & Observability — Nexus AI

> How we **measure** retrieval/answer quality and **see inside** the RAG pipeline.
> Sibling of `00-ARCHITECTURE.md` (§5 "Cross-cutting concerns"); the target →
> current mapping lives in `10-gap-analysis.md`, the sequencing in
> `11-implementation-roadmap.md`.
>
> **Status:** both artifacts described here — the offline eval harness
> `backend/scripts/rag_eval.py` and the per-request tracer
> `backend/app/services/rag_observability.py` — are **[NEW — being implemented
> now]**. Neither exists in the tree yet; this doc is the design they are built to.

A RAG system you cannot measure is a RAG system you cannot improve: a chunking or
reranking change either helps or hurts, and the only honest way to know is a
labeled set and a metric. A RAG system you cannot *see* is one you cannot debug
in production — when an answer is wrong you need to know whether retrieval missed,
the reranker mis-ordered, or the LLM ignored good context. This file covers both:

- **Part A — Evaluation** — an offline, labeled-set harness that scores *retrieval*
  (Recall@K / MRR / nDCG …) and, layered on top, *RAG answers* (groundedness,
  citation correctness …).
- **Part B — Observability** — a `correlation_id` threaded end-to-end and one
  redacted structured log line per retrieval, plus the metrics worth emitting.

As everywhere in this subsystem, each is written as a **target** (the ideal,
scale-out shape) and the **pragmatic Nexus AI tier** that actually ships on the
single-Postgres stack.

---

## Target architecture

### Part A — Evaluation

#### A.1 Labeled test set

The ground truth is a set of `(query, relevant_chunk_ids)` pairs — for each query,
the chunk ids a correct retriever *should* surface. One JSON object per line
(**JSONL**), so the file is append-friendly and diff-friendly:

```jsonl
{"query": "How do I rotate my API keys?", "relevant_chunk_ids": ["chunk_42", "chunk_43"]}
{"query": "What is the max upload size?",  "relevant_chunk_ids": ["chunk_17"]}
{"query": "Which embedding models are supported?", "relevant_chunk_ids": ["chunk_88", "chunk_91", "chunk_92"]}
```

`relevant_chunk_ids` is a list because a question can be answered by more than one
chunk. Ids reference `document_chunks.id` (the chunk primary key,
`models/rag_models.py:117`); §Pitfalls covers how the harness reconciles the
`"chunk_<id>"` string form with the integer PK, and why the set is only valid
against the exact KB + embedding version it was built on.

#### A.2 Retrieval metrics

Computed by comparing the retriever's ranked `chunk_id` list against
`relevant_chunk_ids`. `K` is aligned with production top-K
(`rag_final_top_k = 6`, `config.py:63`).

| Metric | One-line definition | Reads on |
|---|---|---|
| **Recall@K** | Did a correct chunk appear in the top-K? = (relevant chunks retrieved in top-K) / (total relevant). Recall@5 = "was a correct chunk in the first 5". | coverage — did we find it at all |
| **Precision@K** | (relevant chunks in top-K) / K. How much of the returned context is on-target. | noise in the context window |
| **MRR** (Mean Reciprocal Rank) | 1 / (rank of the *first* correct result), averaged over queries. First hit at rank 2 → 0.5. | how high the first good hit sits |
| **nDCG@K** (Normalized Discounted Cumulative Gain) | Rank-position-weighted gain: a correct chunk earns `1/log2(rank+1)`, summed, then divided by the ideal ordering's score. Rewards putting the best chunk first. | ordering quality across all hits |
| **Hit Rate@K** | Fraction of queries with ≥ 1 relevant chunk in top-K (binary Recall). | "did we whiff entirely" |
| **Reranker accuracy** | Agreement of the reranker's top-K with the labeled relevant set, measured *after* the rerank stage vs. *before* (fusion order). Isolates the reranker's contribution. | is rerank helping or hurting |

#### A.3 RAG-answer metrics (layered on top)

Retrieval metrics need no LLM; these grade the *generated answer* and mostly need
an LLM-judge or human labels, so they are the second tier:

| Metric | What it asks |
|---|---|
| **Answer correctness** | Does the answer match the reference/expected answer? |
| **Groundedness / faithfulness** | Is every claim supported by the retrieved context (no unsupported statements)? |
| **Citation correctness** | Do the inline `[n]` citations point at chunks that actually support the sentence? |
| **Context relevance** | Were the retrieved chunks actually relevant to the question (precision of the context, judged)? |
| **Hallucination rate** | Fraction of answers containing a claim not entailed by the context. |
| **Latency** | End-to-end and per-stage wall-clock (also an observability metric, Part B). |
| **Token usage** | Prompt + completion tokens per request. |
| **Cost per request** | Tokens × provider price. |

#### A.4 The harness — `backend/scripts/rag_eval.py` **[NEW]**

A standalone script (not a unit test — it needs a live DB + real embeddings and is
run on demand / CI-gated). Contract:

```
rag_eval.py --dataset eval/kb_7.jsonl --kb 7 --k 6

  load JSONL  ──▶ for each {query, relevant_chunk_ids}:
                    retrieved = rag_retrieval.retrieve(db, kb, query, owner_id)   # 02-retrieval-pipeline
                    ranked_ids = [r["chunk_id"] for r in retrieved]               # best-first
                    recall@k, precision@k, RR, nDCG@k, hit  ← compare to relevant
  ──▶ aggregate (mean over queries)  ──▶ print report
```

It reuses the production retrieval path verbatim (`rag_retrieval.retrieve`,
`services/rag_retrieval.py:140`) so the eval measures *the real system*, not a
re-implementation. Report shape:

```
RAG retrieval eval — KB 7 — 42 queries — provider=mistral/mistral-embed
  Recall@6     0.88
  Precision@6  0.31
  MRR          0.74
  nDCG@6       0.69
  Hit@6        0.93
  empty        2/42   (queries returning 0 chunks)
```

#### A.5 Worked example of the metric math

One query, `K = 5`. Relevant set = `{chunk_42, chunk_43}`. Retriever returns,
best-first:

```
rank:   1         2         3         4         5
id:     chunk_17  chunk_42  chunk_09  chunk_43  chunk_88
hit?    ·         ✓         ·         ✓         ·
```

- **Recall@5** = 2 relevant found / 2 total relevant = **1.00**
- **Precision@5** = 2 relevant / 5 returned = **0.40**
- **Hit Rate@5** = at least one hit → **1**
- **MRR** = first hit is `chunk_42` at rank 2 → `1/2` = **0.50** (this query's RR; MRR averages RR over all queries)
- **nDCG@5**: gain `= 1/log2(rank+1)` for each hit.

```
DCG@5  = 1/log2(2+1) + 1/log2(4+1)
       = 1/1.585     + 1/2.322
       = 0.6309      + 0.4307      = 1.0616

IDCG@5 = ideal order puts both hits first (ranks 1,2):
       = 1/log2(2)   + 1/log2(3)
       = 1.0000      + 0.6309      = 1.6309

nDCG@5 = DCG/IDCG = 1.0616 / 1.6309 = 0.651
```

The retriever found everything (Recall 1.0) but buried the first hit at rank 2 and
padded with three misses, so nDCG (0.65) and Precision (0.40) tell the real story —
exactly the signal a reranker change should move.

---

### Part B — Observability

#### B.1 The trace: one `correlation_id`, end to end

A single id is minted at the request boundary and carried through every stage so
all logs for one question join on one key — Flutter → FastAPI → embed → vector
search → rerank → LLM:

```
Flutter        POST /api/kb/{id}/chat/stream  (X-Correlation-Id header, else server-minted)
   │
   ▼
FastAPI route  adopt/mint correlation_id → request_context.set_correlation_id(cid)
   │           open RagTrace(cid)
   ▼
Query embed    embeddings.embed_one(q, QUERY)      stage=embed     ms · dim · cache_hit
   │
   ▼
Vector search  pgvector <=> (semantic)             stage=vector    ms · rows · avg_sim
   ∥  keyword   FTS ts_rank                         stage=keyword   ms · rows
   ▼
RRF fusion     reciprocal_rank_fusion(k=60)        candidates_in · fused_out
   │
   ▼
Rerank         reranker.rerank (NoOp today)        stage=rerank    ms · top_k
   │
   ▼
LLM generate   route_stream_chat → SSE             stage=llm       ttft · total_ms · tokens · cost
   │
   ▼
one structured summary line   {cid, stages{…}, counts{…}, avg_sim, empty}   ← redacted
```

#### B.2 Recommended metrics

Per stage and per request (emitted on the summary line and/or as process counters):

| Group | Metrics |
|---|---|
| Indexing | embedding latency, embedding failures, chunks processed / min |
| Retrieval | vector-query latency, DB latency, top-K count, average similarity, reranker latency, empty-retrieval rate |
| Generation | LLM latency, **time-to-first-token (TTFT)**, total response time, tokens / request, cost / request |
| Efficiency | cache-hit rate (placeholder — RAG path is cache-free today, §Pitfalls) |
| Quality signal | user feedback score (thumbs up/down fed back per `correlation_id`) |

#### B.3 What must **never** be logged

Redaction is a hard requirement. Logs must **not** expose:

- **API keys / provider secrets** — log the platform name (`mistral`), never the key.
- **JWTs / auth tokens / session ids.**
- **Private document content** — no chunk `text`, no raw file bytes. Log a chunk
  *id* and *length*, not its content.
- **Full embedding vectors** — log the *dimension* (e.g. `dim=1024`), never the floats.
- **PII** — including the raw query string and document filenames, which can
  themselves be sensitive. Log a query *hash* + *length*, not the query.

This mirrors the posture already in the agent path: `_SENSITIVE_FIELD_NAMES` +
`_strip_sensitive_data()` (`services/agent.py:396-426`), which strips secret keys
from tool-result data before it is logged (`agent.py:448`).

#### B.4 The helper — `backend/app/services/rag_observability.py` **[NEW]**

A lightweight, per-request `RagTrace`: it holds the `correlation_id`, a map of
`stage → elapsed_ms`, and a handful of counts/scalars (candidate count, final K,
`empty` flag, `avg_similarity`, TTFT, tokens). A stage is timed with a small
context manager, and on close the trace emits **one** structured `logger.info`
line — greppable by `correlation_id`, low-overhead, no per-stage log spam:

```
[RAG] cid=7f3c… kb=7 stages={embed:31,vector:12,keyword:8,rerank:0,llm:1840}
      cand=10 final=6 avg_sim=0.71 ttft=210 tokens=734 empty=false
```

By construction the trace has **no field** that can hold a key, a JWT, chunk text,
or a raw vector — only ids, counts, dims, and durations — so redaction is
guaranteed rather than remembered. It reads the `correlation_id` from the existing
`request_context` contextvar and may bump the process-wide Prometheus counters
(`services/observability.py`), but keeps per-request timings on the log line to
avoid unbounded metric cardinality.

---

## Current in Nexus AI (baseline)

Everything below is **verified against the tree** at the cited `file:line`.

### Evaluation — nothing measures retrieval quality today

- **No retrieval-quality eval exists.** There is no `backend/scripts/` directory at
  all, hence no `rag_eval.py`. The test suite under `backend/tests/` covers only:
  - **Citations** — `tests/unit/test_citations.py` exercises the `Citation` /
    `CitationTracker` dedup + formatting logic (source URLs, Req 8.x). Nothing
    about *retrieval* relevance.
  - **Properties / invariants** — `tests/unit/test_properties.py` checks
    orchestration invariants (tool routing, planner index uniqueness, sensitive-field
    stripping, correlation-id-is-UUID). The closest thing to a retrieval assertion is
    Property 5, `test_memory_round_trip` (`test_properties.py:82-91`), which stores
    one text and asserts `similarity >= 0.7` on an exact-match search of the *memory*
    tool — a single-item round-trip, **not** KB/document Recall@K / MRR / nDCG over a
    labeled set.
- No labeled `{query, relevant_chunk_ids}` dataset, no metric code anywhere.

### Observability — standard logging, no correlation id on the RAG path

- The `correlation_id` *infrastructure* exists but is used **only by the agent /
  tool-orchestration path**, not the RAG path:
  - `services/request_context.py:11-13` defines the `_correlation_id` contextvar;
    `:52-57` are its get/set helpers; `get_correlation_id()` returns the literal
    `"unknown"` when unset (`request_context.py:57`).
  - It is minted and threaded in the **agent** path: `routes/agent.py:151` mints
    `str(uuid.uuid4())` and `:156` calls `set_request_context(correlation_id=…)`;
    `services/agent.py:1179` sets it too; `services/tool_router.py:65-107` reads/sets
    it and logs structured lines that include `correlation_id={cid}`.
- The **RAG path never sets or logs a correlation id**:
  - `routes/knowledge.py` calls `request_context.set_owner_id(account.id)` at
    `:257` (search) and `:286` (chat stream) — **owner id only, never a
    correlation id**. The grounded-chat handler `kb_chat_stream`
    (`knowledge.py:279-358`) logs plain, id-less f-strings such as the retrieval
    failure at `:292` and the stream errors at `:352`/`:355` (prefix `[KB/Chat]`).
  - `services/rag_retrieval.py:35` is a bare `logging.getLogger(__name__)`; its only
    log statement is the keyword-search fallback warning at `:109`. `retrieve()`
    (`:140-187`) does the embed (`:147`), the `asyncio.to_thread` search (`:158`),
    RRF (`:160`) and rerank (`:167`) with **no stage timing and no correlation id**,
    then returns dicts each carrying `chunk_id` (`:178-186`) — the exact hook the
    eval harness will read.
  - `services/rag_ingestion.py:27` logger emits plain `[Ingest]` / `[ConvRAG]` lines
    (`:150`, `:154`, `:195`, `:227`, `:230`) — no id, no latency.
  - `providers/embeddings.py:27` logs which provider was chosen (`:210`, `:219`,
    `:225`) but no per-embed latency, failure counter, or id. `providers/reranker.py`
    logs nothing (the `NoOpReranker`, `:30-42`).
- The one existing observability facility is **process-wide Prometheus counters**,
  not per-request tracing: `services/observability.py` defines a singleton
  `ObservabilityCollector` over a fixed `Counters` set — `ws_sessions_active`,
  `kafka_events_published`, `redis_hits/misses`, `memory_chunks_stored`,
  `memory_searches` (`observability.py:9-16`) — rendered as Prometheus text at
  `GET /api/agent/metrics` (`routes/agent.py:278-280`). **None** of those counters
  cover a RAG stage (embedding, vector-query, rerank, similarity, empty-retrieval),
  and there is no per-request correlation.

### Target → current

| Capability | Target | Current in Nexus AI |
|---|---|---|
| Labeled retrieval set | `{query, relevant_chunk_ids}` JSONL | none |
| Retrieval metrics | Recall@K / MRR / nDCG / Precision / Hit | none |
| Eval harness | `scripts/rag_eval.py` | **absent** (no `scripts/` dir) — **[NEW]** |
| Answer metrics | groundedness / citations / hallucination | none (citation *formatting* tested only) |
| Correlation id on RAG path | Flutter → LLM | infra exists, wired on agent path only; RAG path sets `owner_id` only |
| Per-request RAG trace | `RagTrace` one-line summary | **absent** — **[NEW]** |
| Stage metrics | embed/vector/rerank/LLM latency, avg_sim, empty | none on RAG path |
| Process metrics | Prometheus at `/api/agent/metrics` | present, but no RAG counters |
| Secret redaction | keys/JWT/text/vectors/PII never logged | pattern exists (`_strip_sensitive_data`), unused by RAG |

---

## Design decisions

1. **Retrieval metrics before answer metrics.** Recall@K / MRR / nDCG are cheap,
   deterministic, and need no LLM — they isolate the retriever, which is where most
   RAG failures originate. Answer-quality metrics (groundedness, correctness) need an
   LLM-judge, are noisier and cost tokens, so they layer on *after* retrieval is
   trustworthy.
2. **Harness reuses the production path.** `rag_eval.py` calls
   `rag_retrieval.retrieve()` (`services/rag_retrieval.py:140`) rather than
   re-implementing search, so the score reflects the shipping system (fusion knobs,
   provider pinning, reranker) and cannot drift from it.
3. **JSONL, not CSV.** `relevant_chunk_ids` is a nested list; one JSON object per
   line keeps the file append-only, line-diffable, and trivial to stream-parse.
4. **A script, not a pytest.** Eval depends on a live DB and real embedding keys and
   is inherently non-hermetic, so it lives in `scripts/` and runs on demand / in a
   CI gate — it must not slow or flake the unit suite (which stays offline, per
   `test_properties.py`'s "runs without redis/kafka/pgvector" contract).
5. **Reuse the existing `correlation_id` contextvar.** The agent path already threads
   it (`routes/agent.py:156`, `tool_router.py:98`); the RAG path only needs to
   *adopt* it — mint/adopt in `knowledge.py`, read it in `RagTrace`. No new
   mechanism, one join key across both subsystems.
6. **One structured summary line per retrieval.** Not per-stage logging — a single
   greppable line per `correlation_id` keeps overhead and log volume low and makes
   "show me everything about this request" a one-line grep.
7. **Redaction by construction.** `RagTrace` accepts only ids, counts, dims and
   durations — it has no field that could carry a key, token, chunk text, or raw
   vector — so it *cannot* leak, echoing `_strip_sensitive_data` (`agent.py:405`).
8. **Complement the Prometheus collector, don't replace it.** Aggregate counters go
   to `services/observability.py` (bounded cardinality, scraped at
   `/api/agent/metrics`); high-cardinality per-request timings stay on the log line.
9. **Eval K = production K.** Metrics default to `K = rag_final_top_k` (6,
   `config.py:63`) with the fusion top-K (10, `config.py:62`) as the candidate pool,
   so the numbers describe what users actually get.

---

## Pitfalls

1. **Chunk ids are not stable across re-ingest.** `document_chunks.id` is an
   auto-increment PK, and re-ingesting a document **deletes and re-inserts** all its
   chunks (`services/rag_ingestion.py:125`) — the ids change. A labeled set built
   before a re-chunk silently goes stale; rebuild it after any chunking change.
2. **The set is bound to one embedding space.** Labels are only valid against the KB
   + embedding model/version they were captured on. Switching the KB's pinned
   provider (Arch §5, same-space rule) re-ranks everything; carry the
   `provider/model` in the dataset header and refuse to score across a mismatch.
3. **Id-format mismatch scores a silent zero.** Labels use strings like `"chunk_42"`;
   `retrieve()` returns integer `chunk_id`s (`rag_retrieval.py:178-186`). The harness
   must normalize (strip the `chunk_` prefix / compare as int) or every query reports
   Recall 0 while looking like it ran.
4. **Reranker accuracy is degenerate today.** `resolve_reranker` returns the
   `NoOpReranker` (`providers/reranker.py:30-42`), which is the identity permutation —
   "reranker accuracy" is trivially the pre-rerank order until a real cross-encoder
   lands. Don't over-read it.
5. **Hash-embedding fallback poisons the numbers.** With no embedding key the pipeline
   falls back to the keyless `HashEmbedding` (`providers/embeddings.py:140-169`,
   selected at `:225-229`), which is lexical, not semantic. An eval run under hash
   mode measures hashing, not retrieval — assert a real provider (or record
   `is_fallback` and flag the run).
6. **Cache-hit rate is a placeholder.** The RAG path is cache-free today (Arch §1 —
   Redis is flag-gated off); the `cache_hit` field will read `false`/`0` until a
   query/embedding cache exists. Ship the field, but don't chart it yet.
7. **`correlation_id` defaults to `"unknown"`.** `get_correlation_id()` returns the
   literal `"unknown"` when unset (`request_context.py:57`). If `knowledge.py` doesn't
   mint/adopt an id at the route boundary, *every* RAG trace logs
   `cid=unknown` and the Flutter↔backend join collapses — set it at the entry point,
   not deep in the stack.
8. **Time the awaited call, not the sync inner function.** Embedding + search run
   inside `asyncio.to_thread` (`rag_retrieval.py:158`); stage timers must wrap the
   `await`, or thread-pool queueing time is mis-attributed to the wrong stage.
9. **Small sets are noisy.** Recall@K / MRR over a handful of queries swing wildly.
   Always report `N`, and treat deltas below a threshold as noise, not signal.
10. **LLM-judge answer metrics are non-deterministic and cost tokens.** Groundedness /
    correctness scored by a model vary run-to-run and add latency + cost — pin the
    judge model, sample rather than scoring every request, and treat the result as a
    trend line, not a pass/fail gate.
11. **The query and filenames are PII too.** It is tempting to log the raw query for
    debugging; don't — log a hash + length. Document names can leak just as much as
    the text they name.
