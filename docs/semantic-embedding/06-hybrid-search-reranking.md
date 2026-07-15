# Hybrid Search & Reranking — Nexus AI

> The two middle stages of the read path: **hybrid retrieval** (dense ∥ sparse →
> fusion) and **reranking** (reorder the fused candidates by deep relevance).
> Companion to `02-retrieval-pipeline.md` (the full read path) and
> `00-ARCHITECTURE.md` §2.2. Storage details for the dense/sparse indexes live in
> `05-vector-storage.md`.

---

## 0. Intro — why one retriever is not enough

Semantic (dense vector) search is the headline feature of a RAG system, and for
paraphrase-style questions it is exactly right: *"How do I reset my password?"*
finds *"steps to recover account access"* even with zero shared words. But dense
search has a systematic blind spot, and the fix is to run a second, completely
different retriever alongside it and merge the results — then optionally reorder
the survivors with a slower, sharper model.

```
        Query
          │
    ┌─────┴─────┐
    ▼           ▼
 Dense        Sparse            two independent rankings
 (pgvector    (Postgres FTS
  cosine)      ts_rank / BM25)
    │           │
    └─────┬─────┘
          ▼
   Score fusion (RRF)            merge into one ranking
          │
          ▼
   Rerank (cross-encoder)        reorder the top-N by deep relevance
          │
          ▼
      Final top 5–10
```

This doc covers the **why** (§1), the **target** design of both stages (§2), what
Nexus AI **actually runs today** (§3), the **decisions** behind it (§4), and the
**pitfalls** (§5).

---

## 1. Why vector-alone fails

Embeddings encode *meaning*, so they are strongest when meaning is the query and
weakest when the query **is a literal string that must match exactly**. Dense
vectors smear rare, precise tokens into their semantic neighborhood — which is
the opposite of what you want when the token *is* the answer:

| Query contains | Vector search tends to return | Why it fails |
|---|---|---|
| exact error code `NullPointerException` | text about "null values", "exceptions in general" | the literal class name is what matters |
| product / class name `RerankerProvider` | semantically "similar" class names | you need *that* symbol, not a cousin |
| API endpoint `/api/auth/refresh` | prose about "authentication and refreshing" | the path is an exact key |
| version number `pgvector 0.8.1` | any text about pgvector, any version | `0.8.1` ≠ `0.7.0` — the digits matter |
| CVE / identifier `CVE-2026-12345` | other CVEs, security prose | one specific advisory |
| rare acronym `RRF`, `HNSW` | expanded/adjacent concepts | the acronym is a precise handle |

The pattern: **low-frequency, high-precision tokens** — error codes, identifiers,
endpoints, versions, acronyms — are exactly the tokens embeddings generalize away.
Keyword search has the opposite profile: it nails the literal string and is blind
to paraphrase. Running both and fusing them covers each other's failure mode.
That is the entire argument for hybrid search.

---

## 2. Target architecture

### 2.1 Two retrievers, one ranking

- **Dense / vector.** Encode the query into a vector, find nearest neighbors by
  cosine distance. Great at paraphrase and concept matching; weak at exact tokens.
- **Sparse / keyword (BM25 / Postgres FTS).** Match query terms against an
  inverted index, score by term frequency × inverse document frequency. Great at
  exact tokens and rare terms; blind to synonyms.

The two produce **two independent rankings of the same corpus**. The problem is
merging them, because their scores are not comparable — a cosine distance of
`0.18` and a `ts_rank` of `0.06` live on different scales with no shared zero.

### 2.2 Fusion option A — weighted score fusion

Normalize both score lists to `[0,1]`, then blend:

```
final_score(d) = 0.65 * vector_score(d) + 0.35 * keyword_score(d)
```

Simple and tunable, **but fragile**: it requires the two scores to be normalized
onto a comparable scale, and the "right" weights (here 0.65 / 0.35) drift with
corpus, model, and query type. One retriever returning wildly scaled scores can
dominate the blend. Use it only when you trust both score distributions.

### 2.3 Fusion option B — Reciprocal Rank Fusion (chosen)

RRF throws away the raw scores and fuses on **rank position** alone, so no
normalization or comparable scales are needed:

```
score(d) = Σ  1 / (k + rank_d)      summed over each ranked list d appears in
           lists
```

- `rank_d` is the document's 0-based position in that list (best = 0).
- `k` is a damping constant, conventionally `k ≈ 60`. Larger `k` flattens the
  weighting (rank 1 vs rank 5 matters less); smaller `k` sharpens it.
- A document ranked highly by **both** retrievers accumulates score from both and
  floats to the top; a document only one retriever found still contributes.

RRF is the right default for hybrid search precisely because dense and sparse
scores are *not* comparable — it sidesteps the normalization problem entirely.

### 2.4 Reranking — two different kinds of model

The retrieval embedder and the reranker sound similar but are architecturally
opposite, and the difference is the whole point:

| | Retrieval embedding model (bi-encoder) | Reranker (cross-encoder) |
|---|---|---|
| Encodes | query and doc **separately**, into vectors | query and doc **together**, as one input |
| Output | a vector per text (compare with cosine) | a single relevance score for the pair |
| Speed | very fast — docs pre-embedded, ANN over **millions** | slow — one model pass **per candidate** |
| Precision | good (no cross-attention between q and d) | high (full attention across q and d) |
| Role | first stage: cast a wide net over the corpus | second stage: sharpen a short list |

A bi-encoder must embed every document *before* the query exists, so query and
document never "see" each other — fast, but it leaves relevance on the table. A
cross-encoder reads the query and a candidate **jointly**, so it can judge
"does *this* passage actually answer *this* question?" — far more accurate, but it
cannot pre-compute anything, so it only scales to tens of candidates.

### 2.5 Target production flow

You use both, in sequence, playing to each one's strength:

```
corpus (10^5–10^6 chunks)
   │  bi-encoder ANN + keyword  →  cheap, wide recall
   ▼
retrieve 30–100 candidates
   │  cross-encoder  →  expensive, narrow precision
   ▼
rerank
   │
   ▼
keep top 5–10  →  LLM context
```

Retrieve wide with the cheap retriever, rerank narrow with the expensive one, keep
only what fits the context budget. The reranker never touches the full corpus, so
its cost stays bounded regardless of KB size.

---

## 3. Current in Nexus AI (baseline)

All of this is implemented in `backend/app/services/rag_retrieval.py`. Two
scope-parallel copies exist — KB search (`semantic_search`/`keyword_search`) and
conversation search (`_semantic_search_conv`/`_keyword_search_conv`) — with
identical mechanics.

### 3.1 Dense branch

`semantic_search()` (`rag_retrieval.py:50-66`):

- pgvector cosine distance via the `<=>` operator —
  `DocumentChunk.embedding.cosine_distance(query_vec)` at `rag_retrieval.py:56`,
  ordered **ascending** (nearest first) at `rag_retrieval.py:61`, limited to
  `settings.rag_semantic_top_n` (default `20`, `config.py:60`).
- Filters on `knowledge_base_id` and `embedding.isnot(None)`
  (`rag_retrieval.py:59-60`).
- When pgvector is absent, `_semantic_python()` (`rag_retrieval.py:69-87`)
  brute-forces cosine in Python — a pure fallback, no index.

### 3.2 Sparse branch

`keyword_search()` (`rag_retrieval.py:90-118`):

- Postgres full-text search: `to_tsvector('english', text)` @@
  `plainto_tsquery('english', query)` — built at `rag_retrieval.py:96-97`, matched
  with the `@@` operator and ordered by `ts_rank(...)` descending at
  `rag_retrieval.py:103`, limited to `settings.rag_keyword_top_n` (default `20`,
  `config.py:61`).
- Backed by a functional **GIN index** created at startup —
  `CREATE INDEX ... ix_document_chunks_fts ON document_chunks USING gin
  (to_tsvector('english', text))` in `_ensure_rag_schema()` (`main.py:67-70`).
- **ILIKE fallback**: if FTS raises (it shouldn't in production), the code rolls
  back and falls back to a `%query%` `ILIKE` scan (`rag_retrieval.py:108-118`).

### 3.3 Fusion

`reciprocal_rank_fusion()` (`rag_retrieval.py:123-135`):

- Exactly the RRF formula from §2.3: `score(d) += 1.0 / (k + rank + 1)` at
  `rag_retrieval.py:132` (the `+1` makes the effective rank 1-based).
- `k = settings.rag_rrf_k` (default `60`, `config.py:65`); returns the top
  `settings.rag_fusion_top_k` (default `10`, `config.py:62`) chunks best-first with
  their fused scores.

### 3.4 Threading

Both branches run **sequentially on one worker thread** via
`asyncio.to_thread(_search_both)` (`rag_retrieval.py:152-158`, and
`rag_retrieval.py:276-281` for the conversation path). This keeps the synchronous
DB work off the event loop **without** sharing the SQLAlchemy `Session` across
threads — a `Session` is not safe for concurrent use, so the two searches are
deliberately serial, not parallel. (The module docstring, `rag_retrieval.py:1-18`,
describes exactly this "semantic + keyword → RRF → top-K" sequence.)

### 3.5 Reranking — wired in, but a no-op today

The reranker interface exists, the call site exists, and the implementation is an
identity function:

- `providers/reranker.py` defines the `RerankerProvider` ABC
  (`reranker.py:17-27`) with `async def rerank(query, documents, top_k) ->
  list[int]` returning candidate indices best-first, truncated to `top_k`.
- `NoOpReranker` (`reranker.py:30-36`) is the only implementation: it returns
  `list(range(len(documents)))[:top_k]` (`reranker.py:36`) — i.e. it **preserves
  the incoming RRF order** and just truncates. No model call, no reordering.
- `resolve_reranker(db, owner_id)` (`reranker.py:39-42`) **always** returns
  `NoOpReranker()` today (`reranker.py:42`); the comment marks the seam where a
  real provider gets wired once a rerank key is configured.

The pipeline already calls it correctly, so dropping in a real reranker needs
**no change to `rag_retrieval.py`**:

- KB path: `reranker = resolve_reranker(...)` then
  `order = await reranker.rerank(query, docs_text, settings.rag_final_top_k)` at
  `rag_retrieval.py:165-167`, keeping `rag_final_top_k` (default `6`,
  `config.py:63`) chunks.
- Conversation path: the same two calls at `rag_retrieval.py:287-288`.

Net effect today: **"rerank" is a truncation of the RRF ranking to the top 6.**
Retrieval quality is entirely the hybrid+RRF stage; the precision-sharpening
second stage is a placeholder.

### 3.6 Being added by the implementation

A **real reranker** — an LLM cross-encoder / LLM-judge, or a hosted reranker such
as Cohere Rerank or Jina Reranker — is being implemented to replace `NoOpReranker`
behind the same `resolve_reranker` seam (`reranker.py:39-42`), following the
production flow in §2.5 (retrieve 30–100 → rerank → keep top 5–10). Because the
interface and call sites already exist, this is an additive change:
`resolve_reranker` returns the new provider and the pipeline is unchanged. Marked
`[REAL, was NoOp]` in `00-ARCHITECTURE.md` §2.2/§3.

---

## 4. Design decisions / how it works

- **RRF over weighted fusion.** Nexus AI fuses on rank, not score
  (`rag_retrieval.py:123-135`), precisely because dense cosine distance and FTS
  `ts_rank` are not comparable scales (§2.2). RRF needs no normalization and no
  per-corpus weight tuning — the module docstring calls this out explicitly
  (`rag_retrieval.py:11-18`).

- **`k = 60` as a sane default.** The conventional RRF constant
  (`rag_rrf_k=60`, `config.py:65`) flattens the contribution curve enough that a
  document strong in *one* retriever still ranks well, while agreement across both
  retrievers still wins. It is exposed as a setting so it can be tuned per
  evaluation (`09-evaluation-observability.md`).

- **Recall-then-precision fan-out.** Fan out to `20 + 20` candidates
  (`config.py:60-61`), fuse to `10` (`config.py:62`), rerank to `6`
  (`config.py:63`). The wide-then-narrow shape is what lets a cheap retriever and
  an (eventual) expensive reranker each do the job they are good at.

- **The reranker is a swappable provider, off by default.** Modeling reranking as
  a provider interface with a no-op default (`reranker.py`) means hybrid retrieval
  works with **zero** extra keys, and a cross-encoder can be added later with no
  pipeline change — the same "interfaces so providers can be replaced" principle
  the embedding layer follows (`embeddings.py:1-13`).

- **Unbounded vector column, exact `<=>` scan.** Chunk embeddings use an
  unbounded pgvector `Vector()` (`rag_models.py:37`, `131`) so any model's
  dimension fits without migration; the cost is no ANN index, i.e. an exact cosine
  scan — fine at personal-KB scale, and the reason the scale-out tier (HNSW) is
  documented but deferred (`05-vector-storage.md`, `00-ARCHITECTURE.md` §7).

- **One thread, two searches.** Serial-on-one-thread
  (`asyncio.to_thread`, `rag_retrieval.py:152-158`) is a correctness decision, not
  a missed optimization — it keeps blocking DB work off the event loop while
  respecting SQLAlchemy `Session` thread-affinity.

---

## 5. Pitfalls

- **Shipping vector-only search.** Without the sparse branch, every exact-match
  query in §1 (`NullPointerException`, `/api/auth/refresh`, `CVE-2026-12345`,
  version numbers, rare acronyms) degrades to "semantically nearby but wrong." The
  keyword branch (`rag_retrieval.py:90-118`) is not optional polish — it is half
  the recall.

- **Blending non-comparable scores.** Weighted fusion (`0.65·vector + 0.35·keyword`)
  silently misbehaves unless both score lists are normalized onto the same scale,
  and the weights drift with corpus and query type. RRF avoids this by design —
  don't "upgrade" to score-blending without a normalization step and an eval.

- **Mis-tuning `k`.** Setting RRF `k` too small over-weights the single best rank
  and effectively ignores the second retriever; too large flattens everything into
  a near-tie. The default `60` (`config.py:65`) is a starting point to validate,
  not a constant to trust blindly.

- **Reading the no-op reranker as real relevance.** Today `rerank()` truncates the
  RRF order (`reranker.py:35-36`) — "top 6" means "top 6 by fusion rank," not "top
  6 a cross-encoder judged most relevant." Any quality claim that assumes deep
  reranking is currently unfounded until §3.6 lands.

- **Reranking the whole corpus.** A cross-encoder scores one (query, doc) pair per
  forward pass; pointing it at the full KB instead of the fused top-N (10–100) is
  quadratically expensive and defeats the two-stage design. Rerank tens, never
  millions.

- **FTS language mismatch.** The GIN index and query both hard-code the `'english'`
  text-search config (`main.py:69`, `rag_retrieval.py:96-97`). Non-English content
  is stemmed/stopword-filtered as if it were English, quietly degrading keyword
  recall — a known limitation of the current single-config setup.

- **Silent ILIKE degradation.** If FTS ever fails and the `ILIKE` fallback engages
  (`rag_retrieval.py:108-118`), keyword search loses ranking and stemming — it
  becomes a substring match. It keeps results flowing, but recall/precision drop
  without an obvious error; watch for it in logs (`"Keyword search failed"`,
  `rag_retrieval.py:109`).

- **Parallelizing the two searches naively.** Splitting dense and sparse onto two
  threads that share one `Session` will corrupt it — the serial-on-one-thread
  design (`rag_retrieval.py:152-158`) is deliberate. If true parallelism is ever
  needed, give each branch its **own** session.
