# Embedding Service — Nexus AI

> The stage that turns cleaned text into vectors: `providers/embeddings.py`.
> Drill-down for the **Embed** box in `00-ARCHITECTURE.md` (§2.1 / §3).
> Target = the hardened, cache-and-retry embedding service; baseline = what
> ships today on the provider-abstraction over user API keys.

The embedding service is the single choke point where *meaning* is minted. Both
pipelines pass through it — the indexing path embeds **passages** (document
chunks) on the write side, the retrieval path embeds the **query** on the read
side — and the two only produce comparable numbers when they run through the
*same* model, version, dimension and normalization. That invariant is the reason
this stage is worth isolating behind a small interface: everything upstream
(chunking) and downstream (pgvector search) depends on the vectors being
mutually comparable, and nothing else in the system can guarantee it.

Because embedding calls are the one place the RAG path touches a paid,
rate-limited, occasionally-flaky network API in a tight loop, this stage also
owns the operational concerns that the rest of the pipeline is free to ignore:
batching, retries, backoff, caching, token accounting and cost.

---

## 1. Target architecture

### 1.1 Responsibilities

A production embedding service does far more than `POST /embeddings`. It owns:

| Responsibility | What it means |
|---|---|
| **Model selection** | Resolve *which* provider/model to use per request (per-user key, KB pin, or config preference) and expose `platform`, `model`, `version`, `dim`. |
| **Batch processing** | Pack many texts into one request — never one HTTP call per chunk. |
| **Token counting** | Count tokens with the model's tokenizer before sending, to size batches and catch over-long inputs. |
| **Truncation protection** | Reject or explicitly (and visibly) truncate inputs over the model's context window — never let the API silently drop the tail of a chunk. |
| **Retries with backoff** | Retry transient failures (5xx, timeouts, connection resets) with exponential backoff + jitter; cap attempts; fail loudly after. |
| **Rate-limit handling** | On `429`, honor `Retry-After`; throttle concurrency; queue rather than hammer. |
| **Caching** | Skip re-embedding identical text. Key = `hash(model_name + version + normalized_text)`. |
| **Vector normalization** | Optionally L2-normalize to unit length so cosine == dot product and scores are stable across providers. |
| **Model-version tracking** | Stamp every stored vector with the exact `provider / model / version / dimension / normalization` that produced it. |
| **Failure reporting** | Surface partial failures to the ingestion job (which chunks failed, why) — never write a half-embedded document as "completed". |
| **Usage / cost tracking** | Aggregate input tokens, request counts and cache-hit rate per owner/model for cost attribution. |

### 1.2 Batch embedding (the load-bearing optimization)

Embedding one chunk per request is the classic RAG performance bug: a 400-chunk
document becomes 400 round-trips, 400x the latency and 400x the rate-limit
pressure. The service **must** accept a list and batch it:

- Group texts into batches bounded by *both* a count cap (payload size) and a
  token cap (model input limit).
- Send each batch as one request; **preserve input order** in the output
  (providers may reorder — sort the response back by its `index`).
- A batch that trips a rate limit is retried as a unit; a batch too large for
  the token budget is split, not dropped.

### 1.3 Input / output contract

The service speaks lists in, lists out, with the provenance attached:

```jsonc
// Request
{
  "model": "mistral/mistral-embed",
  "version": "2026-06",
  "input_type": "passage",              // "query" | "passage"
  "texts": [
    "Refresh tokens rotate on every use.",
    "The access token lives for 15 minutes."
  ]
}

// Response
{
  "model": "mistral/mistral-embed",
  "version": "2026-06",
  "dimension": 1024,
  "normalized": true,
  "embeddings": [
    [ 0.0123, -0.0456,  0.0210, /* … 1024 floats … */ ],
    [-0.0031,  0.0789, -0.0177, /* … 1024 floats … */ ]
  ],
  "usage": { "input_tokens": 21, "requests": 1, "cache_hits": 0 }
}
```

`embeddings[i]` corresponds to `texts[i]`. `dimension` and `model`/`version`
travel with the response so the caller can persist them alongside each vector
(§1.6).

### 1.4 Asymmetric input types — query vs passage

Modern retrieval models are trained **asymmetrically**: a short question and the
long passage that answers it are *deliberately* embedded differently so they
land near each other. The service exposes this as an `input_type`:

- **passage / document** — the stored chunk (write path).
- **query** — the user's question (read path).

Two rules follow:

1. Embed chunks as `passage` and queries as `query`. Getting this backwards
   quietly degrades recall.
2. The type is a *hint to the same model*, **not** a different model. Query and
   passage vectors still share one space and remain directly comparable.

### 1.5 Embedding compatibility rule

> **Never compare vectors produced by different models.**

Cosine similarity between an OpenAI `text-embedding-3-small` vector (1536-dim)
and a Mistral `mistral-embed` vector (1024-dim) is not merely inaccurate — it is
undefined (different dimensionality) or meaningless (different learned geometry
even at equal dimensions). A retrieval result is only trustworthy when the query
vector and every candidate passage vector come from the **same
model + version + normalization**. The corollary drives the whole storage
design: a knowledge base **pins** its embedding space on first ingest, and the
query path reconstructs *that exact* provider before searching.

### 1.6 Embedding-model versioning

Compatibility is enforced by *recording provenance on every vector*. Each stored
embedding carries:

```
embedding_provider   e.g. "mistral"
embedding_model      e.g. "mistral-embed"
embedding_version    e.g. "2026-06"        (bumped when the provider re-trains)
dimension            e.g. 1024
normalization        e.g. "l2" | "none"
```

Why versioning matters: **changing the model changes the dimension, the
distribution and the score scale.** A vector embedded with last quarter's model
is not interchangeable with one from this quarter even if the model *name* is
identical — a silent provider re-train shifts the geometry. So:

- Old and new vectors **must not mix** in one search. A model/version change
  means a **re-embed** (backfill) of the affected collection, not an in-place
  swap.
- The cache key includes `version`, so a version bump naturally invalidates
  stale cached vectors instead of serving them.
- Migrations are gated: search only runs once a collection is *fully* on one
  `(model, version)`; mixed states are quarantined or reindexed.

---

## 2. Current in Nexus AI (baseline)

Everything here is real and verified in
[`backend/app/providers/embeddings.py`](../../app/providers/embeddings.py). The
service today is a clean provider abstraction with correct batching and
asymmetric input types — but **no retries, no cache, and normalization only in
the fallback encoder**.

### 2.1 The `EmbeddingProvider` interface

An ABC that the entire pipeline depends on — never a concrete SDK
(`embeddings.py:38`). Contract:

- `platform` / `model` class attributes identify the vector space
  (`embeddings.py:42-43`).
- `dim` property — learned lazily from the first embed
  (`embeddings.py:48-50`, set in `embed()` at `embeddings.py:68-69`).
- `is_fallback` property — default `False` (`embeddings.py:54-55`).
- abstract `_embed_batch(texts, input_type)` (`embeddings.py:57-59`).
- `embed(texts, input_type)` — batches at `_BATCH`, preserves order
  (`embeddings.py:61-70`); `embed_one(text, input_type)` convenience wrapper
  (`embeddings.py:72-74`).

Batching **is** implemented: `_BATCH = 64` texts per request
(`embeddings.py:34`), and `embed()` walks the input in `_BATCH` slices
(`embeddings.py:66-67`). Asymmetric types exist as the module constants
`INPUT_QUERY = "query"` / `INPUT_PASSAGE = "passage"` (`embeddings.py:30-31`).

### 2.2 The three providers

| Provider | Class | Notes |
|---|---|---|
| Mistral / OpenAI / Vercel / NVIDIA | `OpenAICompatEmbedding` (`embeddings.py:77`) | OpenAI-style `POST {base_url}/embeddings`; **sorts the response by `index`** to restore input order (`embeddings.py:108`); non-200 → `RuntimeError` (`embeddings.py:102-105`). Sends `input_type` only when the model needs it (NVIDIA). |
| Google | `GeminiEmbedding` (`embeddings.py:112`) | `batchEmbedContents` endpoint (`embeddings.py:115`); maps `input_type` → `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` task types (`embeddings.py:124`); default model `text-embedding-004`. |
| _keyless fallback_ | `HashEmbedding` (`embeddings.py:140`) | Deterministic SHA-256 → **768-dim L2-normalized unit vector** (`embeddings.py:158-168`); `is_fallback == True` (`embeddings.py:152-153`). Identical text → identical vector, so the pipeline still runs (and flags low quality) with **no** embedding key. |

`HashEmbedding` is the **only** provider that normalizes its output
(`embeddings.py:167-168`); the real providers return whatever the API sends.

### 2.3 Model selection

Selection is per-user-key, driven by a config preference list:

- `settings.rag_embedding_preference = "mistral,openai,vercel,nvidia,google,hash"`
  (`config.py:68`).
- `resolve_embedding_provider(db, owner_id)` walks that list and returns the
  first provider the user holds a usable key for; `hash` short-circuits to the
  keyless fallback (`embeddings.py:197-229`).
- `_find_key` picks the first **enabled, non-errored** `ApiKey` scoped to the
  owner **or** a global `NULL`-owner (shared) key
  (`embeddings.py:182-194`; `ApiKey` model at
  [`db_models.py:50`](../../app/models/db_models.py)).
- `embedding_provider_for_kb(...)` rebuilds the *exact* provider a KB/document
  is pinned to so queries embed into the stored space
  (`embeddings.py:232-260`) — this is the compatibility rule (§1.5) as code.

Models and dimensions in play:

| Platform | Model | Dim |
|---|---|---|
| mistral | `mistral-embed` | 1024 |
| openai | `text-embedding-3-small` | 1536 |
| vercel | `openai/text-embedding-3-small` | 1536 |
| nvidia | `nvidia/nv-embedqa-e5-v5` | (model-defined) |
| google | `text-embedding-004` | 768 |
| hash | `local-hash-768` | 768 |

(Provider map: `embeddings.py:174-179`; dims cross-checked against the comment
in [`rag_models.py:34-36`](../../app/models/rag_models.py).)

### 2.4 Batching in the ingest path

The provider batches at 64, and the ingestion loop batches *again* at 32 for
progress granularity: `_EMBED_BATCH = 32` in
[`rag_ingestion.py:29`](../../app/services/rag_ingestion.py), embedding chunks in
32-slices with `input_type=INPUT_PASSAGE` (`rag_ingestion.py:110-112`). Because
32 < 64, each ingest slice is exactly one provider request. The query path embeds
a single text with `input_type=INPUT_QUERY`
([`rag_retrieval.py:147`](../../app/services/rag_retrieval.py)).

### 2.5 What is NOT here yet

| Target responsibility | Baseline state |
|---|---|
| Retries / backoff | **None.** A transient 5xx or timeout raises immediately and fails the whole document. |
| Rate-limit handling | **None.** A `429` is a hard error; no `Retry-After`, no throttle. |
| Caching | **None.** Re-ingesting identical text re-embeds and re-pays. |
| Normalization | Only in `HashEmbedding`; real-provider vectors are stored unnormalized. |
| Token counting / truncation guard | **None.** Over-long chunks rely on chunk-size config, not a tokenizer check. |
| Per-chunk model/version | Tracked at **KB / document grain**, not per chunk. `KnowledgeBase.embedding_platform/model/dim` pinned on first ingest (`rag_models.py:60-62`; set in `rag_ingestion.py:138-141`); `Document` mirrors it (`rag_models.py:92-94`). `DocumentChunk` has **no** `embedding_model`/`embedding_version` column yet (`rag_models.py:114-135`). |
| Usage / cost tracking | **None** in the embedding path. |

> **Being added** (per the current implementation effort): retries with
> backoff, an embedding cache keyed on `hash(model + version + normalized_text)`,
> optional normalization, and **per-chunk** `embedding_model` / `embedding_version`
> tracking (see `05-vector-storage.md` §3 for the new chunk columns).

---

## 3. Design decisions

- **Interface over SDK.** The pipeline imports only `EmbeddingProvider`, so a
  provider swap never touches retrieval or ingestion. This is what makes the
  keyless `HashEmbedding` fallback possible — and what will make retries/caching
  a wrapper concern rather than a rewrite.
- **Auto-detect, then pin.** No provider is configured up front; the first
  usable key wins by preference order, and the resulting `(platform, model, dim)`
  is *frozen onto the KB*. This trades a little startup magic for the
  compatibility guarantee — a KB can never silently change vector spaces
  mid-life.
- **Keyless fallback is a feature, not a bug.** `HashEmbedding` keeps the whole
  ingest→store→retrieve loop exercisable with zero API keys (dev, CI, demos).
  It is deliberately marked `is_fallback` so the UI/job status can warn that
  retrieval quality is poor.
- **Two levels of batching.** Provider-level `_BATCH=64` bounds payload size;
  ingest-level `_EMBED_BATCH=32` bounds *progress* granularity so the job bar
  moves smoothly. They compose cleanly because 32 ≤ 64.
- **Cache/retry/normalize as an additive wrapper.** Because provenance already
  travels with the KB pin, adding a cache keyed on `hash(model+version+text)` and
  a retrying `embed()` decorator does not require changing any stored data —
  only hardening the one choke point.

---

## 4. Pitfalls

- **Mixing embedding models.** Comparing vectors across models/versions/dims is
  the cardinal sin (§1.5). The KB pin defends the write side; the risk lives on
  the read side and during model migrations — *always re-embed, never swap
  in place.*
- **Silent provider re-trains.** A model keeping its *name* while its weights
  change shifts the geometry. Without an `embedding_version` stamp, mixed-vintage
  vectors coexist undetected and quietly rot recall. (This is exactly the
  per-chunk versioning gap in §2.5.)
- **No retries = brittle ingest.** Today a single transient 429/5xx aborts an
  entire document (`_embed_batch` raises straight through). Large uploads over a
  rate-limited free-tier key are the most exposed.
- **Unnormalized real-provider vectors.** Cosine distance still works, but score
  magnitudes are not comparable across providers and thresholds tuned on one
  provider mislead on another. Optional normalization closes this.
- **No token guard.** Chunk sizing is char-based (`~4 chars/token`), not
  tokenizer-based; an unusually dense chunk can exceed a model's input window and
  be **silently truncated by the API**, embedding only its head.
- **Forgetting `input_type`.** Embedding a query as a passage (or vice-versa) on
  an asymmetric model looks like it works and quietly loses recall — no error,
  just worse results.
- **Re-embedding cost.** With no cache, every re-ingest of unchanged content pays
  again in tokens and latency; at scale this dominates the ingest bill.
