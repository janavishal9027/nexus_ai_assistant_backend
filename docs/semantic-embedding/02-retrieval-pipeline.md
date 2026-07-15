# Retrieval Pipeline (read path) — Nexus AI

> The **read path** of the semantic-embedding subsystem: how a user's question
> becomes a grounded, cited answer.
> Companion to `00-ARCHITECTURE.md` (§2.2). Hybrid search and reranking — the
> middle two stages — get their own deep dive in `06-hybrid-search-reranking.md`;
> context assembly and the grounded prompt are covered in `07-context-building-rag.md`.

---

## 0. Intro — what "retrieval" means here

Indexing (the write path, `01-indexing-pipeline.md`) turns documents into
embedded, searchable chunks. Retrieval is the mirror pipeline that runs on
**every question**: it takes an ambiguous, conversational query and turns it into
a small, ordered set of evidence the LLM is allowed to answer from.

The whole pipeline is one long funnel — each stage narrows the candidate set and
sharpens relevance:

```
User query
   │  "and how do I refresh it?"
   ▼
Query understanding ── normalize, classify intent, rewrite to standalone   [NEW]
   │                    "How do I refresh a JWT access token in Nexus AI?"
   ▼
Query embedding ── same model/space as the stored chunks
   │               vec ∈ ℝ^d   (d = whatever the KB was pinned to)
   ▼
Hybrid retrieval ── dense (pgvector <=>) ∥ sparse (FTS ts_rank) → RRF(k=60)
   │               owner-scoped, top-N candidates                (06-*)
   ▼
Rerank ── cross-encoder / LLM judge on the top-N                 [REAL, was NoOp]
   │
   ▼
Context build ── dedup, order, token budget, number the sources  (07-*)
   │
   ▼
LLM generation ── grounded, cite-only-from-sources system prompt
   │
   ▼
Grounded answer ── streamed deltas + Sources/citation chips
```

The guiding invariant: **the model may only answer from what retrieval hands it.**
If retrieval misses a relevant chunk, no amount of prompt engineering recovers it —
so the earlier funnel stages (understanding, hybrid recall) matter more than the
generation prompt.

---

## 1. Target architecture

### 1.1 Query preprocessing (normalize, but keep technical tokens)

Normalize the surface form **without destroying meaning**:

- Unicode NFKC normalization, collapse whitespace, strip control characters.
- Case-fold only for the sparse/keyword branch — the dense branch keeps original
  case because embedding models are case-sensitive.
- **Preserve technical tokens verbatim.** Naive tokenizers mangle exactly the
  tokens that carry the most retrieval signal:

  | Token | Naive normalizer does | Correct behavior |
  |---|---|---|
  | `C++` / `C#` | strips `+`/`#` → `c` | keep punctuation |
  | `.NET` | drops leading `.` → `net` | keep as `.NET` |
  | `gemini-embedding-001` | splits on `-` | keep as one token |
  | `/api/auth/refresh` | splits into stopwords | keep the path intact |
  | `CVE-2026-12345` | splits into `cve 2026 12345` | keep the identifier |

  A "cleanup" that discards `+`, `#`, `.`, `/`, `-` turns a precise identifier into
  noise. Preprocessing should be conservative — when unsure, leave the token alone.

### 1.2 Intent classification

Route the query by what the user actually wants, so downstream stages can adapt
(retrieval depth, whether to hit the KB at all, whether to call a tool). Target
taxonomy:

| Intent | Example | Retrieval implication |
|---|---|---|
| `factual` | "What port does the API bind to?" | tight top-K, prefer exact hits |
| `search` | "Show me everything about refresh tokens" | wide fan-out, more sources |
| `summarize` | "Summarize the auth design" | broad recall, parent expansion |
| `compare` | "Difference between access and refresh tokens?" | retrieve both sides |
| `code` | "How is `route_stream_chat` wired?" | keyword-weighted, code-aware |
| `personal-memory` | "What did I say my deadline was?" | conversation memory, not KB |
| `current-info` | "Latest CVE for this library?" | web/tool, KB likely stale |
| `tool-action` | "Create a task for this" | agent tool call, skip RAG |

Intent is a **hint, not a gate** — misclassification should degrade gracefully to
"retrieve and let the model decide," never hard-fail a valid question.

### 1.3 Query rewriting to a standalone form

Conversational queries are riddled with references that only resolve against
history — "it", "that", "the second one", "and the mobile version?". A retriever
embeds the query in isolation, so `"and how do I refresh it?"` embeds nothing
useful. Rewriting resolves the query against the last few turns into a
**self-contained** question:

```
history:  user: "How do JWT access tokens work in Nexus AI?"
          asst: "They're short-lived bearer tokens signed with …"
query:    "and how do I refresh it?"
          ▼ rewrite
standalone: "How do I refresh a JWT access token in Nexus AI?"
```

Two hard rules:

1. **Store both** `original_query` and `rewritten_query`. The rewrite is used for
   retrieval and logging; the *original* is what the user sees and what the LLM
   answers, so intent attribution stays honest.
2. **Never change the user's intent.** Rewriting resolves references and adds
   implied context — it does not answer, editorialize, narrow, or "improve" the
   question. A rewrite that turns "refresh it" into "revoke it" is a bug.

### 1.4 Candidate retrieval (recall-first)

Retrieve **more than you keep**. Retrieval is a two-phase recall/precision split:

- **Recall phase:** fan out wide — `top_k ≈ 30` candidates from hybrid search
  (dense ∥ sparse, fused). The goal is "the right chunk is *somewhere* in here."
- **Precision phase:** rerank and **keep 5–10**. The goal is "the right chunk is
  now near the top, within the LLM's token budget."

Retrieving 30 and reranking down to 6 is far better than retrieving 6 directly:
the cheap first-stage retriever is optimized for recall, the expensive reranker
for precision, and you only pay the reranker on a small set.

### 1.5 Embedding compatibility rule (non-negotiable)

**Query and document vectors must come from the same model and version.** Cosine
distance between a `mistral-embed` document vector and a `text-embedding-3-small`
query vector is meaningless — different models place "meaning" in different
coordinate systems. The rule:

- A KB **pins** its embedding model/version at first ingest.
- The query path **rebuilds that exact provider** to embed the question.
- Use asymmetric `input_type` hints where the model supports them
  (`query` vs `passage` / `RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`) — same model,
  correct role.
- Re-embedding the corpus is the *only* safe way to change models. Never mix
  spaces in one index.

---

## 2. Current in Nexus AI (baseline)

The read path is implemented end-to-end in
`backend/app/services/rag_retrieval.py` and exposed by
`backend/app/routes/knowledge.py`. Two entry points share the same
retrieve→fuse→rerank→build machinery:

| Path | Function | Scope |
|---|---|---|
| KB grounded chat / search | `retrieve()` — `rag_retrieval.py:140` | `knowledge_base_id` |
| Per-conversation attachment RAG | `retrieve_conversation_context()` — `rag_retrieval.py:252` | `conversation_id` |

### 2.1 What runs today (and in what order)

`retrieve()` (`rag_retrieval.py:140-187`):

1. Rebuild the KB's pinned embedding provider —
   `embedding_provider_for_kb(...)` at `rag_retrieval.py:144`.
2. Embed the **raw** query — `provider.embed_one(query, input_type=INPUT_QUERY)`
   at `rag_retrieval.py:147`.
3. Dense + sparse search, both on one worker thread —
   `_search_both()` / `asyncio.to_thread(...)` at `rag_retrieval.py:152-158`.
4. Reciprocal Rank Fusion — `rag_retrieval.py:160`.
5. Rerank (currently a no-op) — `resolve_reranker(...)` / `reranker.rerank(...)`
   at `rag_retrieval.py:165-167`.
6. Resolve source filenames and return chunk dicts with citation metadata —
   `rag_retrieval.py:170-187`.

The grounded prompt is assembled by `build_grounded_messages()`
(`rag_retrieval.py:315-330`) using the `GROUNDED_SYSTEM` template
(`rag_retrieval.py:38-45`), which instructs the model to answer **only** from the
numbered sources and cite inline with `[1]`, `[2]`.

### 2.2 No query understanding yet — the raw message is embedded as-is

This is the biggest gap between target and baseline. There is **no query
preprocessing, no intent classification, and no query rewriting** in the current
code:

- The raw user string is embedded unchanged for the dense branch
  (`rag_retrieval.py:147`).
- The same raw string is handed to the keyword branch verbatim —
  `keyword_search(db, kb.id, query, ...)` at `rag_retrieval.py:155`.
- There is no `original_query` / `rewritten_query` split; there is only `query`.

Concretely: a follow-up like *"and how do I refresh it?"* is embedded and
full-text-searched literally, so the pronoun "it" contributes nothing and recall
suffers. `services/rag_query.py` **does not exist yet** — the query-understanding
stage is marked `[NEW]` in `00-ARCHITECTURE.md` §3 and is being added.

### 2.3 Embedding provider is chosen per user key

Providers are resolved against the *owner's* stored API keys, never hard-coded:

- `resolve_embedding_provider(db, owner_id)` (`embeddings.py:197-229`) walks
  `settings.rag_embedding_preference` (`config.py:68`, default
  `"mistral,openai,vercel,nvidia,google,hash"`) and returns the first provider the
  owner holds an enabled, non-errored key for. `_find_key()`
  (`embeddings.py:182-194`) scopes to the owner's own keys **plus** shared
  NULL-owner keys.
- `embedding_provider_for_kb(db, owner_id, platform, model, dim)`
  (`embeddings.py:232-260`) rebuilds the **exact** provider a KB/document was
  pinned to, so the query embeds into the same space (§2.5). It falls back to
  auto-detection only if the pinned key has since disappeared.
- Asymmetric hints are honored: `GeminiEmbedding` maps `INPUT_QUERY` →
  `RETRIEVAL_QUERY` and passages → `RETRIEVAL_DOCUMENT`
  (`embeddings.py:124`); OpenAI-compatible providers send `input_type` when the
  model needs it (`embeddings.py:92-94`, e.g. NVIDIA in `_OPENAI_COMPAT_EMBED`,
  `embeddings.py:174-179`). A keyless `HashEmbedding` is the last-resort fallback
  so ingestion/retrieval still runs without a key (low quality, flagged).

### 2.4 SSE grounded chat endpoint

`POST /api/kb/{id}/chat/stream` → `kb_chat_stream()`
(`knowledge.py:279-358`):

1. Retrieval runs **up front**, before any streaming, so a retrieval failure
   surfaces as a clean `503` rather than mid-stream — `retrieve(...)` at
   `knowledge.py:290`.
2. `build_grounded_messages(query, sources, body.history)` at `knowledge.py:295`.
3. The SSE `event_generator()` (`knowledge.py:323-356`) emits, in order:
   - **first event = sources** — `{"sources": ..., "conversationId": ..., "done": false}`
     at `knowledge.py:327-328`, so the client renders the source panel before any
     text arrives;
   - then **content deltas** — `{"content": chunk, ...}` per streamed token
     (`knowledge.py:331-334`);
   - a terminal `{"done": true}` event (`knowledge.py:347-350`).
   Delivered via `EventSourceResponse` (`knowledge.py:358`).

The non-streaming preview endpoint `POST /api/kb/{id}/search` → `search_kb()`
(`knowledge.py:250-261`) calls the same `retrieve()` and returns the ranked
sources directly (useful for debugging recall).

### 2.5 Compatibility rule — how it's enforced today

The write and read paths are already symmetric:

- **Ingest** embeds chunks with `INPUT_PASSAGE` (`rag_ingestion.py:112`,
  `205-206`) and **pins** the model on the KB the first time a KB is populated —
  `kb.embedding_platform/model/dim` set at `rag_ingestion.py:138-141`
  (conversation docs record the same on the `Document` row,
  `rag_ingestion.py:214-215`).
- **Retrieve** reads those pinned fields back through `embedding_provider_for_kb`
  and embeds the query with `INPUT_QUERY`. Same model, same space, correct role.
- The KB's pinned fields live on `KnowledgeBase.embedding_platform/model/dim`
  (`rag_models.py:60-62`); the chunk vector column is an unbounded pgvector
  `Vector()` (`rag_models.py:37`, `131`) so any model's dimension fits without a
  migration.

### 2.6 Being added by the implementation

- `services/rag_query.py` **[NEW]** — normalize + intent + standalone rewrite
  (§1.1–1.3). Does not exist yet.
- `providers/reranker.py` **[REAL]** — the interface is wired in
  (`rag_retrieval.py:165-167`) but `resolve_reranker()` returns a `NoOpReranker`
  today (`reranker.py:39-42`); a real cross-encoder / LLM judge is being
  implemented. See `06-hybrid-search-reranking.md`.

---

## 3. Design decisions / how it works

- **Retrieve-then-generate, never generate-then-check.** Grounding context is
  fetched *before* the LLM is called (`knowledge.py:288-295`). The model receives
  only the numbered sources and a system prompt forbidding outside knowledge
  (`rag_retrieval.py:38-45`). This is what makes citations trustworthy.

- **Original query answered, rewritten query retrieved (target).** The rewrite
  exists to help the *retriever*; the user's own words drive the *answer*. Keeping
  both (`original_query` + `rewritten_query`) preserves intent and gives clean
  logs. The baseline has neither yet — it uses the one raw string for both, which
  is the correctness gap §2.2 closes.

- **Recall first, precision second.** Fan out wide (target `top_k≈30`; baseline
  `rag_semantic_top_n=20` + `rag_keyword_top_n=20`, `config.py:60-61`), fuse to a
  short list (`rag_fusion_top_k=10`, `config.py:62`), then rerank to the final few
  (`rag_final_top_k=6`, `config.py:63`). Cheap retriever optimizes recall,
  expensive reranker optimizes precision, and the reranker only ever sees a small
  set.

- **Per-user-key provider resolution.** Nexus AI has no house embedding key; every
  user brings their own. Resolving the provider from owner-scoped keys
  (`embeddings.py:182-229`) and pinning the KB to whatever was used first
  (`rag_ingestion.py:138-141`) is what lets the compatibility rule (§1.5) hold
  without central configuration.

- **Sources-first SSE.** Emitting citations as the first event
  (`knowledge.py:327-328`) lets the UI paint the source panel immediately, so the
  answer streams in against visible provenance rather than appearing ungrounded
  until a trailing footer arrives.

- **Owner scope is a SQL predicate, not a post-filter.** Both retrievers filter by
  `knowledge_base_id` / `conversation_id` inside the query
  (`rag_retrieval.py:59`, `101`, `199`, `231`); the endpoint resolves the KB
  through `_owned_kb()` first (`knowledge.py:41-45`). Authorization happens at
  retrieval, never after — see `08-security-multitenancy.md`.

---

## 4. Pitfalls

- **Embedding the raw conversational query (current baseline).** Pronouns and
  ellipsis ("it", "that one", "and the mobile version?") carry no retrieval signal
  once embedded in isolation. This is the live gap at `rag_retrieval.py:147,155` —
  until `rag_query.py` lands, follow-up questions retrieve poorly.

- **Rewriting that changes intent.** An over-eager rewrite that answers the
  question, narrows scope, or swaps a verb ("refresh" → "revoke") corrupts every
  downstream stage. Rewrite resolves references and adds implied context — nothing
  more. Always keep the original.

- **Over-normalizing technical tokens.** Stripping `+`, `#`, `.`, `/`, `-` destroys
  exactly the high-signal identifiers users search for (`C++`, `.NET`,
  `/api/auth/refresh`, `CVE-2026-12345`, `gemini-embedding-001`). The keyword
  branch is where these matter most — see `06-hybrid-search-reranking.md` §1.

- **Mixing embedding spaces.** Querying a KB pinned to model A with a vector from
  model B yields silent garbage — no error, just bad ranking. The pinned fields
  (`rag_models.py:60-62`) and `embedding_provider_for_kb` (`embeddings.py:232-260`)
  guard this only if the pinned key still exists; if it vanishes and the code falls
  back to auto-detection, the query can silently land in a *different* space than
  the corpus. Prefer failing loudly over falling back across models.

- **Retrieving as many as you keep.** If `top_k` equals the final context size, the
  reranker has nothing to reorder and recall misses are permanent. Keep the
  recall/precision fan-out (30 → 6), not 6 → 6.

- **Trusting the no-op reranker's order as "relevance."** Today `reranker.rerank`
  just truncates the RRF order (`reranker.py:35-36`), so "top 6" means "top 6 by
  fusion rank," not "top 6 by deep relevance." Don't read more into the ordering
  than the current stage provides until the real reranker ships.

- **Streaming before grounding is ready.** Retrieval must complete before the first
  token (`knowledge.py:288-295`); interleaving them would let the model start
  answering ungrounded and cite sources that arrive later. Keep retrieval strictly
  ahead of generation.

- **Session-per-thread safety.** Both searches run **sequentially on one** worker
  thread (`asyncio.to_thread`, `rag_retrieval.py:152-158`) precisely because a
  SQLAlchemy `Session` is not safe for concurrent use. Do not "optimize" this into
  two parallel threads sharing the session (see `06-hybrid-search-reranking.md` §2).
