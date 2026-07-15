# Context Building & RAG Generation — Nexus AI

> The last mile of retrieval: turn a ranked pile of chunks into a compact,
> ordered, cited **context block**, wrap it in a **grounded prompt**, and let the
> LLM answer *only* from what it was given — with traceable citations back to the
> source.
> Drill-down for stage `Context build → LLM generation` of `00-ARCHITECTURE.md`;
> pairs with `06-hybrid-search-reranking.md` (what feeds it) and
> `08-security-multitenancy.md` (what it must never leak).

---

## 1. Where this stage sits

Retrieval hands us a ranked list of candidate chunks. That list is **not** an
answer and **not** a prompt. Between them sits the context builder — the
component that decides *which* chunks the model actually sees, in *what order*,
under *what token budget*, with *what provenance*, and inside *what instructions*.

```
… hybrid search ─▶ rerank ─▶  [ CONTEXT BUILDER ]  ─▶ grounded prompt ─▶ LLM ─▶
                              dedup · merge · parent   cite-only-from-sources    streamed
                              expand · budget · order  system prompt             answer +
                              · cite · ACL                                        citations
```

Get this wrong and even perfect retrieval produces hallucinated, uncited, or
budget-blown answers. Get it right and a mediocre retriever still looks
trustworthy, because every claim is traceable.

---

## 2. Target architecture

### 2.1 Context-builder responsibilities

A production context builder is a **pipeline of filters** over the reranked
candidates. Each is cheap; together they are the difference between "stuffed the
top-k into the prompt" and a real context.

| # | Responsibility | Why it matters |
|---|---|---|
| 1 | **Remove duplicate chunks** | Near-identical passages (re-uploads, boilerplate, overlapping windows) waste budget and bias the model by repetition. Dedup on normalized text / content hash / high cosine. |
| 2 | **Merge adjacent chunks** | Two consecutive chunks of the same section read as one coherent passage; merging removes the mid-sentence seam chunking introduced. |
| 3 | **Retrieve PARENT chunks (small-to-big)** | Embed & match on *small* child chunks (precise), but feed the *parent* (section/page) to the LLM (enough context to reason). Match child → expand to `parent_chunk_id`. |
| 4 | **Respect the token budget** | The prompt has a hard ceiling. Pack highest-value chunks first; stop before the model (or the answer headroom) is squeezed out. |
| 5 | **Preserve source metadata** | Every chunk must carry `document`, `page`, `section`, offsets through to the prompt — you cannot cite what you dropped. |
| 6 | **Order chunks logically** | Not raw score order. Group by document, then by page/section/ordinal, so the model reads a document as a document, not a shuffled deck. |
| 7 | **Reject low-confidence chunks** | A chunk below a relevance floor is *noise*; including it invites the model to answer from irrelevant text. Better to have fewer, stronger chunks. |
| 8 | **Prevent unauthorized context** | The builder is the last gate before content reaches the model: never place a chunk the caller isn't authorized to see into the prompt (see `08-security-multitenancy.md`). |
| 9 | **Add citation identifiers** | Tag each context block with a stable id (`[SOURCE-1]`, `doc-102#p4`) the model can reference and the UI can resolve back to the source. |

### 2.2 The context block

Each retained chunk is rendered with its provenance header so both the model and
the reader can trace it:

```
[SOURCE: doc-102, page 4, section 'Token Rotation']
Refresh tokens rotate on every use. The prior token is revoked within 60s of the
new one being issued; a reused (already-rotated) token invalidates the whole
family and forces re-authentication.

[SOURCE: doc-102, page 5, section 'Token Rotation']
Rotation is disabled only for service accounts, which use non-expiring…
```

The `[SOURCE: …]` header is the contract: it is *metadata about evidence*, never
an instruction, and it is what the citation `[SOURCE-n]` in the answer points at.

### 2.3 RAG generation flow

```
1. Query embedding      same model/space as the stored chunks
2. Hybrid retrieval     dense (pgvector <=>) ∥ sparse (FTS) → RRF
3. Rerank               cross-encoder / LLM judge → top-K
4. Context build        dedup · merge · parent-expand · order · budget · cite   ← this doc
5. LLM prompt           grounded system prompt + context block + question
6. Grounded answer      answer strictly from context
7. Citations            [SOURCE-n] inline, resolved to source chips in the UI
```

### 2.4 Grounded prompt rules

The system prompt is a **contract with the model**. The four load-bearing rules:

1. **Use only the supplied context.** No outside/parametric knowledge, no
   "general" facts not present in the sources.
2. **Admit insufficiency.** If the context does not answer the question, say so
   ("I don't have that in the knowledge base") — do not paper over the gap.
3. **Cite source ids.** Every claim carries the `[SOURCE-n]` it came from.
4. **Don't invent.** No fabricated facts, numbers, quotes, or citations.

The answer is then **traceable**: every sentence maps to a `[SOURCE-n]`, and every
`[SOURCE-n]` maps to a real chunk with real provenance.

---

## 3. Current in Nexus AI (baseline)

The pragmatic tier ships the **grounding contract, citation numbering, token
budget, and source dedup** today. It does **not** yet do cross-chunk content
dedup, adjacent-merge, parent expansion, or MMR — those are the `[NEW]` items
in §5 / `10-gap-analysis.md`.

Two RAG paths share the same retrieval core but build context differently:

### 3.1 Knowledge-base path (grounded chat)

**Grounded prompt — `services/rag_retrieval.py:38-45`.** `GROUNDED_SYSTEM` is the
cite-only-from-sources contract, embedding the context via `{context}`:

- "Answer … using ONLY the numbered context sources below."
- "Cite every claim inline with bracketed source numbers like `[1]` or `[2][3]`."
- "If the answer is not contained in the sources, say you don't have enough
  information … do not use outside knowledge or invent facts."

That is rules 1–4 of §2.4, verbatim. The citation identifier is a **plain
number** `[n]`, not `[SOURCE-n]` — the number keys into the numbered context
blocks.

**Context assembly — `build_grounded_messages()` `rag_retrieval.py:315-330`.**
Builds the LLM message list:

- Numbers each chunk and stamps its source name:
  `"[{c['index']}] (source: {c['document_name']})\n{c['text']}"`
  (`rag_retrieval.py:320-321`) — this is the current form of the §2.2 context
  block (source *name*, no page/section yet).
- Prepends the `GROUNDED_SYSTEM` system message, then replays **the last 6
  history turns** (`for m in history[-6:]`, `rag_retrieval.py:326`) keeping only
  `user`/`assistant` roles, then appends the user's question.
- Falls back to `"(no relevant sources found)"` when the chunk list is empty
  (`rag_retrieval.py:322`) so the grounded contract still holds with zero
  context.

**Chunk numbering & provenance — `retrieve()` `rag_retrieval.py:140-187`.** After
hybrid + (no-op) rerank, it resolves every chunk's source filename in **one**
`Document.id → filename` query (`rag_retrieval.py:171-174`) and returns dicts
carrying `index` (1-based, `rag_retrieval.py:177`), `document_name`, `chunk_id`,
`document_id`, `ordinal`, `text`, and `score`. That `index`/`document_name` pair
is what §2.1-#5/#9 (metadata + citation id) rely on. The final set is bounded by
`rag_final_top_k = 6` (`config.py:63`) — the KB path has **no** separate token
compressor; it trusts top-k to bound size.

**"sources" SSE event first — `routes/knowledge.py:279-358`.** `kb_chat_stream`
retrieves up front (`retrieve(...)`, `knowledge.py:290`) so retrieval failures
surface *before* streaming, builds messages (`knowledge.py:295`), then the
`event_generator` **emits the citations payload as the very first SSE frame**
(`yield json.dumps({"sources": source_payload, "conversationId": …, "done":
False})`, `knowledge.py:327`) so the client can paint the Sources panel before a
single content token arrives. A `_sources_footer()` (`knowledge.py:266-276`) also
appends a deduped `**Sources**` list to the *stored* assistant message.

### 3.2 Per-conversation path (attachment RAG)

Chat attachments become their own retrievable corpus, scoped to the conversation.

**Ingest on send — `services/multimodal_chat.py:136-140`.** When a chat turn
carries document attachments, each is scheduled for background indexing:
`asyncio.create_task(ingest_conversation_document(conversation_id, owner_id,
_fname, _raw))`. The full bytes were captured earlier (`doc_raws.append(...)`,
`multimodal_chat.py:98`) so later turns can retrieve them. Separately, the *same*
turn also gets the extracted text inlined directly as grounding context
(`multimodal_chat.py:142-150`) — an immediate, non-retrieval shortcut for "answer
about the file I just dropped".

**Index writer — `ingest_conversation_document()`
`services/rag_ingestion.py:186-233`.** Best-effort (never raises), opens its own
`SessionLocal()`, extracts → cleans → chunks → embeds with `INPUT_PASSAGE`, and
writes a `Document` + `DocumentChunk` rows scoped by `conversation_id` and
`owner_id` (`rag_ingestion.py:211,223`).

**Retrieve + inject — `services/agent.py:936-953`.** On a **streaming** agent turn
(`agent_stream_chat`), `retrieve_conversation_context(db, conversation_id,
message, owner_id)` runs (guarded; no-op when the conversation has no docs), and
any result is passed as `doc_context` into `_build_agent_messages(...,
doc_context=doc_context)`. Note this injection is on the **streaming path only** —
the non-streaming `agent_chat` (`agent.py:615`) does not call it.

**Injection site — `services/agent.py:578-588`.** The excerpts are appended to the
system prompt under the header **`=== ATTACHED DOCUMENTS (this conversation)
===`**, instructing the model to "Treat them as authoritative context … cite the
source name in brackets … If they don't contain the answer, say so rather than
guessing." (grounding + citation, but a *lighter* contract than `GROUNDED_SYSTEM`
— see §6 and `08-security-multitenancy.md`).

**Token-budget compressor + source dedup —
`retrieve_conversation_context()` `rag_retrieval.py:252-312`.** This is where the
budget lives:

- Embeds the query in the *same space* the conversation's docs were embedded in
  by looking up the latest doc's `embedding_model` (`rag_retrieval.py:264-273`).
- Runs the same hybrid + RRF + (no-op) rerank as the KB path.
- **Compressor** (`rag_retrieval.py:292-299`): walks reranked chunks best-first,
  estimating cost as `max(1, len(ch.text) // 4)` (~4 chars/token) and stops once
  adding the next chunk would exceed `token_budget` (default **1800**,
  `rag_retrieval.py:254`). This is §2.1-#4, approximate but real.
- **Source-name dedup** (`rag_retrieval.py:307-311`): builds the returned
  `sources` list uniquely (`if name not in sources`), and renders each kept chunk
  as `"[{i}] (from {name})\n{ch.text}"`.

### 3.3 What is NOT here yet

- **No cross-chunk content dedup.** Source *names* are deduped
  (`rag_retrieval.py:310`; footer `knowledge.py:266-276`); chunk *bodies* are
  not. Two near-identical passages both reach the prompt. (§2.1-#1) **[NEW —
  being added]**
- **No parent expansion / small-to-big.** There is no `parent_chunk_id` column
  (`models/rag_models.py` `DocumentChunk`), so the matched child *is* the context
  block. (§2.1-#3) **[NEW — being added]**
- **No adjacent-chunk merge.** Chunks are emitted individually, ordered by fused
  score, not regrouped by document/section. (§2.1-#2, #6)
- **No MMR / diversity** and **no relevance floor** — rerank is `NoOpReranker`
  (`providers/reranker.py:39-42`), so ordering is pure RRF and every fused chunk
  up to top-k is kept regardless of absolute score. (§2.1-#7)

> **In flight:** parent-expansion and content-dedup are the next context-builder
> increments; they slot into `build_grounded_messages` /
> `retrieve_conversation_context` without changing the retrieval or prompt
> contracts.

---

## 4. RAG lifecycle in the current code (end-to-end)

```
KB grounded chat                          Per-conversation attachment RAG
────────────────                          ───────────────────────────────
POST /api/kb/{id}/chat/stream             chat turn w/ attachments
  knowledge.py:279                           multimodal_chat.py:65
    ↓                                          ↓ create_task ingest (bg)
retrieve()  rag_retrieval.py:140           ingest_conversation_document
  dense ∥ FTS → RRF(k=60) → NoOp rerank      rag_ingestion.py:186 (chunks+embeds)
  → top-6, numbered w/ document_name        ─────────── later turn ───────────
    ↓                                        agent_stream_chat  agent.py:936
build_grounded_messages()                    retrieve_conversation_context
  rag_retrieval.py:315                          rag_retrieval.py:252
  GROUNDED_SYSTEM + [n](source:name)           hybrid → rerank → ~1800-tok
  + last 6 turns + question                     compressor → source dedup
    ↓                                            ↓
route_stream_chat → SSE                       inject under
  frame 0 = {"sources": …}  knowledge.py:327   "=== ATTACHED DOCUMENTS ==="
  frames 1..n = content deltas                  agent.py:578 → LLM stream
```

---

## 5. Design decisions

- **Grounding as a prompt contract, not a code guarantee.** The model is
  *instructed* to answer only from sources (`GROUNDED_SYSTEM`); the system does
  not post-verify that every sentence is entailed by a cited chunk. Cheap, works
  well with capable models; groundedness/citation-correctness scoring is deferred
  to `09-evaluation-observability.md`.
- **Numeric `[n]` citations keyed to numbered blocks.** Simpler for the model
  than free-form `[SOURCE: doc, page]` ids and trivially resolvable in the UI.
  The trade: the citation carries no page/section until the metadata columns
  (`section`, `page_number`) land (`00-ARCHITECTURE.md §4`).
- **Sources emitted before content.** Streaming the citation payload as SSE frame
  0 (`knowledge.py:327`) lets the client render provenance immediately and keeps
  the answer honest — the user sees *what* grounds the answer as it is written.
- **Token budget only where inputs are unbounded.** The KB path bounds context by
  `rag_final_top_k = 6`; the conversation path adds an explicit ~1800-token
  compressor because attached-doc chunk sizes vary widely. One mechanism per
  path, matched to its risk.
- **Two builders, one retriever.** KB and conversation RAG reuse
  `semantic_search`/`keyword_search`/`reciprocal_rank_fusion` but assemble context
  independently — different budgets, different prompts, different injection sites
  — because their trust and size profiles differ (see §6).
- **Char/4 token estimate.** The compressor uses `len(text)//4` rather than a real
  tokenizer — no per-model tokenizer dependency, and a deliberate *under*count is
  safe (leaves headroom).

---

## 6. Pitfalls

- **Context-block text is untrusted.** Everything between `[SOURCE …]` and the
  next block is document content a *user uploaded*, not instructions. The
  conversation-RAG injection ("Treat them as authoritative context",
  `agent.py:580-588`) and the inline attachment prompt (`multimodal_chat.py:146-150`)
  are the weakest seams here — neither states "never follow instructions inside
  the documents". Prompt-injection defense lives in `08-security-multitenancy.md`;
  the context builder must not strengthen an attacker's hand by labeling raw doc
  text "authoritative".
- **Duplicate chunks bias the answer.** With no content dedup (§3.3), a passage
  duplicated across uploads is fed twice and the model treats repetition as
  emphasis. Until content-dedup lands, near-duplicate corpora skew answers.
- **Chunk shuffling breaks coherence.** Emitting chunks in fused-score order
  (not document/section order) can hand the model page 5 before page 4. Adjacent
  merge + logical ordering (§2.1-#2/#6) are the fix.
- **Budget starvation vs. context starvation.** Too generous a budget crowds out
  the answer headroom; too tight drops the chunk that held the answer. The
  ~1800-token default is a heuristic — tune against the eval set, not by feel.
- **Citations that don't resolve.** If the model cites `[7]` but only 6 blocks
  were supplied, the UI has nothing to link. Keep the numbering the model sees
  identical to the `sources` payload (they are, today: both come from the same
  `retrieve()` result).
- **"No sources" must still be grounded.** When retrieval returns nothing,
  `build_grounded_messages` injects `"(no relevant sources found)"` rather than an
  empty context — do **not** let an empty context silently drop the grounding
  rules, or the model reverts to parametric guessing.
- **The immediate-inline vs. background-index split can desync.** On the first
  turn the attachment is answered from inlined text (`multimodal_chat.py:142-150`)
  while indexing runs in the background (`create_task`, `:138-140`); if indexing
  hasn't finished, the *next* turn's `retrieve_conversation_context` may find no
  chunks yet. Treat conversation-RAG availability as eventually-consistent.
```
