# Indexing Pipeline — Nexus AI

> The **write path** of the semantic‑embedding subsystem: how a raw file becomes
> embedded, searchable chunks. Read `00-ARCHITECTURE.md` first for the whole
> system; this file drills into the left‑hand pipeline
> (`Source → Ingest → Parse/Clean → Dedup → Chunk → Embed → Store`) and
> `03-chunking.md` drills into the chunk stage specifically.
>
> Written in the house style: a **Target architecture** (the scale‑out ideal) and
> the **Current in Nexus AI (baseline)** — what actually ships on the
> single‑Postgres stack, with verifiable `file:line` references.

---

## 1. What this stage is responsible for

Indexing is a one‑way transform run once per document (and again on re‑ingest):

```
   bytes in                                                         rows out
  ─────────                                                        ──────────
  a PDF / DOCX / .md / .py / chat attachment
        │
        ▼
   parse → clean → dedup → chunk → embed  ───────────────▶  document_chunks
                                                            (text + pgvector + FTS)
```

Everything downstream (retrieval, reranking, grounded answers) can only be as good
as what this stage stores. Its contract: **every chunk that lands in
`document_chunks` is clean, bounded in size, embedded in the KB's one vector space,
and attributable to its source document.**

---

## 2. Target architecture (event‑driven ideal)

The scale‑out target is an **event‑driven pipeline** where each stage is an
independent consumer reacting to the previous stage's event. Nothing blocks the
upload request; every transition is a durable, replayable message.

```
 Client ─upload─▶ API ──┐
                        │  put bytes                     ┌──────────────┐
                        ├───────────────────────────────▶│ MinIO / S3   │  object store
                        │                                └──────────────┘
                        ▼ emit
             ┌────────────────────── Kafka topics ──────────────────────┐
             │                                                           │
   document.uploaded ─▶ document.parsed ─▶ document.chunked ─▶ document.embedded ─▶ document.indexed
             │                │                  │                  │                  │
        [parser]         [cleaner]           [chunker]          [embedder]         [indexer]
             │                │                  │                  │                  │
             └────────────── on error, N retries with backoff ──────┘
                                     │ exhausted
                                     ▼
                             document.dead_letter  (DLQ → alert / manual replay)
```

Target properties:

- **Correlation id.** One `correlation_id` is minted at upload and threaded through
  every event, log line, and metric, so a single document's journey
  (`uploaded → … → indexed`) is one traceable story across services.
- **Idempotency key = `document_id + chunk_hash + model_version`.** Re‑delivering an
  event (Kafka is at‑least‑once) or re‑ingesting an unchanged file is a no‑op:
  a chunk whose `(document, content_hash, embedding_model)` triple already exists is
  skipped, never re‑embedded. This makes retries safe and makes re‑ingest cheap.
- **Explicit status machine.**
  `UPLOADED → PARSING → PARSED → CHUNKING → EMBEDDING → INDEXED`, with `FAILED`
  (terminal, error captured) and `DELETED` (tombstone) as off‑ramps. Each state is a
  persisted fact, not just a log message.
- **Dead‑letter + retry.** Transient failures (embedding provider 429/5xx) retry with
  exponential backoff; permanent failures land in a dead‑letter queue for inspection
  and manual replay — a poison document never blocks the topic.
- **Object storage (MinIO/S3).** Raw bytes live in an object store keyed by content
  hash; the relational row holds only a pointer. Large files never bloat the DB.

---

## 3. Current in Nexus AI (baseline)

> Paths below are relative to `backend/app/`. Line numbers verified against the
> repository at the time of writing.

The pipeline is the **same seven stages**, but it runs **in‑process** — no Kafka, no
MinIO, no separate workers. The whole unit of work is a single coroutine,
`ingest_document`, that a Celery/RQ worker could later call verbatim.

```
POST /api/kb/{id}/documents          routes/knowledge.py:138  upload_document
   │  validate ext + size, read bytes           :144 / :152 / :150
   │  Document(raw=bytes, status="pending")      :158
   │  create_job() → ingestion_jobs row          :167
   │  background.add_task(ingest_document, …)     :170   ◀── FastAPI BackgroundTasks
   ▼
ingest_document(document_id, owner_id)   services/rag_ingestion.py:57
   │
   ├─ 1 extract   clean_text(extract_text(filename, raw))      :84
   ├─ 2 chunk     chunk_text(text, rag_chunk_size, overlap)    :91
   ├─ 3 embed     provider.embed(batch, INPUT_PASSAGE)         :109–116  (batches of 32)
   ├─ 4 store     delete old chunks → insert DocumentChunk     :125 / :126–135
   │              pin KB embedding model on first ingest        :138–141
   └─ 5 finish    doc.status="completed"; job progress=100      :143–149
```

### 3.1 Two trigger paths

| Path | Trigger site | Mechanism |
|---|---|---|
| **KB documents** | `routes/knowledge.py:170` (and re‑ingest at `:231`) | `background.add_task(rag_ingestion.ingest_document, doc.id, account.id)` — runs after the HTTP response is sent |
| **Conversation attachments** | `services/multimodal_chat.py:138–140` | `asyncio.create_task(ingest_conversation_document(...))` — fire‑and‑forget so the chat turn streams immediately |

Both converge on the same extract → clean → chunk → embed → store body
(`ingest_conversation_document` at `rag_ingestion.py:186` is the conversation‑scoped
twin of `ingest_document`). `enqueue_ingestion` (`:236`) is a small helper that
schedules onto the running loop or falls back to `asyncio.run`.

### 3.2 The worker, stage by stage

- **Extract** — `extract_text(filename, raw)` (`rag_chunking.py:36`) dispatches by
  extension: `pypdf` for `.pdf` (`_extract_pdf`, `:56`), `python-docx` for `.docx`
  (`_extract_docx`, `:80`), and direct UTF‑8 decode for **37 text/code extensions**
  (`_TEXT_EXTS`, `:16–21`; `.txt .md .py .ts .go .rs .sql …`). Unknown extensions are
  accepted only if they decode as mostly‑printable text (`_looks_binary`, `:101`).
  `clean_text` (`:110`) NFC‑normalizes, strips control chars, collapses whitespace,
  and caps blank‑line runs while preserving paragraph breaks so chunking has
  boundaries to cut on.
- **Chunk** — `chunk_text(text, size, overlap)` (`rag_chunking.py:129`), driven by
  `settings.rag_chunk_size = 1200` chars and `rag_chunk_overlap = 200`
  (`config.py:54–55`). Character‑window splitting with boundary preference; see
  `03-chunking.md` for the full treatment.
- **Embed** — the KB's pinned provider is rebuilt via `embedding_provider_for_kb`
  (`rag_ingestion.py:98–104` → `providers/embeddings.py:232`), then chunks are embedded
  in **batches of 32** (`_EMBED_BATCH = 32`, `:29`; loop `:109–116`) with
  `input_type=INPUT_PASSAGE`. The 32 is a **progress‑granularity** knob — the job's
  `embedded_chunks`/`progress` update after each batch (`:114–116`). The provider
  itself re‑batches at 64 (`providers/embeddings.py:34`), so a 32‑item ingest batch is
  never split further. A count mismatch aborts the ingest (`:118`).
- **Store** — previous chunks for the document are deleted first
  (`:125`, making re‑ingest idempotent by document identity), then one
  `DocumentChunk` row per chunk is inserted (`:126–135`) carrying
  `document_id, knowledge_base_id, owner_id, ordinal, text, token_count, embedding`.
  `token_count` is the estimate `max(1, len(chunk) // 4)` (`:133`). The KB is
  **pinned** to this embedding platform/model/dim on first successful ingest
  (`:138–141`) so every later chunk and every query share one vector space.
- **Finish** — `doc.status = "completed"`, `chunk_count` set, job marked
  `progress=100` (`:143–149`). Failures never raise out of the worker: they are caught
  and recorded on the document + job by `_fail` in a fresh DB session (`:153–155`,
  `:160`).

### 3.3 Storage & status

- **Raw bytes live in the database**, not an object store:
  `Document.raw = Column(LargeBinary)` → Postgres `BYTEA` (`models/rag_models.py:102`).
  This is the single seam a MinIO/S3 adapter would replace.
- **Chunks** are `document_chunks` rows with an **unbounded** pgvector column
  (`_vector_column()` → `Vector()` with no fixed dim, `rag_models.py:32–37`), so any
  model dimension (Mistral 1024, Gemini 768, OpenAI 1536) fits with no migration.
  Keyword search rides a functional **GIN FTS index**
  (`ix_document_chunks_fts ON document_chunks USING gin (to_tsvector('english', text))`,
  created in `main.py:67–70`).
- **Progress** is persisted to the `ingestion_jobs` row and **polled** by the client
  via `GET /api/kb/{id}/documents/{doc_id}/job` (`routes/knowledge.py:199`).

### 3.4 Status machines (current vs target)

Nexus AI runs **two** lowercase status fields, not the single explicit machine of the
target:

| | Current values | Where |
|---|---|---|
| `documents.status` | `pending → processing → completed \| failed` | `rag_models.py:96–97`; set at `rag_ingestion.py:76, 143, 167` |
| `ingestion_jobs.status` | `pending → parsing → chunking → embedding → completed \| failed` | `rag_models.py:149–150`; set at `rag_ingestion.py:79, 90, 106, 148` |

Mapped onto the target `UPLOADED/PARSING/PARSED/CHUNKING/EMBEDDING/INDEXED/FAILED/DELETED`:
the job field already distinguishes `PARSING/CHUNKING/EMBEDDING`, but there is **no
distinct `PARSED`, `INDEXED`, or `DELETED`** state — `completed` collapses
"embedded" and "indexed", and deletion is a hard `db.delete` (`knowledge.py:243`)
rather than a tombstone.

### 3.5 Stage → code map

| Stage | Target | Nexus AI baseline (`backend/app/…`) |
|---|---|---|
| Ingest | accept, persist, emit `document.uploaded` | `routes/knowledge.py:138`, `rag_ingestion.create_job:32` |
| Parse | `document.parsed`, structure preserved | `rag_chunking.extract_text:36` (`_extract_pdf:56`, `_extract_docx:80`) |
| Clean | normalize, keep headings/code | `rag_chunking.clean_text:110` |
| **Dedup** | content‑hash skip, chunk idempotency | **absent** — `rag_ingestion.py:125` replaces by `document_id` |
| Chunk | structure‑aware + metadata | `rag_chunking.chunk_text:129` (char‑window) |
| Embed | batch, retry, cache, version | `rag_ingestion.py:109–116` + `providers/embeddings.py` |
| Store | pgvector + FTS + provenance | `rag_ingestion.py:126–135`, `main.py:67–70` |
| Trace | correlation id + per‑stage metrics | partial — `ingestion_jobs` progress only |

---

## 4. Design decisions / how it works

- **In‑process, but lift‑and‑shift shaped.** `ingest_document` is a single self‑contained
  coroutine with all its I/O explicit (owner id is *passed in*, not read from a
  contextvar — `knowledge.py:169`). That is deliberate: the day a Kafka/Celery worker
  is introduced, it calls the same function unchanged. `BackgroundTasks` buys async
  ingestion with **zero extra infrastructure** to boot.
- **Response returns before work starts.** Upload validates, persists `Document(raw=…)`
  and an `IngestionJob`, then schedules the task and returns `{document, job_id}`
  immediately (`knowledge.py:170–172`). The client polls the job for progress — a
  poor‑man's event stream over the `ingestion_jobs` row.
- **Idempotent by document identity (not content).** Re‑ingest does *delete‑then‑insert*
  (`:125`), so re‑running always yields one consistent set of chunks — but it
  **re‑embeds everything** even if the file is byte‑identical, because there is no
  `content_hash` to compare against. The target's `document_id + chunk_hash +
  model_version` key upgrades this from "correct" to "correct **and** cheap".
- **Never‑raise workers.** A failed document must not take down the request thread or
  poison the session, so the worker catches everything and records failure in a fresh
  session (`_fail`, `:160`). This is the in‑process stand‑in for a dead‑letter queue.
- **Pin‑on‑first‑ingest.** The KB commits to one embedding platform/model/dim the first
  time a document embeds (`:138–141`); every later chunk and every query rebuild that
  exact provider. Mixing vector spaces silently destroys retrieval, so the space is
  chosen once and recorded.
- **Bytes in the DB, on purpose (for now).** `BYTEA` keeps the stack to one datastore
  and makes re‑ingest/re‑view trivial. It is explicitly the seam for MinIO/S3 — a
  scale‑out concern, not a correctness one.

---

## 5. The gap: what's landing next

| Capability | Target | Baseline today | Status |
|---|---|---|---|
| **Content‑hash dedup** | `sha256(raw)` on the document + `chunk_hash` per chunk; skip unchanged | none — re‑ingest replaces by `document_id` and re‑embeds all | **being implemented** |
| **Structure metadata on chunks** | `section` path, `page_number`, `char_start/char_end`, `content_hash`, `parent_chunk_id` | `document_chunks` has none of these (`rag_models.py:114–135`) | **being implemented** (see `03-chunking.md`) |
| **Explicit status machine** | `UPLOADED…INDEXED/DELETED` | two lowercase fields, no `PARSED/INDEXED/DELETED` | partial |
| **Kafka event pipeline** | `document.uploaded → … → indexed` + DLQ | in‑process `BackgroundTasks` / `asyncio.create_task` | **scale‑out tier** (deferred) |
| **MinIO/S3 object storage** | pointer + object store | `documents.raw BYTEA` | **scale‑out tier** (deferred) |

The first two (dedup, structure‑aware chunking with per‑chunk provenance) are the
**pragmatic tier** and are actively being added — they change the *contents* of the
existing tables and the chunker, not the deployment topology. Kafka and MinIO are the
**scale‑out tier**: documented so the code can grow into them (hence the
`ingest_document`‑as‑one‑coroutine and `raw`‑as‑one‑seam designs) but intentionally
not on the single‑Postgres roadmap.

---

## 6. Pitfalls

- **Re‑embedding unchanged content.** Without content hashing, every re‑ingest pays the
  full embedding cost (and provider rate limit) again. Add `content_hash` before
  worrying about throughput.
- **Blocking the request thread.** Ingestion must *never* run inline in the upload
  handler; the whole design depends on `background.add_task` / `create_task` returning
  first. A synchronous extract of a 25 MB PDF would stall the response.
- **Losing structure at parse time.** `_extract_pdf` joins pages with `\n\n` and
  **discards page numbers** (`rag_chunking.py:66–71`); `clean_text` flattens heading
  formatting. Whatever provenance you don't capture *here* is unrecoverable downstream —
  the chunk can't cite a page it never knew about.
- **Silent fallback embeddings.** With no embedding key, ingestion still "succeeds" using
  the keyless `HashEmbedding` (`providers/embeddings.py:140`), which has poor semantic
  quality. The job completes green; retrieval is quietly bad. Surface the fallback state.
- **Vector‑space drift.** If a KB's pinned provider key disappears, ingestion falls back
  to auto‑detection (`embeddings.py:256–260`) and may write chunks in a *different*
  space than earlier ones. Same‑KB vectors must all share one model — treat a pin miss
  as a hard error, not a warning, once dedup lands.
- **At‑least‑once ≠ exactly‑once (future).** When the Kafka tier arrives, events will be
  redelivered; without the `document_id + chunk_hash + model_version` idempotency key,
  a redelivery double‑inserts chunks. Build the key with the dedup work, not after.
- **Unbounded vector = no ANN index.** The `Vector()` column can't carry HNSW/IVFFlat, so
  storage scales as an exact `<=>` scan. Fine at personal‑KB scale; a real cost if a
  single KB grows to millions of chunks (that's the `05-vector-storage.md` fixed‑dim
  tier).
