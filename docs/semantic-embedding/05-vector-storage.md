# Vector Storage — Nexus AI

> Where vectors live and how they are searched: pgvector in PostgreSQL.
> Drill-down for the **Store** box in `00-ARCHITECTURE.md` (§2.1 / §4).
> Target = fixed-dim, ANN-indexed, ACL-in-SQL storage; baseline = the two
> distinct embedding subsystems that ship today.

Vector storage is where the two pipelines meet. The indexing path writes chunk
vectors here; the retrieval path reads its nearest neighbours from here. Nexus AI
keeps this in **one PostgreSQL database** via the **pgvector** extension — no
separate vector database — so relational scope keys, full-text search and dense
vectors coexist in the same rows and the same transaction. This document covers
how vectors are stored, indexed, filtered, and (critically) how *access control*
is enforced inside the query rather than after it.

---

## 1. Target architecture

### 1.1 pgvector in Postgres

A `vector` column holds the embedding; a distance operator ranks it. Nexus AI
uses cosine distance `<=>` (`similarity = 1 - distance`). The write side stores
`embedding vector(D)` with a **fixed** dimension `D` per model; the read side
issues `ORDER BY embedding <=> :q LIMIT k`. Fixed dimension is the precondition
for an approximate-nearest-neighbour (ANN) index — you cannot index a column
whose vectors vary in length.

### 1.2 ANN index — HNSW vs IVFFlat

An exact `<=>` scan is O(rows); at scale you need an ANN index. pgvector offers
two, and the choice is a genuine trade-off:

| | **HNSW** | **IVFFlat** |
|---|---|---|
| Search speed | Fast | Fast (once tuned) |
| Recall | High, robust | Good, but sensitive to `lists`/`probes` |
| Build | **No train stage**; incremental inserts fine | Needs a **train stage** over representative data |
| Data needed | Works from row 1 | Needs *enough* rows before the clustering is meaningful |
| Memory | **Higher** (graph in memory) | **Lower** |
| Build time | Slower to build | Faster to build |
| Tuning | `m`, `ef_construction`, `ef_search` | `lists` (build), `probes` (query) |

Rule of thumb: **HNSW** when recall and query latency matter more than RAM and
you want zero tuning ceremony; **IVFFlat** when memory is tight and you have
enough data to train and the patience to tune `lists`/`probes`.

### 1.3 Access control *inside* the query

> Authorization is a **`WHERE` clause**, never a post-filter.

The single most important storage rule for multi-tenancy: the tenant/owner
predicate and the vector search run in the **same** SQL statement, so the ANN
index only ever ranks rows the caller is allowed to see. Filtering *after* a
top-k search is both a correctness bug (you get fewer than `k`, or leak counts)
and a security bug.

```sql
SELECT id, document_id, text,
       1 - (embedding <=> :q) AS similarity
FROM   document_chunks
WHERE  owner_id = :authenticated_user_id     -- ACL, in-query
  AND  knowledge_base_id = :kb_id            -- scope
  AND  embedding IS NOT NULL
ORDER  BY embedding <=> :q                    -- cosine distance, index-backed
LIMIT  :k;
```

### 1.4 Metadata filtering

Beyond ACL, real retrieval narrows by structured metadata carried on the chunk
row, combined with the vector order-by:

- `tenant_id`, `org_id` — hard isolation boundaries.
- `collection_id` / `knowledge_base_id` — which corpus.
- `source_type` — pdf / markdown / code / chat.
- `language` — restrict to the query's language.
- `created_at` — recency windows / "as of" queries.
- `tags` — user- or system-applied labels.

Each is an indexed `WHERE` predicate *alongside* `ORDER BY embedding <=> :q`, so
the database prunes first and ranks the survivors.

---

## 2. Current in Nexus AI (baseline)

Nexus AI ships **two separate embedding subsystems** with different storage
strategies, dimensions, index stories *and* migration mechanisms. They do not
share a table and should not be conflated.

### 2.1 Subsystem 1 — RAG `document_chunks` (unbounded, no ANN)

[`rag_models.py`](../../app/models/rag_models.py) defines `document_chunks` with
an **UNBOUNDED** pgvector column:

- `_vector_column()` returns `Column(_Vector())` — a `vector` with **no fixed
  dimension** (`rag_models.py:32-37`), so any auto-detected model (mistral 1024,
  gemini 768, openai 1536, …) fits without a migration.
- Guarded by a `HAS_PGVECTOR` toggle; when pgvector is absent the column
  **degrades to `JSON`** (`rag_models.py:38-42`).
- The column is wired onto the chunk at `rag_models.py:131`; the full row today
  is `id, document_id, knowledge_base_id, conversation_id, owner_id, ordinal,
  text, token_count, embedding, created_at` (`rag_models.py:114-135`).

**Consequence — no ANN index.** An unbounded `vector` **cannot carry an HNSW or
IVFFlat index**, so semantic search is an **exact `<=>` scan**:

- With pgvector: `semantic_search()` orders by
  `DocumentChunk.embedding.cosine_distance(query_vec)` over the KB's rows
  ([`rag_retrieval.py:50-66`](../../app/services/rag_retrieval.py)) — correct,
  but a sequential scan of the filtered set.
- Without pgvector (JSON mode): `_semantic_python()` brute-forces cosine in
  Python over a **hard cap of 5000 rows** (`rag_retrieval.py:69-87`, cap at
  `:74`); the per-conversation variant caps at 4000 (`rag_retrieval.py:206-207`).

Keyword search *does* get an index: a functional **GIN full-text index** on
`document_chunks`, created in DDL as
`ix_document_chunks_fts ... USING gin (to_tsvector('english', text))`
([`main.py:67-70`](../../app/main.py)) and used via `to_tsvector`/`ts_rank` in
`keyword_search()` (`rag_retrieval.py:90-118`).

**ACL note.** The vector/keyword queries filter by `knowledge_base_id` (or
`conversation_id`), *not* directly by `owner_id` — owner isolation is enforced
upstream by loading the KB scoped to the caller. The denormalized `owner_id`
column exists on the chunk (`rag_models.py:126`, "so search filters in one
predicate") but is not yet part of the retrieval `WHERE` clause. Tightening this
to the in-query `owner_id` predicate of §1.3 is a target item.

### 2.2 Subsystem 2 — `memory_chunks` (fixed-dim, IVFFlat)

Conversation **memory** uses the opposite strategy, via Alembic
([`migrations/versions/001_add_agent_tables.py`](../../migrations/versions/001_add_agent_tables.py)):

- `embedding Vector(EMBEDDING_DIM)` with `EMBEDDING_DIM = 1536`, `nullable=False`
  (`001:19`, `001:64`) — a **fixed** dimension.
- A real ANN index: **IVFFlat** cosine, `lists = 100`
  (`001:71-74`):
  ```sql
  CREATE INDEX memory_chunks_embedding_idx ON memory_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
  ```

So memory retrieval is index-backed and fixed to 1536-dim (OpenAI
`text-embedding-3-small`, per `config.embedding_model`), while RAG retrieval is
an exact scan over whatever dimension its KB pinned.

### 2.3 The split migration story

The two subsystems even *evolve differently*:

- **RAG schema** grows via **idempotent DDL at boot** — `CREATE EXTENSION IF NOT
  EXISTS vector` (`main.py:48-57`), `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
  and `CREATE INDEX IF NOT EXISTS` in `_ensure_auth_schema` / `_ensure_rag_schema`
  (`main.py:60-84`), all run from `lifespan` around `create_all`
  (`main.py:91-94`). No Alembic revision touches the RAG tables.
- **Memory schema** uses **Alembic** (`001_add_agent_tables`) with a real
  `upgrade()`/`downgrade()`.

One database, two migration philosophies — worth remembering when reasoning
about how a column actually got there.

---

## 3. Design decisions

- **Unbounded `Vector()` buys zero-migration multi-model support.** Auto-detecting
  the embedding model at first ingest means the column must accept *any*
  dimension, so RAG deliberately trades **ANN-indexability** for flexibility. At
  personal-KB scale an exact `<=>` scan over a KB's chunks is fast enough, and
  the cost (sequential scan) is bounded by how much one user uploads. The KB pin
  (`05` cross-refs `04` §2.3) keeps every vector in a KB the same length even
  though the *column* permits mixed lengths.
- **Scale-out tier = fixed-dim per-model tables + HNSW.** When a single KB grows
  past exact-scan comfort, the documented path is per-model tables with a fixed
  `vector(D)` and an HNSW index — the `memory_chunks` pattern, generalized. This
  is a new table shape, not an `ALTER`, precisely because the current column is
  unbounded.
- **FTS is indexed even though vectors are not.** The GIN index means the sparse
  half of hybrid search is always fast; hybrid retrieval (RRF of dense + sparse)
  partly compensates for the un-indexed dense scan.
- **New chunk metadata columns (being added).** To reach the target
  provenance/structure model, `document_chunks` gains:

  | Column | Purpose |
  |---|---|
  | `content_hash` | sha256 of normalized chunk text — dedup + embedding-cache key |
  | `section` | heading path, e.g. `Auth ▸ Refresh` — structure metadata |
  | `page_number` | source page (PDF) — citation |
  | `char_start`, `char_end` | offsets into the cleaned document — highlighting |
  | `parent_chunk_id` | → `document_chunks.id` — small-to-big / parent expansion |
  | `is_parent` | parent vs child chunk |
  | `embedding_model` | e.g. `mistral/mistral-embed` — per-chunk versioning |
  | `embedding_version` | e.g. `2026-06` — migration safety |

  These add per-chunk provenance (so the compatibility rule is enforceable *per
  row*, not just per KB) and the structure needed for citations and parent
  expansion. They arrive through the same idempotent `ADD COLUMN IF NOT EXISTS`
  DDL as the rest of the RAG schema.

---

## 4. Pitfalls

- **Mixing embedding models in one column.** The unbounded `Vector()` *permits*
  mixed dimensions/spaces in `document_chunks` — nothing at the schema level stops
  a 768-dim vector and a 1536-dim vector sharing the table. Only the KB pin keeps
  them apart. Break the pin (model swap, key change, fallback flip to
  `HashEmbedding`) and you silently poison a KB with incomparable vectors. This is
  why per-chunk `embedding_model`/`embedding_version` (§3) matters.
- **Unbounded vector = no ANN, forever.** You cannot bolt an HNSW/IVFFlat index
  onto `document_chunks.embedding` while it is dimensionless. Any latency fix
  requires migrating to a fixed-dim table first — plan for it before a KB gets
  large, not after.
- **JSON fallback does not scale.** With pgvector absent, dense search is
  Python brute-force capped at ~5000 rows (`rag_retrieval.py:74`). Beyond that cap
  results are silently truncated to whatever 5000 rows the query happened to load
  — acceptable for dev/CI, not for real corpora. Treat `HAS_PGVECTOR == False` as
  a degraded mode.
- **ACL after search, not in it.** Any future change that filters ownership in
  Python after a top-k vector query reintroduces the classic leak/short-result
  bug. Keep the owner/tenant predicate in the SQL (§1.3); prefer promoting the
  existing denormalized `owner_id` into the retrieval `WHERE` clause.
- **Two migration mechanisms, one database.** A column may come from boot-time
  DDL (RAG) *or* Alembic (memory). Editing an Alembic revision will not touch RAG
  tables and vice-versa; know which subsystem you are changing.
- **IVFFlat needs data + tuning.** `memory_chunks`' `lists = 100` is a fixed
  build-time choice; too few rows makes the clustering meaningless and recall
  drops, and `probes` must be tuned at query time. It is not "set and forget".
