# Part D — Memory & Knowledge — Nexus AI

> Layered memory that makes the assistant remember *the user* across sessions —
> and *the project* across its chats. Adapted to Nexus AI's stack (FastAPI +
> Postgres/pgvector), reusing the RAG embedding provider. Built in phases;
> **Phases 1–5 are shipped** (the layered stack + project brain + content graph +
> a personal memory graph).

```
   WORKING memory ── session end / idle ──▶ EPISODIC memory ── distil ──▶ SEMANTIC memory
   (in-session       Reflector pipeline       (durable per-user            (skills · prefs ·
    ring buffer)                                Q&A log + vector)            lessons + vector)
        │                                             ▲                            │
     cleared                                          └────── recall ──────────────┘
```

## Layers → code

| Layer | Module | Persistence | Status |
|---|---|---|---|
| **Working** | `app/memory/working.py` | none (in-proc ring buffer) | ✅ Phase 1 |
| **Episodic** | `app/memory/episodic.py` + `db_models.MemoryChunk` | Postgres + pgvector | ✅ Phase 1 |
| **Semantic** | `app/memory/semantic.py` (+ `skills` table) | Postgres + pgvector | ✅ Phase 2 |
| **Reflector** | `app/memory/skills_extractor.py` | — | ✅ Phase 2 |
| **Feedback** | `MessageFeedback` + `POST /api/chat/feedback` | Postgres | ✅ Phase 2 |
| **Lifecycle** | `app/memory/data_lifecycle.py` + `routes/memory.py` | — | ✅ Phase 3 |
| **Project brain** | `app/memory/project_brain.py` + `ProjectBrainEntry` | Postgres + pgvector | ✅ Phase 4 |
| **Knowledge graph** | `app/rag/knowledge_graph.py` + `KgEdge` | Postgres | ✅ Phase 4 |
| **Personal memory graph** | `app/memory/memory_graph.py` + `MemoryEdge` | Postgres | ✅ Phase 5 |

---

## Phase 1 — foundation (SHIPPED ✅)

The starting point was **one naive feature**: `MemoryChunk` written at turn end /
read at turn start, but with **fake SHA-256 "embeddings"**, **Python-only cosine**
(the IVFFlat index was dead), and scoping by **conversation, not user**. Phase 1
turned it into real per-user memory:

- **Working memory** — `working.py`: a bounded, in-process per-conversation ring
  buffer of recent turns + a scratch dict, LRU-evicted across conversations. The
  turn-end hook pushes into it (feeds the future Reflector); `snapshot()` exposes
  it. Ephemeral by design ("cleared at session end").
- **Episodic memory** — `episodic.py`: `store()` / `search()` using the **real RAG
  embedding provider** (`providers/embeddings.py` — cached + retried; hash only as
  the keyless fallback) and **real pgvector cosine** (`<=>`). **Owner-scoped** (the
  authenticated `account.id`) so recall is per-user across conversations;
  **dimension-guarded** (`embedding_dim` per row) so a model change never mixes
  vector spaces. Owner-less/legacy tool calls fall back to conversation scoping.
- **Write-through** — the row is committed first, then its vector is set, so a
  vector hiccup never blocks a write. `store()`/`search()` never raise.
- **Schema** — `MemoryChunk` gained `owner_id` + `embedding_dim`; `embedding`
  widened to an **unbounded `vector`** (idempotent DDL in `main.py::
  _ensure_memory_schema`, handling the legacy JSON→vector conversion) and the old
  fake vectors cleared once.
- **Rewire** — `memory_tool.py` (the 4 LLM tools) and `memory_manager.py`
  (turn-start/turn-end hooks) both delegate to `episodic`; `owner_id` is threaded
  into the orchestrated **and** WebSocket paths.

**Verified:** migration (JSON→vector) applied; store→recall round-trips at sim 1.0;
owner B gets 0 results from owner A's memory; 27 existing tests pass; backend boots
healthy.

**Config:** `memory_similarity_threshold` (0.7) is the recall cut-off. Old fake
memories were cleared — real per-user memory accrues from here.

---

## Phase 2 — semantic memory + Reflector + feedback (SHIPPED ✅)

- **Semantic memory** — `semantic.py` + the `skills` table: distilled
  preferences / skills / lessons with real embeddings. `upsert_skill()` dedups —
  a near-identical existing skill is REINFORCED (support_count↑, confidence↑)
  instead of duplicated; `search()` recalls the most relevant skills (owner-scoped,
  looser threshold since skills are broad). Verified: dedup collapses repeats
  (support_count=2), distinct skills stay separate.
- **Reflector** — `skills_extractor.py`: `maybe_reflect()` fires debounced (once
  per `memory_reflect_every_turns` stored turns) and in the background; `reflect()`
  reads the transcript + 👍/👎 feedback, asks the LLM to distil durable facts about
  the *user* ("responds well to X" / "this isn't landing"), and upserts each skill.
  Verified: JSON parsing (rejects bad kinds) + debounce (fires once per N turns).
- **Feedback capture** — `MessageFeedback` table + `POST /api/chat/feedback`
  (owner-scoped, upsert per (owner, conversation, message_index)); the Flutter
  thumbs up/down now POSTs (`ApiService.submitFeedback` ← `message_bubble.dart`).
  Verified live: up→down toggle updates one row.
- **Recall wiring** — `memory_manager.auto_search` injects **## Relevant Memory**
  (episodic) **and ## About the user** (semantic skills) at turn start;
  `auto_store` triggers reflection at turn end.

## Phase 3 — lifecycle: retention / export / purge (SHIPPED ✅)

`app/memory/data_lifecycle.py` + `routes/memory.py` (owner-scoped, JWT):

- **Retention** — `apply_retention(days)` purges EPISODIC rows older than the
  window (the raw Q&A log grows unbounded); **skills stay** (distilled + durable).
  A background sweeper runs it every `memory_retention_sweep_hours`, gated on
  `memory_retention_days > 0` (**default 0 = keep forever**). Verified: an old row
  is purged while recent ones + skills + feedback survive.
- **Export** — `GET /api/memory/export` returns the user's full memory as a
  downloadable JSON (episodic + skills + feedback; `Content-Disposition`
  attachment). Data portability.
- **Purge** — `DELETE /api/memory?scope=all|episodic|skills|feedback` — the
  per-user "forget me". Verified: purge-all clears every layer to 0.
- **View** — `GET /api/memory` returns counts + top skills.
- **UI** — a **Memory & Privacy** section in Settings shows the counts + "What I
  know about you", with **Export my memory** (saves the JSON) and **Clear my
  memory** (confirm → purge). Verified live (endpoints + 401 when unauthenticated).

## Phase 4 — project brain + content knowledge graph (SHIPPED ✅)

Where Phases 1–3 remember *the user*, Phase 4 remembers *the project* — durable
knowledge shared by every chat filed under it, learned automatically.

- **Project brain** — `project_brain.py` + the `project_brain_entries` table: an
  auto-maintained store of **facts / decisions / conventions / goals** about the
  project itself, extending the static `Project.instructions`. A per-project
  **Reflector** (`reflect()`, debounced once per `memory_project_reflect_every_turns`
  turns, background, project chats only) reads the transcript, asks the LLM to
  distil stable *project-level* knowledge (not personal prefs, not transient task
  detail), and `add_entry()` **dedup-reinforces** it (real embeddings +
  pgvector cosine; a near-duplicate above `memory_brain_dedup_threshold` bumps
  `support_count` instead of adding a row) — same machinery as semantic skills.
  `render()` injects a **=== PROJECT BRAIN ===** block (grouped goal → decision →
  convention → fact) into the system prompt for every chat in the project.
- **Content knowledge graph** — `rag/knowledge_graph.py` + the `kg_edges` table:
  `extract()` (debounced once per `memory_kg_every_turns` turns, background) asks
  the LLM for subject–relation–object triples over durable entities (people,
  tools, tech, files, orgs, endpoints), `_store()` dedups them on
  lowercased (source, relation, target) scoped by project / conversation, and
  `graph()` / `query()` / `render()` expose a nodes+edges view, keyword neighbour
  lookup, and a **=== RELATED FACTS ===** prompt block keyed to the user's message.
- **Scoping & safety** — everything **owner-scoped** (authenticated `account.id`)
  and project/conversation-scoped; config-gated (`memory_project_brain_enabled`,
  `memory_kg_enabled`, both default on); triggered from `memory_manager.auto_store`
  in try/except so they can never interrupt a turn; deleting a project drops its
  brain + graph rows.
- **APIs** — `GET /api/projects/{id}/brain`, `DELETE /api/projects/{id}/brain/{entry_id}`,
  `GET /api/projects/{id}/graph` (all owner-checked via `_owned`).
- **UI** — a **Brain** tab on the project page lists what the assistant has learned
  (grouped by kind, with ×N reinforcement badges + a per-entry "forget" action) and
  a header summary of the linked-facts graph; pull-to-refresh reloads it.

**Verified:** 14/14 module smoke checks (insert, dedup-reinforce → support_count,
owner isolation, KG store/dedup/query/graph/render); live HTTP round-trip through
auth (brain GET/DELETE, graph GET, cross-owner 404); 139 backend tests unaffected
(pre-existing websearch/agent-config failures only); desktop + APK rebuilt.

**Config:** `memory_project_reflect_every_turns` (8), `memory_brain_dedup_threshold`
(0.9), `memory_kg_every_turns` (4).

## Phase 5 — personal memory graph (SHIPPED ✅)

Where Phase 4's *content* knowledge graph (`rag/knowledge_graph.py`, `KgEdge`) is
scoped to a project/conversation, the **personal memory graph** (`memory/
memory_graph.py`, `MemoryEdge`) is owner-scoped and **cross-conversation** — "what
I know about *you*", relationally. It captures only **people/orgs** you mention and
**tools/tech** you work with, as subject–relation–object triples distilled from
episodic memory across all chats.

- **Extraction** — `extract()` asks the LLM for personal triples (subject usually
  "User"); `_parse()` normalizes `source_type`/`target_type` and keeps only
  `person | org | tool | tech | user` (topic/preference/goal triples are dropped so
  the graph stays focused). `maybe_extract()` is debounced (once per
  `memory_graph_every_turns` turns, default 3) and runs in the background.
- **Reinforce + decay** — `_store()` dedups on lowercased (source, relation,
  target); a repeat bumps `support_count` (strongest recalled first). A slow decay
  (`memory_graph_decay_days`, default 90) fades edges not reinforced within the
  window and drops the weakest — wired into the lifecycle sweeper alongside
  episodic retention. (Decay deletes weak stale edges outright and preserves
  `updated_at` when fading stronger ones, so a fact keeps decaying across sweeps
  instead of a bulk-update's `onupdate` resetting its staleness.)
- **Recall (query-relevant)** — `render()` injects only the graph facts whose
  entities match the current message; `memory_manager.auto_search` adds them as a
  **## What I know about you** block at turn start (low token cost).
- **Graph API** — `GET /api/memory/graph` (nodes+edges), `…/graph/neighbors?entity=`
  (edges touching an entity), `…/graph/query?q=` (keyword-matched); all owner-scoped
  via JWT. Included in the memory `summary` / `export` / `purge?scope=graph|all`.
- **Bug fix on the way through:** `knowledge_graph.maybe_extract` (async) was being
  called from `auto_store` **without `await`**, so the content-KG trigger's coroutine
  was created and discarded (never ran) — now awaited, so the content graph
  actually populates from chats too.

**Verified:** 20/20 module smoke checks (type-filtering, insert + reinforce →
support_count, query/neighbors/graph/render, owner isolation, decay fades/deletes
stale while fresh survives, summary/purge); live HTTP round-trip through auth
(`/graph`, `/graph/neighbors`, `/graph/query`, counts). Backend-only (surfaces:
prompt recall + Graph API) — no app rebuild.

**Config:** `memory_graph_enabled` (on), `memory_graph_every_turns` (3),
`memory_graph_decay_days` (90; 0 = keep forever).
