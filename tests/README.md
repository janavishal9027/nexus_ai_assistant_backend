# Backend tests

```bash
cd backend
pytest tests/unit                 # no database needed
pytest tests/integration          # needs Postgres + pgvector
pytest tests/ -q                  # everything
```

## The test database

Integration tests run against a **dedicated database**, never your development
one — they create and delete accounts, and would destroy real data.

`tests/conftest.py` creates `<your_db>_test` (e.g. `chatapp_test`) on first run,
enables pgvector in it, and points the app there before `app.database` binds its
engine. Override the name with `TEST_POSTGRES_DB`.

Two things about this are load-bearing:

- The override is **`POSTGRES_DB`, not `DATABASE_URL`**. `Settings.database_url`
  is a computed property built from the `postgres_*` fields, so exporting
  `DATABASE_URL` does nothing. This is easy to get wrong and fails *silently* by
  pointing tests at the dev database.
- conftest **refuses to run** against a database whose name doesn't end in
  `_test`, skipping the integration tests with a reason instead. That assertion
  has already caught one misconfiguration that would have run the suite against
  real user data. Don't remove it.

If Postgres isn't reachable, integration tests skip (with the reason) and unit
tests still run. Use `-rs` to see skip reasons.

## What's covered

| File | Guards |
|---|---|
| `integration/test_account_deletion.py` | `DELETE /api/auth/me` erases **every** owner-scoped row. Account ids are sequential, so a leaked row is inherited by the next account with that id. `test_no_owner_scoped_table_is_missed` walks the ORM and fails when a model gains an `owner_id` but isn't wired into `_OWNER_SCOPED` — it caught the RAG tables (documents, chunks, jobs, knowledge bases) leaking. |
| `integration/test_memory_lifecycle.py` | Purge scopes delete exactly their own layer; an unknown scope raises/422s instead of meaning "all" (it once wiped everything on a typo); export carries every layer; both graphs reinforce repeats and decay, preserving `updated_at` so edges keep ageing. |
| `integration/test_memory_api.py` | Prefs default on, patch partially, fail open, and can't override an operator flag. One account cannot read or delete another's graph edges. |
| `unit/test_graph_recall.py` | Edge recall: the keyword fallback searches `relation` (it never used to), cosine is safe on bad input, recall never raises without an embedding provider. |

## Known state

The pre-existing suite has **7 failures and 13 errors** unrelated to the above
(`test_web_search_tool`, `test_tool_executor_batch`, `test_agent_config`). They
fail with or without these fixtures. CI runs the memory suites as gating and the
rest non-gating until they're fixed.
