# Implementation Plan: Full-Stack Agent Orchestration

## Overview

Build the full-stack agent orchestration platform on top of the existing FastAPI chatapp backend. The implementation is layered: infrastructure and configuration first, then service tools, then orchestration components, then the gateway and API layer, and finally observability and wiring. All new files are additive; no existing files are deleted.

## Tasks

- [ ] 1. Project setup — configuration, feature flags, and database models
  - [ ] 1.1 Extend `app/config.py` Settings with new fields
    - Add `redis_url`, `kafka_bootstrap_servers`, `fcm_credentials_path`, `embedding_model`, `memory_similarity_threshold`, `realtime_event_buffer_size`, and `agent_features` fields with documented defaults
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.7_

  - [ ] 1.2 Implement `app/services/feature_flags.py`
    - Implement `AgentFeatures` frozen dataclass and `get_agent_features()` with `lru_cache`
    - Parse comma-separated `AGENT_FEATURES` env var into boolean flags: `planner`, `redis_cache`, `kafka`, `fcm`, `websocket`
    - _Requirements: 15.7, 15.8_

  - [ ] 1.3 Add new SQLAlchemy models to `app/models/db_models.py`
    - Add `User`, `Task`, `MemoryChunk` (with pgvector `Vector(1536)` column), `FCMToken`, and `AuditLog` models
    - Ensure no `password`, `password_hash`, or `api_key` fields exist on `User`
    - _Requirements: 5.9, 19.4, 13.3, 6.2, 19.7_

  - [ ] 1.4 Add Pydantic schema extensions to `app/models/schemas.py`
    - Add `AgentChatRequest`, `AgentChatResponse`, `SubtaskStatus`, `PlanSummary`, `UserDto`, `TaskDto`, `PaginatedUsers`, `PaginatedTasks`, `HealthComponent`, `HealthResponse`
    - _Requirements: 1.1, 1.6, 1.8, 3.8, 5.3, 6.2_

  - [ ] 1.5 Create Alembic migration `migrations/versions/001_add_agent_tables.py`
    - Enable `pgvector` extension, create `users`, `tasks`, `memory_chunks`, `fcm_tokens`, `audit_logs` tables
    - Create IVFFlat index on `memory_chunks.embedding` using `vector_cosine_ops`
    - Do not modify existing tables
    - _Requirements: 17.5, 8.1_


- [ ] 2. Infrastructure services — Redis Cache and Kafka
  - [ ] 2.1 Implement `app/services/redis_cache.py`
    - Implement `RedisCache` with async connection pool (min 5, max 20), 500 ms operation timeout on every public method, `get`, `set`, `delete`, `ping`, `close`, and key-helper statics
    - All methods catch exceptions and return `None`/`False` as cache misses; log `WARNING` on failure
    - _Requirements: 9.1, 9.4, 9.6, 9.8, 9.9, 9.11_

  - [ ]* 2.2 Write property test for Redis cache miss transparency (Property 7)
    - **Property 7: Redis Cache Miss Falls Back Transparently**
    - **Validates: Requirements 9.7**

  - [ ] 2.3 Implement `app/services/kafka_producer.py`
    - Implement `KafkaProducer` with `acks=all`, `linger_ms=100`, `max_batch_size=16384`, and 5 s publish timeout
    - `publish()` is fire-and-forget; logs `WARNING` on timeout or broker error, never raises
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.8_

  - [ ] 2.4 Implement `app/services/kafka_consumer.py` and `EventBuffer`
    - Implement `EventBuffer` bounded ring buffer (per-topic `deque(maxlen=N)`, newest-first)
    - Implement `KafkaConsumer` that subscribes to `agent.commands`, routes `notify_user` to `WebSocketManager`, and pushes all events into `EventBuffer`
    - _Requirements: 10.6, 10.7, 18.3, 18.4_

  - [ ]* 2.5 Write unit tests for Kafka producer fire-and-forget behavior
    - Verify `KafkaProducer.publish()` does not raise when broker is unavailable
    - _Requirements: 10.5_


- [ ] 3. WebSocket Manager
  - [ ] 3.1 Implement `app/services/ws_manager.py`
    - Implement `WebSocketManager` with session registry (max 100), atomic `register`/`unregister` (single dict mutation), `send`, `send_error_and_close`, `count`, and `is_active`
    - Keepalive loop: ping every 30 min idle; close session if no pong within 10 s; ignore late pongs
    - `start()` / `stop()` for lifespan integration
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

  - [ ]* 3.2 Write unit tests for WebSocket Manager session lifecycle
    - Test register/unregister atomicity, capacity limit (RuntimeError at 100+), send-error-close flow
    - _Requirements: 1.4, 1.5, 2.6, 2.7_

- [ ] 4. Tool Router
  - [ ] 4.1 Implement `app/services/tool_router.py`
    - Implement `classify_tool()` pure prefix function and `ToolRouter` class with handler injection
    - `route()`: classify → validate registry presence → dispatch to correct handler → log at INFO (tool name, category, handler class, correlation_id)
    - Return `ToolResult(status="error")` for unregistered tools; return `ToolResult(status="error")` for disabled tools
    - Include correlation_id in every downstream call and result
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11, 19.5_

  - [ ]* 4.2 Write property test for Tool Router prefix classification (Property 1)
    - **Property 1: Tool Router Prefix Classification Invariant**
    - **Validates: Requirements 4.1, 4.8**

  - [ ]* 4.3 Write property test for Tool Router unregistered tool error (Property 2)
    - **Property 2: Tool Router Unregistered Tool Returns Error**
    - **Validates: Requirements 4.7**

  - [ ]* 4.4 Write unit tests for Tool Router dispatch
    - Test service_tool prefix routing, disabled-tool guard, and INFO log output
    - _Requirements: 4.1, 4.2, 19.5_


- [ ] 5. Checkpoint — core infrastructure passes
  - Ensure all tests pass for tasks 1–4, ask the user if questions arise.

- [ ] 6. User Service Tool
  - [ ] 6.1 Implement `app/tools/user_service.py`
    - Register `user_get`, `user_list`, `user_create`, `user_update` in the Tool_Registry
    - `user_get`: check Redis cache first; on DB timeout > 2 s return `degraded: true`; strip sensitive fields
    - `user_list`: paginated response (default page size 20) with `items`, `page`, `page_size`, `total_count`
    - `user_create`: validate email uniqueness, return error on duplicate
    - `user_update`: apply only provided fields; invalidate `user:{user_id}` Redis key on success
    - Validate all inputs against JSON Schema before any DB operation; write `AuditLog` entries for create/update
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 9.4, 9.5, 19.6, 19.7_

  - [ ]* 6.2 Write property test for User Service Tool sensitive field stripping (Property 8)
    - **Property 8: User Service Tool Strips Sensitive Fields**
    - **Validates: Requirements 5.9, 19.4**

  - [ ]* 6.3 Write unit tests for User Service Tool
    - Test `user_get` not-found error, `user_create` duplicate email error, `user_update` partial fields, validation-fail no-DB-op guarantee
    - _Requirements: 5.6, 5.7, 5.8_

- [ ] 7. Task Service Tool
  - [ ] 7.1 Implement `app/tools/task_service.py`
    - Register `task_get`, `task_list`, `task_create`, `task_update`, `task_complete` in the Tool_Registry
    - `task_list`: return all required fields; support optional filters (`status`, `assignee_id`, `due_date_from`, `due_date_to`, `priority`); cap at 100 rows with `next_page_token`; include pagination metadata
    - `task_create`: validate required fields before DB write; return created record with assigned `id`
    - `task_complete`: set `status = "completed"` and `completed_at = UTC now`
    - Validate all inputs against JSON Schema; write `AuditLog` entries for create/update/complete/delete; enforce user-scoped authorization
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 19.6, 19.7_

  - [ ]* 7.2 Write unit tests for Task Service Tool
    - Test `task_list` filter combinations, `task_create` missing fields error, `task_complete` timestamp, not-found errors, validation-fail no-DB-op guarantee
    - _Requirements: 6.3, 6.6, 6.9_


- [ ] 8. Database Tool
  - [ ] 8.1 Implement `app/tools/database_tool.py`
    - Register `query_database` in the Tool_Registry
    - Pipeline: LLM generates SQL → syntax check → keyword check (`_validate_sql_keywords`) → parameterization/interpolation check → execute via SQLAlchemy `text()` with empty params → return rows
    - Cap `max_rows` at 500; include `truncated: true` and `total_available` when results are capped
    - Return exact error messages: `"Generated SQL is syntactically invalid"`, `"Only SELECT queries are permitted"`, `"Parameterization required: direct value interpolation detected"`, `"Could not generate a valid SQL query from the provided description"`
    - Log generated SQL and execution duration at INFO; never return partial results on any error
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 19.2, 19.3_

  - [ ]* 8.2 Write property test for Database Tool DML/DDL rejection (Property 9)
    - **Property 9: Database Tool Rejects DML/DDL Keywords**
    - **Validates: Requirements 7.3**

  - [ ]* 8.3 Write property test for Database Tool unparameterized literal rejection (Property 10)
    - **Property 10: Database Tool Rejects Unparameterized Literals**
    - **Validates: Requirements 7.7, 19.3**

  - [ ]* 8.4 Write unit tests for Database Tool
    - Test syntax-invalid SQL error message, each forbidden keyword variant (case-insensitive), LLM parse failure, 500-row cap and `truncated` field
    - _Requirements: 7.3, 7.4, 7.5, 7.8_

- [ ] 9. Memory Service Tool and pgvector integration
  - [ ] 9.1 Implement `app/tools/memory_tool.py`
    - Register `memory_store`, `memory_search`, `memory_delete`, `memory_store_batch` in the Tool_Registry
    - `memory_store`: generate embedding via configured `EMBEDDING_MODEL`; persist `MemoryChunk` row; return error on embedding failure without DB write
    - `memory_search`: generate query embedding; cosine similarity query filtered by `MEMORY_SIMILARITY_THRESHOLD`; default `top_k=5`, hard max 20; return all qualifying chunks when `top_k` exceeds stored count
    - `memory_delete`: check ownership (`user_id` / `conversation_id`); return "Access denied" on scope violation
    - `memory_store_batch`: accept up to 50 items; generate embeddings in parallel; persist in a single transaction
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.9, 8.11, 8.12, 13.1, 13.3, 13.4, 13.5, 13.6_

  - [ ]* 9.2 Write property test for Memory store-then-search round-trip (Property 5)
    - **Property 5: Memory Store-Then-Search Round-Trip**
    - **Validates: Requirements 8.10, 13.2**

  - [ ]* 9.3 Write property test for Memory duplicate store creates distinct records (Property 6)
    - **Property 6: Memory Duplicate Store Creates Distinct Records**
    - **Validates: Requirements 13.4**

  - [ ]* 9.4 Write unit tests for Memory Service Tool
    - Test embedding failure path, `memory_delete` not-found and access-denied errors, `top_k > stored count` returns all chunks, batch size cap at 50
    - _Requirements: 8.2, 8.6, 8.12, 8.11_


- [ ] 10. Real-Time Events Tool
  - [ ] 10.1 Implement `app/tools/realtime_tool.py`
    - Register `realtime_get_state` and `realtime_recent_events` in the Tool_Registry
    - `realtime_get_state`: read from `RedisCache`; include `as_of` UTC timestamp; fall back to service tool on Redis failure; return error "No live state for '{key}'" when absent
    - `realtime_recent_events`: read from `EventBuffer`, newest-first; default limit 20, hard max 100; return required fields per event
    - Return "Real-time features are disabled" error when `kafka` or `redis_cache` flags are absent
    - Tool is read-only; return error "Real-time events tool is read-only" on any write-style operation
    - Log each invocation at INFO (operation, key/topic, result count, correlation_id)
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9_

  - [ ]* 10.2 Write unit tests for Real-Time Events Tool
    - Test Redis unavailable fallback, feature-flag-disabled error, read-only guard, event buffer newest-first ordering
    - _Requirements: 18.5, 18.6, 18.7, 18.8_

- [ ] 11. Memory Manager and FCM Notifier services
  - [ ] 11.1 Implement `app/services/memory_manager.py`
    - Implement `MemoryManager.auto_search()`: called at turn start; prepend `## Relevant Memory` block to system prompt when chunks are found
    - Implement `MemoryManager.auto_store()`: called at turn end; store user + assistant messages via `memory_store_batch`; log `WARNING` on failure, never raise
    - _Requirements: 8.7, 8.8_

  - [ ] 11.2 Implement `app/services/fcm_notifier.py`
    - Implement `FCMNotifier` with firebase-admin SDK; 3 retries with exponential backoff (1s, 2s, 4s)
    - `send_task_completed()` and `send_plan_created()` helpers
    - On permanent failure set delivery_status = FAILED and log ERROR
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [ ]* 11.3 Write unit tests for FCM Notifier retry and failure behavior
    - Test 3-retry exhaustion sets FAILED status, skip when no device token, skip when WebSocket session is active
    - _Requirements: 11.2, 11.5, 11.6_


- [ ] 12. Checkpoint — all tools and services pass
  - Ensure all tests pass for tasks 6–11, ask the user if questions arise.

- [ ] 13. Planner Agent
  - [ ] 13.1 Implement `app/services/planner.py`
    - Implement `Subtask` and `ExecutionPlan` dataclasses
    - `PlannerAgent.classify_and_plan()`: LLM multi-step classification → plan generation → `_validate_plan()` → retry up to `max_retries` (default 2) → single-step fallback with WARNING log on exhaustion
    - Truncate to max 10 subtasks and log warning when LLM generates more
    - `_validate_plan()`: verify unique consecutive 1-based indices and no forward dependencies
    - Include full tool list in planning prompt
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.9, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [ ]* 13.2 Write property test for Planner Agent index uniqueness invariant (Property 3)
    - **Property 3: Planner Agent Index Uniqueness Invariant**
    - **Validates: Requirements 12.3, 12.7**

  - [ ]* 13.3 Write property test for Planner Agent forward-only dependencies (Property 4)
    - **Property 4: Planner Agent Forward-Only Dependencies**
    - **Validates: Requirements 12.4**

  - [ ]* 13.4 Write unit tests for Planner Agent
    - Test `_validate_plan()` rejects forward dependencies and duplicate indices, truncation at 10 subtasks, single-step fallback after retry exhaustion
    - _Requirements: 3.9, 12.4, 12.5, 12.6_

- [ ] 14. Observability Collector
  - [ ] 14.1 Implement `app/services/observability.py`
    - Implement `ObservabilityCollector` with `Counters` dataclass (ws_sessions_active, kafka_events_published, redis_hits, redis_misses, memory_chunks_stored, memory_searches)
    - `to_prometheus_text()` outputs valid Prometheus text format with `# HELP` and `# TYPE` lines
    - Module-level `observability` singleton
    - _Requirements: 16.5, 16.6_

  - [ ]* 14.2 Write unit tests for Observability Collector
    - Test counter increments and `to_prometheus_text()` format correctness
    - _Requirements: 16.5, 16.6_


- [ ] 15. Agent Gateway route and HTTP/WebSocket endpoints
  - [ ] 15.1 Implement `app/routes/agent.py` — HTTP and WebSocket endpoints
    - `POST /api/agent/chat`: validate schema (422), require auth (401), check rate limit (429), attach UUID v4 `Correlation_ID`, return `X-Correlation-ID` header, 60 s timeout → 504, orchestrator error → 503 (no stack traces)
    - `WS /api/agent/ws/{session_id}`: validate session token before handshake (reject with 401 / close code 4001 on invalid/missing), register with `WebSocketManager` (close 4003 at capacity), unregister on disconnect
    - Implement `_require_auth()`, `_check_rate_limit()`, `_validate_session_token()` helpers
    - Log all auth failures (source IP + endpoint)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_

  - [ ]* 15.2 Write property test for Correlation ID UUID v4 invariant (Property 11)
    - **Property 11: Correlation ID Is Always a Valid UUID v4**
    - **Validates: Requirements 1.8**

  - [ ]* 15.3 Write unit tests for Agent Gateway HTTP endpoint
    - Test 422 on missing `message`, 401 without auth, 503 on orchestrator error, 504 on timeout, `X-Correlation-ID` header present
    - _Requirements: 1.3, 1.7, 1.8, 1.9_

- [ ] 16. Implement health and metrics endpoints
  - [ ] 16.1 Implement `GET /api/agent/health` and `GET /api/agent/metrics` in `app/routes/agent.py`
    - Health check runs all component checks concurrently with 2 s timeout per check; marks timed-out components as `"timeout"` + `"degraded"`;  overall status is `"healthy"` only when all components are `"healthy"`, otherwise `"degraded"`
    - Metrics endpoint returns `observability.to_prometheus_text()` with correct content-type
    - _Requirements: 16.1, 16.2, 16.3, 16.6, 16.7_

  - [ ]* 16.2 Write unit tests for health endpoint logic
    - Test all-healthy → 200 `"healthy"`, one degraded → 200 `"degraded"`, component timeout → component marked timeout + overall degraded
    - _Requirements: 16.2, 16.3, 16.7_


- [ ] 17. Extend Agent Orchestrator with new pipeline
  - [ ] 17.1 Extend `app/services/agent.py` — Planner, ToolRouter, Redis, Kafka wiring
    - At turn start: `MemoryManager.auto_search()` → prepend `## Relevant Memory`; read `session:{session_id}` from Redis (fallback to DB on miss/error)
    - Publish `request_received` Kafka event
    - If `planner` flag enabled: call `PlannerAgent.classify_and_plan()`; if multi-step, execute subtasks in index order; inject completed subtask output into dependent subtask parameters; mark dependent subtasks skipped with reason on failure; send `plan_created` WebSocket message
    - Dispatch each tool call through `ToolRouter.route()`; publish `tool_started`, `tool_completed`, `tool_failed` Kafka events
    - Handle `Requires_Fresh_Data`: bypass Redis reads for involved entities; populate `fetched_at` and `source: "live"` on results; record `requires_fresh_data` in structured log
    - At turn end: `MemoryManager.auto_store()`; write updated session to Redis; publish `response_generated`; send FCM push if no active WebSocket session
    - Emit structured JSON log per request (correlation_id, session_id, conversation_id, planner_used, subtask_count, tool_calls_made, total_duration_ms, llm_provider, requires_fresh_data)
    - _Requirements: 3.5, 3.6, 3.7, 3.8, 3.10, 8.7, 8.8, 9.2, 9.3, 9.7, 9.10, 10.1, 10.3, 11.2, 11.3, 16.4, 17.3, 19.1, 19.8, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6_

  - [ ]* 17.2 Write property test for Tool-Mediated Access — no credentials in LLM prompts (Property 12)
    - **Property 12: Tool-Mediated Access — No Credentials in LLM Prompts**
    - **Validates: Requirements 19.1, 19.4**

  - [ ]* 17.3 Write integration tests for Agent Orchestrator pipeline
    - Test Redis unavailable → request succeeds via DB fallback; Kafka unavailable → request succeeds (event loss logged); `Requires_Fresh_Data` bypasses cache; subtask dependency injection; failed subtask marks dependent as skipped
    - _Requirements: 3.6, 3.7, 9.7, 20.1, 20.2_

- [ ] 18. WebSocket streaming integration
  - [ ] 18.1 Wire `WebSocketManager` into Agent Orchestrator streaming
    - Stream `{"type": "token", "content": "..."}` for each LLM token
    - Send `{"type": "done", "conversation_id": ..., "model": "...", "platform": "..."}` on completion
    - Send `{"type": "tool_start", "tool_name": "..."}` and `{"type": "tool_end", "tool_name": "...", "duration_ms": ...}` around each tool execution
    - Send `{"type": "plan_created", "subtask_count": N, "subtasks": [...]}` when Execution_Plan is produced
    - On error: send `{"type": "error", "message": "..."}` then close session regardless of send success
    - HTTP `/api/agent/chat` with active WebSocket → stream to WebSocket; no active WebSocket + < 100 sessions → SSE fallback; at/above 100 sessions → 503
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.10, 3.10_

  - [ ]* 18.2 Write unit tests for WebSocket message formats
    - Test `token`, `done`, `tool_start`, `tool_end`, `plan_created`, `error` JSON message shapes
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6_


- [ ] 19. Register tools and update `app/tools/__init__.py` and `app/main.py`
  - [ ] 19.1 Update `app/tools/__init__.py` to import all new tool modules
    - Add additive imports: `user_service`, `task_service`, `database_tool`, `memory_tool`, `realtime_tool`
    - Ensure all tools are discoverable by the existing `DecisionEngine` and `ToolRegistry`
    - _Requirements: 17.4_

  - [ ] 19.2 Extend `app/main.py` lifespan with new service startup/shutdown
    - Initialize and start `RedisCache`, `KafkaProducer`, `KafkaConsumer`, `WebSocketManager`, `FCMNotifier` based on feature flags
    - Fail fast (raise `RuntimeError`) if a feature-flagged service is unreachable at its default address
    - Log a descriptive warning for every missing env var before using the default
    - Include `agent` router (`app/routes/agent.py`) in the FastAPI app
    - Keep existing `/api/chat/*` endpoints untouched
    - _Requirements: 15.6, 17.1, 17.2, 17.3_

  - [ ]* 19.3 Write unit tests for lifespan and feature flag startup
    - Test flag-disabled services are not initialized, flag-enabled + unreachable service raises at startup
    - _Requirements: 15.6, 15.8_

- [ ] 20. Backward compatibility and live-data freshness audit
  - [ ] 20.1 Verify existing endpoints are unchanged and annotate live/cache source fields
    - Confirm `POST /api/chat/send` and `GET /api/chat/stream` continue to work and schema is unmodified
    - Ensure all `ToolResult.data` objects include `source: "live"` / `source: "cache"` and appropriate `fetched_at`/`age_seconds` fields where required
    - Ensure `DatabaseTool` and `RealTimeEventsTool` results always carry `source: "live"` and are never served from Redis
    - Confirm `degraded: true` is set when falling back to cached data after live-source failure
    - _Requirements: 17.1, 17.2, 20.2, 20.4, 20.5_

  - [ ]* 20.2 Write integration tests for live data freshness and backward compatibility
    - Test `source: "live"` on fresh fetch, `source: "cache"` + `age_seconds` on cache hit, `degraded: true` on source-unavailable fallback, existing SSE endpoint still streams
    - _Requirements: 20.2, 20.4, 20.5, 17.1, 17.2_

- [ ] 21. Property-based test file setup and smoke tests
  - [ ] 21.1 Create `tests/unit/test_properties.py` with hypothesis configuration
    - Set up `@settings(max_examples=100)` and pytest markers for all property tests (Properties 1–12)
    - Create `tests/helpers.py` with `create_mock_memory_tool` helper and other shared fixtures
    - _Requirements: all property test requirements_

  - [ ]* 21.2 Write smoke tests in `tests/smoke/test_agent_smoke.py`
    - `GET /api/agent/health` returns 200; `GET /api/agent/metrics` returns Prometheus text; all new tools discoverable via `tool_registry.get_enabled()`
    - _Requirements: 16.1, 16.6, 17.4_

- [ ] 22. Final checkpoint — all tests pass
  - Ensure all unit, property, integration, and smoke tests pass. Run `pytest -m "not integration"` at minimum. Ask the user if questions arise.


## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints at tasks 5, 12, and 22 provide incremental validation gates
- Property tests use `hypothesis` and target the 12 correctness properties in `design.md`
- Unit tests are complementary and cover example-based and error-path scenarios
- The implementation is purely additive: no existing files are deleted or replaced (only `app/main.py`, `app/config.py`, `app/models/db_models.py`, `app/models/schemas.py` receive additive edits)
- All new tool modules register via the `@tool_registry.tool(...)` decorator; `app/tools/__init__.py` imports them at startup
- Run property tests only: `pytest -m property --hypothesis-seed=0`
- Run all except integration: `pytest -m "not integration"`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "1.4", "1.5"] },
    { "id": 2, "tasks": ["2.1", "2.3", "2.4", "3.1", "4.1", "14.1"] },
    { "id": 3, "tasks": ["2.2", "2.5", "3.2", "4.2", "4.3", "4.4"] },
    { "id": 4, "tasks": ["6.1", "7.1", "8.1", "9.1", "10.1", "11.1", "11.2"] },
    { "id": 5, "tasks": ["6.2", "6.3", "7.2", "8.2", "8.3", "8.4", "9.2", "9.3", "9.4", "10.2", "11.3", "14.2"] },
    { "id": 6, "tasks": ["13.1"] },
    { "id": 7, "tasks": ["13.2", "13.3", "13.4"] },
    { "id": 8, "tasks": ["15.1", "16.1"] },
    { "id": 9, "tasks": ["15.2", "15.3", "16.2"] },
    { "id": 10, "tasks": ["17.1"] },
    { "id": 11, "tasks": ["17.2", "17.3", "18.1"] },
    { "id": 12, "tasks": ["18.2"] },
    { "id": 13, "tasks": ["19.1", "19.2"] },
    { "id": 14, "tasks": ["19.3", "20.1"] },
    { "id": 15, "tasks": ["20.2", "21.1"] },
    { "id": 16, "tasks": ["21.2"] }
  ]
}
```
