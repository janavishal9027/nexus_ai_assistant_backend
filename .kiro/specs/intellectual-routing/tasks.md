# Implementation Plan: Intellectual Routing (Deep Research Continuation)

## Overview

Deliver the Deep-Research Relay entirely within the existing `Fallback_Router`
and `Agent_Orchestrator`. The strategy: model-gating and the streaming Relay were
already present; this plan factors the proven Relay into a single shared core so
the non-streaming path relays identically, then wires the orchestrator and
verifies behavior. All changes are additive to existing files; ordinary chat
routing is untouched.

Checked boxes (`[x]`) are implemented in the current codebase.

## Tasks

- [x] 1. Deep-Research model gating and Model_Pool
  - [x] 1.1 `_is_deep_research_model` / `_param_billions` / `_KNOWN_PARAMS_B` classify chat-capable ≥400B models
    - Explicit `<n>b` token wins over the known-params map so distilled variants read correctly
    - _Requirements: 1.3, 3.6_
  - [x] 1.2 `_get_ordered_models(..., deep_research=True)` returns the largest-first, priority-tiebroken pool over platforms with live keys; ignores requested model and complexity tier
    - _Requirements: 1.3, 3.6_
  - [x] 1.3 `DeepResearchUnavailableError` with an actionable, provider-naming message
    - _Requirements: 6.1, 6.3_

- [x] 2. Continuation and attribution helpers
  - [x] 2.1 `_dr_continue_messages` builds the Continuation_Prompt (original request + work-so-far as assistant content + resume-without-repeating instruction), truncated to `_DR_CONTEXT_CHARS`
    - _Requirements: 2.2, 2.3, 3.4_
  - [x] 2.2 `_dr_footer` renders the Models_Used_Footer for 0 / 1 / ≥2 models
    - _Requirements: 5.1, 5.2, 5.3_
  - [x] 2.3 Relay safety constants: `DEEP_RESEARCH_MAX_ITERS`, `DEEP_RESEARCH_BUDGET_S`, `_DR_CONTEXT_CHARS`, `_DR_STALL_TIMEOUT_S`
    - _Requirements: 4.2, 4.3, 4.5_

- [x] 3. Streaming provider interface
  - [x] 3.1 `stream_chat_completion_ex` yields `content` deltas and a final `finish` event with `finish_reason`
    - OpenAI-compatible providers override to expose real `finish_reason`; base class default reports `None`
    - _Requirements: 2.1, 4.1_

- [x] 4. Shared Relay core — `_deep_research_relay(db, models, messages, temperature, max_tokens)`
  - [x] 4.1 Extract the segment loop into a shared async generator yielding `handoff` / `content` / `done` events, with all state as locals and `messages` passed as a parameter (no module-level cross-request state)
    - _Requirements: 7.1, 7.4_
  - [x] 4.2 Same-model continuation on `finish_reason == "length"`; natural stop ends the Relay
    - _Requirements: 2.1, 2.4, 4.1_
  - [x] 4.3 Hand-off on retryable error (cooldown + record hit), on stall (`_DR_STALL_TIMEOUT_S` per-token wait), and skip-whole-platform on auth failure; partial text retained in `accumulated`
    - _Requirements: 3.1, 3.2, 3.3, 3.5_
  - [x] 4.4 Model eligibility (live/enabled/non-error/non-cooled key) and no-retry-within-a-Relay via `skip_model_ids`
    - _Requirements: 3.6, 3.7_
  - [x] 4.5 Ordered, de-duplicated `models_used`; terminal `done` event carries `models_used` + `produced_any`
    - _Requirements: 5.4, 5.5_
  - [x] 4.6 Termination bounds: iteration cap, wall-clock budget, pool exhaustion
    - _Requirements: 4.2, 4.3, 4.4_

- [x] 5. Entry points
  - [x] 5.1 `route_deep_research_stream` → `StreamRouteResult`: streams `content`, renders `handoff` markers, appends `_dr_footer` / `_DR_EMPTY_MSG` on `done`
    - _Requirements: 5.1, 6.2, 7.2_
  - [x] 5.2 `route_deep_research` (non-streaming) → `RouteResult`: consumes the same relay to completion, joins content, appends `_dr_footer` / `_DR_EMPTY_MSG`
    - _Requirements: 7.1, 7.3_

- [x] 6. Agent_Orchestrator wiring
  - [x] 6.1 `agent_stream_chat` dispatches to `route_deep_research_stream` when `deep_research`
    - _Requirements: 1.2, 7.2_
  - [x] 6.2 `agent_chat` dispatches to `route_deep_research` when `deep_research` (previously used `route_chat(deep_research=True)`, which did not relay — the gap closed here)
    - _Requirements: 1.2, 7.3_
  - [x] 6.3 `DeepResearchUnavailableError` surfaced as user-facing text; generic errors never leak internals
    - _Requirements: 6.2, 6.3_

- [x] 7. Frontend activation (pre-existing)
  - [x] 7.1 "Deep Research" toggle sets `deep_research` on the Chat_Request (streaming + non-streaming API calls)
    - _Requirements: 1.1, 1.2_

- [x] 8. Verification
  - [x] 8.1 Relay smoke test: A truncates → same-model continue → A rate-limits → hand-off to B → B finishes → footer credits `[A, B]` in order
    - _Requirements: 2.1, 3.1, 3.4, 3.5, 5.3, 5.4_
  - [x] 8.2 Permanent unit test `tests/unit/test_deep_research_relay.py`: truncation-continue, rate-limit hand-off relaying accumulated work, stall hand-off, auth-skip-platform, all-fail empty message, empty-pool raise, and streaming/non-streaming parity — with scripted fake provider/DB (12 tests)
    - _Requirements: 2.1, 3.1, 3.2, 3.3, 3.5, 5.3, 5.4, 6.1, 6.2, 7.1, 7.3_
  - [x] 8.3 Integration test `tests/integration/test_deep_research_integration.py`: real `/api/chat/send` and `/api/chat/stream` with `deep_research=true` over in-memory SQLite + stubbed account + scripted provider (671B truncates→rate-limits, 405B continues); asserts both segments, the models-used footer, and no "continue" prompt
    - _Requirements: 1.2, 2.4, 5.1_

## Notes

- The only production code change required to satisfy this spec beyond what
  already existed was factoring the streaming loop into `_deep_research_relay`
  and adding `route_deep_research` for the non-streaming path (tasks 4.1, 5.2,
  6.2). All tasks are now complete: the behavior is locked in by the unit suite
  (task 8.2, 12 tests) and the endpoint integration tests (task 8.3, 2 tests),
  all passing.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "3.1", "7.1"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3"] },
    { "id": 2, "tasks": ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6"] },
    { "id": 3, "tasks": ["5.1", "5.2"] },
    { "id": 4, "tasks": ["6.1", "6.2", "6.3"] },
    { "id": 5, "tasks": ["8.1", "8.2", "8.3"] }
  ]
}
```
