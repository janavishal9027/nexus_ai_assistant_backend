# Requirements Document

## Introduction

This document specifies requirements for **Intellectual Routing** — the Deep Research
answer-continuation strategy of the chatapp backend. When a user turns on Deep
Research, a single large model frequently cannot finish a big deliverable (for
example, a full project implementation) in one response: it runs out of output
tokens and stops, or it hits its provider rate limit part-way through. Today the
only recovery for a truncated ordinary chat is for the user to type "continue"
again and again, and a rate-limited model simply fails the request.

Intellectual Routing removes both of those failure modes **for Deep Research
mode only**. It treats the answer as one continuous piece of work relayed across
one or more large (≥400B-parameter) models: while the current model is healthy
it keeps going on its own; when it is truncated by its token cap, the same model
resumes; and when it rate-limits, stalls, or errors, the **next eligible model
receives the work produced so far, analyzes it, and continues exactly where the
previous model stopped** — without repeating anything. The user never types
"continue", and the request is never abandoned while another usable large model
remains. Every completed answer ends with a footer naming every model that
contributed, in the order they contributed.

This feature builds on the already-implemented multi-provider `Fallback_Router`,
the Deep-Research model-gating logic (`_is_deep_research_model`), the streaming
provider interface (`stream_chat_completion_ex` with `finish_reason`), the
`Rate_Limit_Service` (cooldowns, penalties, retryable-error detection), and the
`Agent_Orchestrator` (`agent.py`). It is entirely scoped to the Deep Research
path; ordinary chat routing is unchanged.

## Glossary

- **Deep_Research_Mode**: The per-request mode enabled by the frontend "Deep Research" toggle, carried to the backend as the `deep_research` boolean on the Chat_Request.
- **Intellectual_Routing**: The Deep-Research answer-continuation strategy defined by this spec — relaying one continuous deliverable across one or more large models with automatic token-cap continuation and cross-model hand-off.
- **Deep_Research_Model**: A chat-capable model whose parameter count is ≥ `_MIN_DEEP_RESEARCH_B` (400B). Only these models are eligible in Deep_Research_Mode.
- **Model_Pool**: The ordered list of Deep_Research_Models available under the current account's keys, sorted largest-first then by priority (`_get_ordered_models(..., deep_research=True)`).
- **Segment**: One continuous generation produced by a single model within a single Deep Research answer.
- **Relay**: The full sequence of Segments, across one or more models, that together form one Deep Research answer.
- **Continuation_Prompt**: The synthesized message set handed to a model that is resuming a Relay — the original request, the work-so-far as prior assistant content, and an instruction to resume exactly where it stopped without repeating (`_dr_continue_messages`).
- **Truncation**: A Segment that ended because the model hit its output token cap, signalled by `finish_reason == "length"`.
- **Hand_Off**: Switching the Relay from one model to a different model because the current model rate-limited, stalled, errored, or produced nothing.
- **Stall**: A model that has an open connection but produces no token for `_DR_STALL_TIMEOUT_S` seconds.
- **Models_Used_Footer**: The trailing note appended to a completed Deep Research answer listing every model that contributed, in order (`_dr_footer`).
- **Iteration_Cap**: The maximum number of Segments stitched into one answer (`DEEP_RESEARCH_MAX_ITERS`).
- **Wall_Clock_Budget**: The maximum total time a single Relay may run (`DEEP_RESEARCH_BUDGET_S`).
- **Rate_Limit_Service**: The existing `app/services/rate_limit.py` providing cooldowns, penalty tracking, and `is_retryable_error`.
- **Fallback_Router**: The existing `app/services/fallback_router.py` multi-provider routing layer that hosts Intellectual_Routing.
- **Agent_Orchestrator**: The existing `app/services/agent.py` that builds messages and dispatches to the router.
- **Deep_Research_Unavailable**: The condition where Deep_Research_Mode is requested but the Model_Pool is empty (no ≥400B model available under the current keys).

## Requirements

### Requirement 1: Deep-Research-Only Activation

**User Story:** As a user, I want intelligent multi-model continuation to apply only when I explicitly ask for Deep Research, so that ordinary quick chats stay fast and behave exactly as before.

#### Acceptance Criteria

1. WHEN a Chat_Request arrives with `deep_research == false`, THE Fallback_Router SHALL route it through the ordinary single-model routing path and SHALL NOT invoke Intellectual_Routing.
2. WHEN a Chat_Request arrives with `deep_research == true`, THE Agent_Orchestrator SHALL dispatch it to the Intellectual_Routing path on both the streaming and non-streaming entry points.
3. THE Intellectual_Routing path SHALL restrict its Model_Pool to Deep_Research_Models only, ignoring any specific `model` requested on the Chat_Request and ignoring the ordinary complexity-tier heuristic.
4. THE ordinary (non-Deep-Research) routing behavior SHALL remain unchanged by this feature.

### Requirement 2: Automatic Token-Cap Continuation (No "Continue")

**User Story:** As a user requesting a large deliverable, I want the answer to keep going by itself when a model hits its output-token limit, so that I never have to type "continue".

#### Acceptance Criteria

1. WHEN a Segment ends with Truncation (`finish_reason == "length"`) and the current model is still healthy, THE Fallback_Router SHALL continue the Relay using the SAME model with a Continuation_Prompt.
2. WHEN the Fallback_Router builds a Continuation_Prompt, it SHALL include the original request, the work produced so far as prior assistant content, and an explicit instruction to resume exactly where generation stopped without repeating, re-listing, or restating any content already produced.
3. WHEN the work-so-far exceeds `_DR_CONTEXT_CHARS`, THE Fallback_Router SHALL pass the most recent `_DR_CONTEXT_CHARS` characters of it to the continuing model.
4. THE Fallback_Router SHALL NOT require any user input between Segments of a single Relay.

### Requirement 3: Cross-Model Hand-Off on Failure

**User Story:** As a user, I don't want my Deep Research request to break when one model rate-limits or fails; I want another capable model to pick up the work and finish it.

#### Acceptance Criteria

1. WHEN the current model raises a retryable error (rate limit / quota / timeout) mid-Relay, THE Fallback_Router SHALL record a cooldown and rate-limit hit for that model+key via the Rate_Limit_Service and SHALL Hand_Off to the next eligible model in the Model_Pool.
2. WHEN the current model produces no token for `_DR_STALL_TIMEOUT_S` seconds (Stall), THE Fallback_Router SHALL treat it as a failure of that Segment and Hand_Off to the next eligible model.
3. WHEN a model's error indicates an authentication/authorization failure (401/403/unauthorized/forbidden/permission/invalid api key/authentication), THE Fallback_Router SHALL skip the entire platform for the remainder of the Relay, not just the one model.
4. WHEN a Hand_Off occurs, THE next model SHALL receive a Continuation_Prompt containing all work produced so far by every prior Segment, so it can analyze that work and resume from the exact stopping point.
5. WHEN a Segment produces partial text before failing, THE Fallback_Router SHALL retain that partial text in the accumulated work so the next model continues from it rather than discarding it.
6. WHEN a model is selected, THE Fallback_Router SHALL only consider models on platforms with an enabled, non-error key that is not currently on cooldown.
7. A model that has already failed or produced nothing within the current Relay SHALL NOT be retried within that same Relay.

### Requirement 4: Relay Termination and Safety Bounds

**User Story:** As an operator, I want a Relay to always terminate in bounded time and calls, so that a runaway build cannot loop forever or exhaust resources.

#### Acceptance Criteria

1. WHEN a Segment ends with a natural stop (`finish_reason` of `stop`, or an unknown/absent reason after producing text), THE Fallback_Router SHALL treat the deliverable as complete and end the Relay.
2. THE Fallback_Router SHALL stop the Relay after at most `DEEP_RESEARCH_MAX_ITERS` Segments.
3. THE Fallback_Router SHALL stop the Relay once `DEEP_RESEARCH_BUDGET_S` wall-clock seconds have elapsed.
4. WHEN no eligible model remains in the Model_Pool, THE Fallback_Router SHALL end the Relay with whatever work has been accumulated.
5. THE per-Segment token wait SHALL be bounded by `_DR_STALL_TIMEOUT_S` so a single hung model cannot block the Relay indefinitely.

### Requirement 5: Models-Used Attribution

**User Story:** As a user, I want to see which models actually produced my Deep Research answer, so that I understand and trust how it was assembled.

#### Acceptance Criteria

1. WHEN a Relay produces any content, THE Fallback_Router SHALL append a Models_Used_Footer to the answer.
2. WHEN exactly one model contributed, THE footer SHALL name that single model.
3. WHEN two or more models contributed, THE footer SHALL state the number of models and list their display names in contribution order.
4. THE footer SHALL credit a model at most once even if the Relay returned to it across multiple Segments, and SHALL list models in the order they first contributed.
5. A model that was tried but produced no usable text SHALL NOT appear in the footer.

### Requirement 6: Availability and Graceful Failure

**User Story:** As a user without a large-model provider, I want a clear, actionable message rather than a silent failure when Deep Research cannot run.

#### Acceptance Criteria

1. WHEN Deep_Research_Mode is requested and the Model_Pool is empty (Deep_Research_Unavailable), THE Fallback_Router SHALL raise a `DeepResearchUnavailableError` carrying a user-facing message that names the missing capability and suggests a concrete provider/model to add.
2. WHEN a Relay ends having produced no content because every large model was specifically rate-limited or marked as unavailable (not due to authentication failures or network issues), THE Fallback_Router SHALL emit a clear, non-technical message telling the user to try again shortly, rather than an empty answer or a raw exception.
3. THE Agent_Orchestrator SHALL surface the `DeepResearchUnavailableError` message to the user and SHALL NOT expose internal stack traces.

### Requirement 7: Behavioral Parity Across Streaming and Non-Streaming

**User Story:** As a client developer, I want Deep Research to continue across models identically whether I call the streaming or the non-streaming endpoint, so behavior is predictable.

#### Acceptance Criteria

1. THE streaming Deep Research entry point and the non-streaming Deep Research entry point SHALL share a single Relay implementation so their continuation and hand-off behavior is identical.
2. THE streaming entry point SHALL emit content incrementally as tokens arrive, SHALL emit a visible hand-off marker when it switches models mid-Relay, and SHALL emit the Models_Used_Footer at the end.
3. THE non-streaming entry point SHALL run the same Relay to completion and return one assembled answer with the Models_Used_Footer appended.
4. THE shared Relay implementation SHALL NOT hold cross-request state in module-level mutable variables; per-request inputs SHALL be passed as parameters so concurrent Deep Research requests cannot interfere.
