# Implementation Plan: Agent-Based Real-Time Data Access System

## Overview

Implement an intelligent LLM-driven tool orchestration layer that replaces the current keyword-based web search. The system adds a Tool Registry, Decision Engine, Tool Executor, Token Budget Manager, and Citation Tracker, all integrated into the existing FastAPI/SQLAlchemy/agent orchestrator architecture.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create `app/tools/` directory and `__init__.py`
  - Create `tests/unit/`, `tests/integration/`, `tests/e2e/` directories with `__init__.py` files
  - Add `jsonschema` and `pytest-asyncio` to `requirements.txt` (if not already present)
  - Define `ToolDefinition`, `ToolCall`, `ToolResult`, `DecisionResult`, and `Citation` dataclasses in `app/services/tool_models.py`
  - Extend `ChatResponse` schema in `app/models/schemas.py` with `tool_calls_made: int = 0` and `tool_rounds: int = 0`
  - Extend `MessageDto` to support `role="tool"` (verify existing schema allows it)
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 11.8_

- [x] 2. Implement Tool Registry
  - [x] 2.1 Create `app/services/tool_registry.py` with `ToolRegistry` class and module-level `tool_registry` singleton
    - Implement `register(definition)` with duplicate-name check and JSON Schema validation via `jsonschema.Draft7Validator.check_schema`
    - Implement `tool()` decorator for registering functions as tools
    - Implement `get(name)`, `get_enabled()`, `enable(name)`, `disable(name)` methods
    - Raise `ValueError` with descriptive message on duplicate name or invalid schema
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 13.1, 13.6, 13.7_

  - [ ]* 2.2 Write unit tests for ToolRegistry
    - Test successful registration with valid schema
    - Test duplicate name raises `ValueError`
    - Test invalid JSON Schema raises `ValueError` with description
    - Test `enable`/`disable` update the enabled flag
    - Test `get_enabled()` returns only enabled tools
    - Test decorator registration creates `ToolDefinition` correctly
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8_

- [x] 3. Implement Token Budget Manager
  - [x] 3.1 Create `app/services/token_budget.py` with `TokenBudgetConfig` dataclass and `TokenBudgetManager` class
    - Implement `estimate_tokens(text)` using character-to-token ratio (default 4.0)
    - Implement `fits(results, current_messages)` to check if adding results exceeds budget
    - Implement `truncate(results)` prioritizing recent results and adding a truncation notice
    - Handle the edge case where truncation notice itself exceeds the budget (return all results unchanged)
    - Log warning when usage exceeds 80% threshold
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

  - [ ]* 3.2 Write unit tests for TokenBudgetManager
    - Test `estimate_tokens()` returns reasonable values
    - Test `fits()` correctly detects budget overflow
    - Test `truncate()` keeps recent results and prepends truncation notice
    - Test `truncate()` returns all results unchanged when truncation notice doesn't fit
    - Test 80% truncation threshold respected
    - _Requirements: 7.2, 7.3, 7.4, 7.7_

- [x] 4. Implement Citation Tracker
  - [x] 4.1 Create `app/services/citations.py` with `Citation` dataclass and `CitationTracker` class
    - Implement `ingest(results)` to extract source URLs from `ToolResult.sources` and deduplicate by URL
    - Implement `format_citations()` to produce a numbered Markdown list of references
    - Handle `ToolResult` entries with no `sources` field without raising errors
    - Preserve the `tool_name` that provided each citation
    - _Requirements: 8.1, 8.2, 8.4, 8.6, 8.7, 8.8_

  - [ ]* 4.2 Write unit tests for CitationTracker
    - Test `ingest()` extracts sources from multiple `ToolResult` objects
    - Test duplicate URLs are deduplicated
    - Test `format_citations()` generates a correctly numbered Markdown list
    - Test empty citations return empty string
    - Test `ToolResult` with no sources handled without error
    - _Requirements: 8.1, 8.2, 8.4, 8.6, 8.8_

- [x] 5. Checkpoint — Core utilities complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Tool Executor
  - [x] 6.1 Create `app/services/tool_executor.py` with `ToolExecutor` class
    - Implement `_execute_one(call)` with `jsonschema.validate` for input parameters and `asyncio.wait_for` for timeout enforcement
    - Return `ToolResult(status="timeout")` on `asyncio.TimeoutError`, `ToolResult(status="error")` on any other exception
    - Log each invocation with tool name, parameters, and execution duration
    - Wrap synchronous tool implementations using `asyncio.to_thread()`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 3.9, 14.1, 14.7_

  - [x] 6.2 Implement concurrent batch execution in `ToolExecutor`
    - Implement `execute_batch(tool_calls, max_concurrent)` using `asyncio.gather(return_exceptions=True)`
    - Return `ToolResult` list in same order as input `tool_calls` regardless of completion order
    - Ensure a failure in one tool does not block other tools in the batch
    - Implement cancellation propagation for both running and queued tools
    - Respect `max_concurrent` global limit
    - _Requirements: 14.2, 14.3, 14.4, 14.5, 14.6, 14.8, 3.7, 3.8_

  - [ ]* 6.3 Write unit tests for ToolExecutor
    - Test successful execution returns `status="success"` with correct data
    - Test tool timeout returns `status="timeout"` with error message
    - Test tool exception returns `status="error"` with exception message
    - Test invalid parameters fail schema validation before execution
    - Test concurrent batch execution preserves result order
    - Test partial failure in batch doesn't block other tools
    - Test sync tools are executed in thread pool
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 14.2, 14.5, 14.7_

- [x] 7. Implement HTTP API Tool Support in Tool Executor
  - [x] 7.1 Add HTTP API tool execution capability to `ToolExecutor`
    - Construct HTTP requests with endpoint, method, headers from tool definition metadata
    - Inject API keys from environment variables into API tool requests
    - Return error `ToolResult` for non-2xx HTTP status codes
    - Implement configurable retry policy for failed API requests
    - Log API request and response details for debugging
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7, 6.8_

  - [x] 7.2 Implement rate limiting for API tools
    - Enforce `rate_limit_rpm` from `ToolDefinition` configuration
    - _Requirements: 6.6_

- [x] 8. Implement Web Search Tool
  - [x] 8.1 Create `app/tools/web_search.py` with Tavily and DuckDuckGo implementations
    - Register as `web_search` tool via `@tool_registry.tool(...)` decorator
    - Use Tavily API (async via `httpx`) when `TAVILY_API_KEY` environment variable is set and valid
    - Fall back to DuckDuckGo (via `asyncio.to_thread`) when Tavily key is missing or call fails
    - Return results as list of `{"title": str, "snippet": str, "url": str}` dicts
    - Populate `ToolResult.sources` with `[{"url": ..., "title": ...}]` for citation tracking
    - Return `ToolResult(status="success", data={"results": [], "total_found": 0})` when search produces no results
    - Return `ToolResult(status="error", error_message="Web search unavailable")` when both providers fail
    - Limit results to configurable max (default 5)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [ ]* 8.2 Write unit and integration tests for web search tool
    - Test Tavily success path returns formatted results with sources
    - Test Tavily failure falls back to DuckDuckGo
    - Test both providers failing returns error `ToolResult`
    - Test empty results returns success with empty list
    - Test timeout is enforced
    - _Requirements: 5.2, 5.3, 5.7, 5.8_

- [x] 9. Implement Decision Engine
  - [x] 9.1 Create `app/services/decision.py` with `DecisionEngine` class
    - Build the tool selection prompt including current date/time, tool descriptions with schemas, and recent conversation
    - Make a non-streaming LLM call via `FallbackRouter` and parse the JSON response
    - Map valid `tool_calls` from JSON to `ToolCall` dataclass instances with UUID `call_id`
    - Log and filter out unrecognized tool names; preserve valid calls
    - Return `DecisionResult(tool_calls=[], proceed_without_tools=True)` on JSON parse failure
    - Return `DecisionResult(tool_calls=[], proceed_without_tools=True)` when no tools are available
    - Handle provider failover via the existing `FallbackRouter` mechanism
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

  - [ ]* 9.2 Write unit tests for DecisionEngine
    - Test valid LLM JSON response parsed into `ToolCall` objects
    - Test invalid JSON returns empty `tool_calls` with `proceed_without_tools=True`
    - Test unrecognized tool name is filtered out, valid calls preserved
    - Test empty available tool list returns `proceed_without_tools=True`
    - Test LLM "no tools needed" response returns empty `tool_calls`
    - Test fallback router provider failover is handled gracefully
    - _Requirements: 2.1, 2.3, 2.4, 2.6, 2.8_

- [x] 10. Checkpoint — All new service components complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Load tool configuration from `providers_config.json`
  - [x] 11.1 Extend `app/config.py` (or existing config loading in `agent.py`) to parse the `agent` and `tools` sections from `providers_config.json`
    - Load `tool_calling_enabled`, `max_tool_rounds`, `max_concurrent_tools`, `tool_timeout_seconds`, and `token_budget` settings
    - Load per-tool settings (`enabled`, `timeout_seconds`, `max_results`) from `tools` section
    - Apply per-tool enabled/disabled status to `tool_registry` on startup
    - Support `reload_config()` to update settings without restart
    - Log an error and use default settings if the configuration file is missing or invalid
    - _Requirements: 9.1, 9.2, 9.3, 9.6, 9.7, 9.8, 1.7_

  - [x] 11.2 Update `providers_config.json` with the `agent` and `tools` configuration sections
    - Add `agent.tool_calling_enabled`, `agent.max_tool_rounds`, `agent.max_concurrent_tools`, `agent.tool_timeout_seconds`, and `agent.token_budget` fields
    - Add `tools.web_search` configuration block with `enabled`, `provider`, `max_results`, and `timeout_seconds`
    - _Requirements: 9.1, 9.2_

- [x] 12. Integrate tool system into Agent Orchestrator
  - [x] 12.1 Enhance `app/services/agent.py` `agent_chat()` function with the multi-step tool orchestration loop
    - Import and wire `tool_registry`, `decision_engine`, `tool_executor`, `token_budget`, and `citation_tracker`
    - Add orchestration loop: call `decision_engine.decide()` → `tool_executor.execute_batch()` → append results → repeat up to `max_rounds`
    - Use `token_budget.fits()` / `token_budget.truncate()` before appending results to message context
    - Ingest results into `CitationTracker` and append formatted citations to final response content
    - Return enriched response dict including `tool_calls_made` and `tool_rounds` counts
    - Preserve existing conversation setup, `_build_agent_messages`, and `FallbackRouter` usage unchanged
    - Short-circuit the loop when `tool_calling_enabled` is False (existing behavior preserved)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.7, 11.8, 4.1, 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 8.3, 10.1, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [x] 12.2 Implement `_append_tool_results()` and `_format_tool_result()` helpers in `agent.py`
    - Format successful results as `[Tool: {name}] Status: success | Duration: {ms}ms\n{data_json}`
    - Format error/timeout results as `[Tool: {name}] Status: {status} | Error: {message}`
    - Append formatted content as `MessageDto(role="tool", content=...)` entries
    - Strip sensitive information from tool result data before adding to context
    - _Requirements: 15.1, 15.2, 15.3, 15.5, 15.6, 15.7, 15.8_

  - [x] 12.3 Enhance `agent_stream_chat()` in `agent.py` for streaming mode with tool support
    - Tool decision and execution phases run in non-streaming mode (existing behavior preserved for final stream)
    - After tool loop completes, resume streaming final LLM response
    - Append citation text as a final non-streaming chunk after stream ends
    - Handle `asyncio.CancelledError` for client disconnects gracefully
    - _Requirements: 11.5, 2.7, 4.5_

- [x] 13. Implement error handling and disabled-tool guard
  - [x] 13.1 Add error handling and fallback guard to orchestrator and decision engine
    - Wrap `decision_engine.decide()` call in try/except; on exception, log and set `proceed_without_tools=True`
    - Include error `ToolResult` context in the LLM prompt to allow graceful degradation
    - Never expose stack traces or internal error details in the user-facing response
    - Handle the case where all tool attempts fail by always generating a response from conversation history
    - Log tool failure rates for monitoring (per-tool success/failure counters)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.8_

  - [x] 13.2 Implement disabled-tool guard in `agent.py`
    - When `tool_calling_enabled` is False, skip the tool orchestration loop entirely
    - When a specifically disabled tool is referenced, return an appropriate error message in the response
    - _Requirements: 9.4, 9.5, 11.7_

- [x] 14. Implement structured logging and observability
  - Add per-tool invocation logging with timestamp, tool name, parameters, execution duration, and `conversation_id`
  - Log `DecisionEngine` decisions (tools selected, reasoning) at DEBUG level
  - Log token consumption per tool call
  - Log warning when tool execution exceeds 50% of timeout
  - Maintain success/failure counters per tool type
  - Include correlation IDs (UUIDs) for tracing multi-step tool sequences
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

- [ ] 15. Write integration and end-to-end tests
  - [ ]* 15.1 Write integration tests for the agent orchestrator tool loop
    - Test single tool call: web_search → LLM response with citation appended
    - Test multi-round tool calling: decision → tool → re-decision → second tool → final response
    - Test `max_rounds` limit stops the loop and generates response with available info
    - Test tool error → LLM sees error context → response includes limitation note
    - Test token budget exceeded → truncation applied → response generated
    - Test no tools available → proceeds directly to response
    - Test disabled tool excluded from available tools list
    - _Requirements: 4.1, 4.4, 4.5, 4.7, 7.3, 9.4, 10.1, 10.6_

  - [ ]* 15.2 Write end-to-end tests for complete agent flow
    - Mock LLM and external APIs to simulate realistic scenarios
    - Test: user asks about current event → web_search invoked → citation in response
    - Test: user asks factual question → no tools invoked → direct LLM response
    - Test: streaming mode → tool calls run non-streaming → streaming final response delivered with citations
    - _Requirements: 2.1, 2.8, 5.4, 5.9, 8.3, 8.5, 11.5_

- [x] 16. Final checkpoint — Full integration verified
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The design document explicitly states no property-based testing is appropriate for this feature (non-deterministic LLM outputs, external API calls, orchestration complexity). Unit tests and integration tests are used instead.
- All new service files go in `app/services/`; tool implementations go in `app/tools/`
- The existing `app/services/web_search.py` may be refactored into `app/tools/web_search.py` during task 8
- `ChatRequest` and `ChatResponse` schemas must remain backward-compatible (new fields default to 0)
- All tool I/O operations are async; synchronous tools are wrapped with `asyncio.to_thread()`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "4.2", "6.1"] },
    { "id": 3, "tasks": ["6.2", "6.3", "8.1", "9.1", "11.1"] },
    { "id": 4, "tasks": ["7.1", "7.2", "8.2", "9.2", "11.2"] },
    { "id": 5, "tasks": ["12.1", "12.2"] },
    { "id": 6, "tasks": ["12.3", "13.1", "13.2"] },
    { "id": 7, "tasks": ["14"] },
    { "id": 8, "tasks": ["15.1", "15.2"] }
  ]
}
```
