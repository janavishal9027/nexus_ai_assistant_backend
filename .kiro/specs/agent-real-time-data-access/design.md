# Design Document: Agent-Based Real-Time Data Access System

## Overview

This design document specifies a comprehensive agent architecture for real-time data access in the chatapp FastAPI backend. The system replaces the current keyword-based web search with an intelligent LLM-driven tool orchestration layer that determines when external data is needed, selects appropriate tools, executes them reliably, and integrates results into conversation context with proper citation tracking.

### Core Objectives

1. **Intelligent Tool Selection**: Use LLM reasoning to determine when and which tools to invoke based on user queries
2. **Extensible Tool Registry**: Provide a centralized, decorator-based system for registering and managing tools
3. **Robust Execution**: Execute tools asynchronously with timeout handling, retries, and graceful error recovery
4. **Multi-Step Reasoning**: Support iterative tool invocation where results from one tool inform subsequent tool selection
5. **Token Budget Management**: Track and enforce context window limits across tool descriptions, calls, and results
6. **Source Attribution**: Track and format citations from all tool-retrieved information
7. **Seamless Integration**: Preserve existing agent orchestrator, fallback router, and multi-provider LLM support

### High-Level Architecture

The system introduces four major components into the existing FastAPI/SQLAlchemy architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Endpoint Layer                     │
│                  (app/routes/chat.py)                        │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Agent Orchestrator (Enhanced)                  │
│            (app/services/agent.py)                          │
│  • Conversation management                                   │
│  • System prompt building                                    │
│  • Multi-step tool orchestration loop                        │
└────────┬────────────────────────┬──────────────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐    ┌──────────────────────────────────┐
│  Fallback Router │    │     Decision Engine (NEW)       │
│   (existing)     │    │   (app/services/decision.py)    │
│                  │    │  • Analyzes user intent with LLM │
│                  │    │  • Generates structured tool     │
│                  │    │    calls from LLM output         │
│                  │    │  • Handles both streaming and    │
│                  │    │    non-streaming modes           │
└──────────────────┘    └────────────┬─────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Tool Registry (NEW)                         │
│           (app/services/tool_registry.py)                   │
│  • Stores tool definitions with schemas                     │
│  • Validates tool schemas                                    │
│  • Provides enabled tools list                              │
│  • Supports decorator-based registration                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Tool Executor (NEW)                         │
│           (app/services/tool_executor.py)                   │
│  • Validates tool calls against schemas                     │
│  • Executes tools with timeout enforcement                  │
│  • Handles concurrent execution                             │
│  • Captures and structures results                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Supporting Components                          │
│  • Token Budget Manager (app/services/token_budget.py)      │
│  • Citation Tracker (app/services/citations.py)             │
│  • Tool implementations (app/tools/*)                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

Based on [research from protocol-agnostic tool management](https://arxiv.org/html/2507.10593) and [production AI agent patterns](https://markaicode.com/architecture/ai-agent-tool-calling-architecture/), this design prioritizes:

1. **Async-first execution**: All tool operations use Python's `asyncio` with proper timeout and cancellation handling
2. **Schema-driven validation**: JSON Schema for input/output validation reduces integration code and prevents malformed tool calls
3. **Separation of concerns**: Registry, decision, execution, and budget management are independent components
4. **Graceful degradation**: Tool failures never block response generation; errors are logged and surfaced to the LLM

## Architecture

### Component Interactions

The request flow through the system follows this sequence:

1. **Request Entry**: User sends `ChatRequest` to `/api/chat` or `/api/chat/stream`
2. **Orchestrator Initialization**: Agent orchestrator creates/loads conversation, saves user message to database
3. **Decision Phase**: Decision engine receives conversation history + tool registry, asks LLM if tools are needed
4. **Tool Execution Phase** (if tools requested): Tool executor validates and runs tools concurrently, captures results
5. **Iteration** (optional): Orchestrator appends tool results to context, repeats decision phase (up to max rounds)
6. **Final Response**: LLM generates response using conversation context + tool results
7. **Citation Integration**: Citation tracker formats sources, orchestrator appends to response
8. **Database Persistence**: Assistant message saved to database, conversation updated

### Async Execution Model

All I/O operations are async to prevent blocking:

- **Database queries**: Use existing synchronous SQLAlchemy session (executed in FastAPI dependency injection context)
- **LLM calls**: Async via existing provider implementations (httpx.AsyncClient)
- **Tool execution**: Async tool implementations with `asyncio.gather()` for concurrency
- **Web search**: Async Tavily API calls or sync DuckDuckGo calls wrapped in `asyncio.to_thread()`

Critical async patterns (based on [production asyncio guidance](https://andrewodendaal.com/python-async-programming-asyncio-patterns/)):

- Use `asyncio.wait_for()` with explicit timeout for all tool calls
- Use `asyncio.gather(return_exceptions=True)` for concurrent tool execution (partial failure tolerance)
- Properly propagate cancellation when request is cancelled
- Use `asyncio.to_thread()` for synchronous tool implementations (DuckDuckGo search, blocking I/O)

### Database Integration

The existing database layer uses synchronous SQLAlchemy 2.0 with connection pooling. Tool operations do NOT interact directly with the database:

- Tool execution results are held in memory during a conversation turn
- Only the agent orchestrator persists to database (user message, assistant message, conversation metadata)
- No async database session needed (simplifies migration)

Sequence for database operations:

```python
# In agent_chat() and agent_stream_chat():
# 1. Save user message (DB write)
user_msg = Message(conversation_id=conversation_id, role="user", content=request.message)
db.add(user_msg)
db.commit()

# 2. Load conversation history (DB read)
messages = db.query(Message).filter(...).all()

# 3-5. Decision + tool execution + iteration (NO DB operations)
# All tool results held in memory

# 6. Save assistant response (DB write)
assistant_msg = Message(conversation_id=conversation_id, role="assistant", content=result.content, ...)
db.add(assistant_msg)
db.commit()
```

### Configuration Management

Tool configuration extends `app/providers_config.json`:

```json
{
  "agent": {
    "system_prompt": "...",
    "max_context_messages": 20,
    "default_temperature": 0.7,
    "default_max_tokens": 4096,
    "web_search_enabled": true,
    "tool_calling_enabled": true,
    "max_tool_rounds": 3,
    "max_concurrent_tools": 5,
    "tool_timeout_seconds": 30,
    "token_budget": {
      "enabled": true,
      "max_tokens": 100000,
      "reserve_for_response": 4096,
      "truncation_threshold": 0.8
    }
  },
  "tools": {
    "web_search": {
      "enabled": true,
      "provider": "tavily",
      "max_results": 5,
      "timeout_seconds": 15
    }
  }
}
```

Configuration is loaded once at startup and can be reloaded via `reload_config()` (existing pattern from `agent.py`).


## Components and Interfaces

### 1. Tool Registry (`app/services/tool_registry.py`)

**Responsibility**: Centralized catalog of available tools. Manages registration, validation, enable/disable, and metadata retrieval.

```python
from dataclasses import dataclass, field
from typing import Callable, Any
import jsonschema

@dataclass
class ToolDefinition:
    name: str                           # Unique identifier
    description: str                    # Human-readable purpose for LLM prompt
    input_schema: dict                  # JSON Schema for parameters
    output_schema: dict                 # JSON Schema for results
    fn: Callable                        # The actual callable
    enabled: bool = True
    timeout_seconds: float = 30.0
    rate_limit_rpm: int | None = None
    requires_auth: bool = False
    examples: list[dict] = field(default_factory=list)

class ToolRegistry:
    _tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """Register a tool. Raises ValueError on duplicate name or invalid schema."""
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' already registered")
        self._validate_schema(definition.input_schema)
        self._validate_schema(definition.output_schema)
        self._tools[definition.name] = definition

    def tool(self, name: str, description: str, input_schema: dict, output_schema: dict, **kwargs):
        """Decorator for registering a function as a tool."""
        def decorator(fn: Callable):
            self.register(ToolDefinition(
                name=name, description=description,
                input_schema=input_schema, output_schema=output_schema,
                fn=fn, **kwargs
            ))
            return fn
        return decorator

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_enabled(self) -> list[ToolDefinition]:
        return [t for t in self._tools.values() if t.enabled]

    def enable(self, name: str) -> None: ...
    def disable(self, name: str) -> None: ...

    def _validate_schema(self, schema: dict) -> None:
        """Validate a dict is a valid JSON Schema. Raises jsonschema.SchemaError on failure."""
        jsonschema.Draft7Validator.check_schema(schema)

tool_registry = ToolRegistry()  # Module-level singleton
```

**Decorator Usage Example**:

```python
from app.services.tool_registry import tool_registry

@tool_registry.tool(
    name="web_search",
    description="Search the web for real-time information. Use when the user asks about current events, news, prices, weather, or anything requiring up-to-date data.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "total_found": {"type": "integer"}
        }
    },
    timeout_seconds=15.0
)
async def web_search_tool(query: str, max_results: int = 5) -> dict:
    ...
```


### 2. Decision Engine (`app/services/decision.py`)

**Responsibility**: Uses the LLM (via Fallback Router) to analyze whether tools are needed and generate structured tool calls.

```python
from dataclasses import dataclass
from typing import Any
from sqlalchemy.orm import Session
from ..models.schemas import MessageDto

@dataclass
class ToolCall:
    tool_name: str
    parameters: dict[str, Any]
    call_id: str  # UUID for correlation tracking

@dataclass
class DecisionResult:
    tool_calls: list[ToolCall]          # Empty = no tools needed
    reasoning: str                       # LLM's reasoning (for logging/debugging)
    proceed_without_tools: bool          # True when no tools available but LLM wanted them

class DecisionEngine:
    def __init__(self, registry: ToolRegistry, router: FallbackRouter): ...

    async def decide(
        self,
        db: Session,
        messages: list[MessageDto],
        available_tools: list[ToolDefinition],
        requested_model: str | None = None,
    ) -> DecisionResult:
        """
        Ask the LLM if tools are needed.
        Returns a DecisionResult with 0+ tool calls.
        """
```

**Decision Prompt Structure**:

The prompt sent to the LLM for tool selection uses this format:

```
You are a tool-calling assistant. Based on the conversation, determine if any tools are needed to answer the user's question.

Current date and time: {datetime.now().strftime("%A, %B %d, %Y %H:%M UTC")}

Available tools:
{for each tool in available_tools}
- {tool.name}: {tool.description}
  Parameters: {json.dumps(tool.input_schema["properties"])}

Conversation:
{recent_messages}

Respond with a JSON object in this exact format:
{
  "tools_needed": true|false,
  "reasoning": "brief explanation",
  "tool_calls": [
    {"tool": "tool_name", "parameters": {...}}
  ]
}

If no tools are needed, set tools_needed to false and tool_calls to [].
```

**Parsing Logic**: The decision engine parses the LLM's JSON response. If `tool_calls` contains an unrecognized tool name, it logs a warning and drops that call (does not raise). If JSON parsing fails entirely, it falls back to `proceed_without_tools=True`.

**Streaming Support**: The decision engine makes a single non-streaming call for the decision phase (JSON output is required). The subsequent final response generation continues to support streaming.


### 3. Tool Executor (`app/services/tool_executor.py`)

**Responsibility**: Validates tool calls against schemas, executes tools with timeout and concurrency management, captures results.

```python
from dataclasses import dataclass
from typing import Any
import asyncio
import logging

@dataclass
class ToolResult:
    call_id: str                        # Matches ToolCall.call_id
    tool_name: str
    status: str                         # "success" | "error" | "timeout"
    data: Any | None                    # Actual result data (if success)
    error_message: str | None           # Error details (if error/timeout)
    execution_time_ms: float
    sources: list[dict] | None = None   # URLs + metadata for citation tracking

class ToolExecutor:
    def __init__(self, registry: ToolRegistry): ...

    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        max_concurrent: int = 5,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls concurrently.
        Returns results in same order as tool_calls (preserves order).
        Uses asyncio.gather(return_exceptions=True) for partial failure tolerance.
        """
        tasks = [self._execute_one(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [self._handle_result(r) for r in results]

    async def _execute_one(self, call: ToolCall) -> ToolResult:
        """
        Execute a single tool call with timeout enforcement.
        """
        tool = self.registry.get(call.tool_name)
        if not tool:
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name,
                status="error", data=None,
                error_message=f"Tool '{call.tool_name}' not found",
                execution_time_ms=0)

        # Validate parameters against input schema
        try:
            jsonschema.validate(instance=call.parameters, schema=tool.input_schema)
        except jsonschema.ValidationError as e:
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name,
                status="error", data=None,
                error_message=f"Invalid parameters: {e.message}",
                execution_time_ms=0)

        # Execute with timeout
        start_time = asyncio.get_event_loop().time()
        try:
            result = await asyncio.wait_for(
                tool.fn(**call.parameters),
                timeout=tool.timeout_seconds
            )
            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name,
                status="success", data=result,
                error_message=None, execution_time_ms=execution_time_ms)
        except asyncio.TimeoutError:
            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name,
                status="timeout", data=None,
                error_message=f"Tool execution exceeded {tool.timeout_seconds}s timeout",
                execution_time_ms=execution_time_ms)
        except Exception as e:
            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name,
                status="error", data=None,
                error_message=str(e), execution_time_ms=execution_time_ms)
```

**Concurrency Control**: The `max_concurrent` parameter limits how many tools execute simultaneously. This prevents resource exhaustion when the decision engine requests many tool calls in one round.


### 4. Agent Orchestrator (Enhanced `app/services/agent.py`)

**Responsibility**: Multi-step orchestration loop integrating the existing agent flow with the new tool system.

The enhanced `agent_chat()` function replaces the current keyword-based web search with the intelligent tool loop:

```python
async def agent_chat(db: Session, request: ChatRequest) -> dict:
    config = get_config()
    agent_cfg = config.get("agent", {})

    # --- Existing conversation setup (unchanged) ---
    conversation_id = _ensure_conversation(db, request)
    _save_user_message(db, conversation_id, request.message)

    # --- New tool orchestration loop ---
    tool_calling_enabled = agent_cfg.get("tool_calling_enabled", True)
    max_rounds = agent_cfg.get("max_tool_rounds", 3)
    all_tool_results: list[ToolResult] = []
    citation_tracker = CitationTracker()

    messages = _build_agent_messages(db, conversation_id, request.message)

    if tool_calling_enabled:
        for round_num in range(max_rounds):
            enabled_tools = tool_registry.get_enabled()
            if not enabled_tools:
                break

            decision = await decision_engine.decide(db, messages, enabled_tools, request.model)

            if not decision.tool_calls:
                break  # LLM decided no tools needed

            results = await tool_executor.execute_batch(decision.tool_calls)
            all_tool_results.extend(results)
            citation_tracker.ingest(results)

            # Check token budget before adding to context
            if not token_budget.fits(results, messages):
                truncated = token_budget.truncate(results)
                messages = _append_tool_results(messages, truncated)
            else:
                messages = _append_tool_results(messages, results)

            # Next round: re-decide with updated context
            # If last round, will fall through to final response

    # --- Final response generation (uses existing Fallback Router) ---
    result = await route_chat(db=db, messages=messages, requested_model=request.model, ...)

    # Append citations to content if any sources were found
    citations_text = citation_tracker.format_citations()
    final_content = result.content + (f"\n\n{citations_text}" if citations_text else "")

    _save_assistant_message(db, conversation_id, final_content, result)
    return {..., "content": final_content, "tool_calls_made": len(all_tool_results)}
```

**Tool Result Formatting in Context**: Tool results are appended to the message list as a `tool` role message:

```python
def _append_tool_results(messages: list[MessageDto], results: list[ToolResult]) -> list[MessageDto]:
    for result in results:
        content = _format_tool_result(result)
        messages.append(MessageDto(role="tool", content=content))
    return messages

def _format_tool_result(result: ToolResult) -> str:
    if result.status == "success":
        data_str = json.dumps(result.data, indent=2) if isinstance(result.data, dict) else str(result.data)
        return (
            f"[Tool: {result.tool_name}] Status: success | "
            f"Duration: {result.execution_time_ms:.0f}ms\n{data_str}"
        )
    else:
        return (
            f"[Tool: {result.tool_name}] Status: {result.status} | "
            f"Error: {result.error_message}"
        )
```


### 5. Token Budget Manager (`app/services/token_budget.py`)

**Responsibility**: Track token consumption across tool descriptions, tool calls, and tool results. Enforce limits and truncate when necessary.

```python
from dataclasses import dataclass

@dataclass
class TokenBudgetConfig:
    enabled: bool = True
    max_tokens: int = 100_000
    reserve_for_response: int = 4096
    truncation_threshold: float = 0.8
    chars_per_token: float = 4.0  # Conservative estimate

class TokenBudgetManager:
    def __init__(self, config: TokenBudgetConfig): ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens from text using character-to-token ratio."""
        return int(len(text) / self.config.chars_per_token)

    def fits(self, results: list[ToolResult], current_messages: list[MessageDto]) -> bool:
        """Check if adding results would exceed token budget."""
        current_tokens = sum(self.estimate_tokens(m.content) for m in current_messages)
        result_tokens = sum(self.estimate_tokens(json.dumps(r.data)) for r in results if r.data)
        total = current_tokens + result_tokens + self.config.reserve_for_response
        return total < (self.config.max_tokens * self.config.truncation_threshold)

    def truncate(self, results: list[ToolResult]) -> list[ToolResult]:
        """
        Truncate tool results to fit budget.
        Prioritize recent results (keep last N that fit).
        If truncation notice itself doesn't fit, return all results unchanged.
        """
        available = int(self.config.max_tokens * self.config.truncation_threshold)
        truncation_notice = "[...earlier tool results truncated due to token budget...]"
        notice_tokens = self.estimate_tokens(truncation_notice)

        # Calculate how many recent results fit
        total_tokens = notice_tokens
        kept_results = []
        for result in reversed(results):
            result_tokens = self.estimate_tokens(json.dumps(result.data) if result.data else result.error_message or "")
            if total_tokens + result_tokens > available:
                break
            kept_results.insert(0, result)
            total_tokens += result_tokens

        # If we couldn't keep any results, return all (notice doesn't fit)
        if not kept_results:
            return results

        # Add truncation indicator
        first_result = kept_results[0]
        if first_result.status == "success" and isinstance(first_result.data, dict):
            first_result.data["_truncation_notice"] = truncation_notice

        return kept_results
```

**Token Estimation**: Uses a conservative character-to-token ratio (4.0) suitable for typical English text. This avoids dependency on tiktoken (OpenAI-specific tokenizer) and provides reasonable estimates across different models.


### 6. Citation Tracker (`app/services/citations.py`)

**Responsibility**: Extract source URLs from tool results, deduplicate, and format for appending to the response.

```python
from dataclasses import dataclass

@dataclass
class Citation:
    url: str
    title: str
    tool_name: str

class CitationTracker:
    def __init__(self):
        self.citations: dict[str, Citation] = {}  # url -> Citation

    def ingest(self, results: list[ToolResult]) -> None:
        """Extract citations from tool results. Deduplicates by URL."""
        for result in results:
            if result.status == "success" and result.sources:
                for source in result.sources:
                    url = source.get("url")
                    if url and url not in self.citations:
                        self.citations[url] = Citation(
                            url=url,
                            title=source.get("title", url),
                            tool_name=result.tool_name
                        )

    def format_citations(self) -> str:
        """Format citations as numbered references."""
        if not self.citations:
            return ""

        lines = ["## Sources"]
        for i, (url, citation) in enumerate(self.citations.items(), start=1):
            lines.append(f"{i}. [{citation.title}]({url})")

        return "\n".join(lines)

    def get_count(self) -> int:
        return len(self.citations)
```

**Source Extraction from Tool Results**: Tools that retrieve external data (web search, API calls) should populate the `sources` field in ToolResult:

```python
# Example from web_search tool
result = ToolResult(
    ...,
    status="success",
    data={"results": [...], "total_found": 5},
    sources=[
        {"url": "https://example.com/article1", "title": "Article Title"},
        {"url": "https://example.com/article2", "title": "Another Article"},
    ]
)
```


## Data Models

This section defines the core data structures used throughout the agent tool system, including new models for tool management and enhancements to existing conversation models.

### New Data Models

#### ToolDefinition

Represents a registered tool with its metadata, schemas, and configuration.

```python
@dataclass
class ToolDefinition:
    name: str                           # Unique identifier (e.g., "web_search")
    description: str                    # Human-readable purpose for LLM prompt
    input_schema: dict                  # JSON Schema for parameters
    output_schema: dict                 # JSON Schema for results
    fn: Callable                        # The actual async or sync callable
    enabled: bool = True                # Runtime enable/disable flag
    timeout_seconds: float = 30.0       # Max execution time
    rate_limit_rpm: int | None = None   # Rate limit (requests per minute)
    requires_auth: bool = False         # Whether tool needs API credentials
    examples: list[dict] = field(default_factory=list)  # Example inputs for LLM
```

**Field Constraints**:
- `name`: Must be unique across registry, lowercase with underscores
- `input_schema` and `output_schema`: Must be valid JSON Schema Draft 7
- `timeout_seconds`: Must be > 0, default 30.0
- `rate_limit_rpm`: Optional, enforced by rate limiter if set


#### ToolCall

Represents a request from the LLM to execute a specific tool.

```python
@dataclass
class ToolCall:
    tool_name: str                      # References ToolDefinition.name
    parameters: dict[str, Any]          # Must conform to tool's input_schema
    call_id: str                        # UUID for correlation tracking (generated)
```

**Lifecycle**: Created by DecisionEngine → validated by ToolExecutor → executed → produces ToolResult


#### ToolResult

Represents the outcome of a tool execution (success, error, or timeout).

```python
@dataclass
class ToolResult:
    call_id: str                        # Matches ToolCall.call_id
    tool_name: str                      # Tool that was executed
    status: str                         # "success" | "error" | "timeout"
    data: Any | None                    # Actual result data (if success)
    error_message: str | None           # Error details (if error/timeout)
    execution_time_ms: float            # Duration in milliseconds
    sources: list[dict] | None = None   # URLs + metadata for citation tracking
```

**Status Values**:
- `"success"`: Tool executed successfully, `data` contains result
- `"error"`: Tool raised exception, `error_message` contains details
- `"timeout"`: Execution exceeded `timeout_seconds`, operation cancelled

**Sources Structure** (for citation tracking):
```python
sources = [
    {"url": "https://...", "title": "...", "snippet": "..."},
    ...
]
```


#### DecisionResult

Represents the LLM's decision about which tools (if any) to invoke.

```python
@dataclass
class DecisionResult:
    tool_calls: list[ToolCall]          # Empty list = no tools needed
    reasoning: str                       # LLM's reasoning (for logging/debugging)
    proceed_without_tools: bool          # True when LLM wanted tools but none available
```

**Decision Logic**:
- `len(tool_calls) == 0` and `proceed_without_tools == False`: LLM decided no tools needed
- `len(tool_calls) == 0` and `proceed_without_tools == True`: LLM wanted tools but registry empty or JSON parse failed
- `len(tool_calls) > 0`: LLM requested specific tools


#### Citation

Represents a single source reference extracted from tool results.

```python
@dataclass
class Citation:
    url: str                            # Full URL to source
    title: str                          # Human-readable title
    tool_name: str                      # Tool that provided this source
```

**Uniqueness**: Citations are deduplicated by URL (CitationTracker maintains `dict[str, Citation]`)


### Enhanced Existing Models

#### MessageDto (Extended)

The existing `MessageDto` schema is extended to support tool messages in the conversation context:

```python
class MessageDto(BaseModel):
    role: str                           # "user" | "assistant" | "system" | "tool"
    content: str                        # Message content or formatted tool result
    model: Optional[str] = None
    platform: Optional[str] = None
```

**New Role Type**: `"tool"` - represents a tool execution result in the conversation history. Content is formatted as:
```
[Tool: {tool_name}] Status: {status} | Duration: {ms}ms
{data or error_message}
```


#### ChatRequest (Unchanged)

The existing `ChatRequest` schema requires no changes. Tool calling is transparent to the client:

```python
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    history: Optional[list[MessageDto]] = None
```


#### ChatResponse (Extended)

The existing `ChatResponse` schema is extended to include tool call metadata:

```python
class ChatResponse(BaseModel):
    conversation_id: Optional[int] = None
    content: str                        # Includes citations appended
    model: Optional[str] = None
    platform: Optional[str] = None
    fallback_attempts: int = 0
    tool_calls_made: int = 0            # NEW: Total tool calls in this turn
    tool_rounds: int = 0                # NEW: Number of decision-execution rounds
```

**Backward Compatibility**: New fields have default values (0), so existing clients continue to work.


### Configuration Models

#### TokenBudgetConfig

Configuration for token budget enforcement.

```python
@dataclass
class TokenBudgetConfig:
    enabled: bool = True                # Enable/disable budget management
    max_tokens: int = 100_000           # Maximum total tokens
    reserve_for_response: int = 4096    # Tokens reserved for final response
    truncation_threshold: float = 0.8   # Truncate if usage > threshold * max
    chars_per_token: float = 4.0        # Estimation ratio
```

**Source**: Loaded from `providers_config.json` under `agent.token_budget`


#### ToolConfig

Per-tool configuration (loaded from `tools_config.json`):

```python
{
  "tools": {
    "web_search": {
      "enabled": true,
      "provider": "tavily",           // "tavily" | "duckduckgo"
      "max_results": 5,
      "timeout_seconds": 15
    }
  }
}
```


### Database Models (Unchanged)

The existing SQLAlchemy models (`Conversation`, `Message`, `ApiKey`, `ChatModel`) require no schema changes. Tool results are not persisted directly to the database; they exist only in memory during a conversation turn. The final assistant response (which may include citations from tool results) is saved to the `messages` table as usual.

**Future Extension**: If tool call persistence is needed for analytics, add optional fields to `Message`:
- `tool_calls: JSON` - Array of tool call metadata
- `tool_execution_time_ms: Integer` - Total tool execution time


### Data Flow Summary

```
ChatRequest (client)
    ↓
Conversation + Messages (DB read)
    ↓
[Tool Orchestration Loop]
    ↓
DecisionResult → list[ToolCall]
    ↓
list[ToolResult] → CitationTracker
    ↓
[Repeat or exit]
    ↓
Final ChatResponse (with citations)
    ↓
Message (DB write)
```

All tool-related data structures (`ToolCall`, `ToolResult`, `DecisionResult`, `Citation`) are ephemeral—they exist only in memory during request processing and are not persisted.


## Correctness Properties

*This feature is not suitable for property-based testing due to its nature as an integration-heavy orchestration system. The system involves external API calls, LLM invocations, database operations, and complex state management with non-deterministic LLM outputs. Instead, correctness is verified through unit tests (for individual components), integration tests (for tool execution and orchestration), and end-to-end tests (for the complete agent flow).*

The testing strategy below outlines the appropriate testing approaches for each component.


## Error Handling

This section defines how the system handles errors at each layer, ensuring graceful degradation and user-friendly error messages.

### Error Categories

#### 1. Tool Registry Errors

**Scenario**: Invalid tool registration (duplicate name, invalid schema, missing required fields)

**Handling**:
- Raise `ValueError` with descriptive message at registration time (startup or decorator invocation)
- Log error with full stack trace
- System startup fails if core tools (web_search) cannot register
- Do NOT silently ignore registration errors

**Example**:
```python
try:
    tool_registry.register(tool_def)
except ValueError as e:
    logger.error(f"Tool registration failed: {e}")
    raise  # Fail fast at startup
```


#### 2. Decision Engine Errors

**Scenario**: LLM returns invalid JSON, unrecognized tool name, or fails to respond

**Handling**:
- **Invalid JSON**: Log warning, set `proceed_without_tools=True`, return empty `tool_calls`
- **Unrecognized tool name**: Log warning, filter out invalid tool call, continue with valid ones
- **LLM failure**: Fallback router handles provider failover; if all providers fail, return empty `tool_calls` (proceed without tools)
- **Never expose internal errors to user**: LLM receives sanitized error context

**Example**:
```python
try:
    decision = await decision_engine.decide(...)
except Exception as e:
    logger.error(f"Decision engine error: {e}")
    # Graceful fallback: proceed without tools
    decision = DecisionResult(tool_calls=[], reasoning="error", proceed_without_tools=True)
```


#### 3. Tool Execution Errors

**Scenario**: Tool times out, raises exception, or returns invalid output

**Handling**:
- **Timeout**: Cancel async task, return `ToolResult(status="timeout", error_message="...")`
- **Exception**: Catch all exceptions, return `ToolResult(status="error", error_message=str(e))`
- **Invalid output schema**: Treat as error, log schema validation failure
- **Concurrent execution**: Use `asyncio.gather(return_exceptions=True)` so one failure doesn't block others
- **Error ToolResults are passed to LLM**: Decision engine sees error context and can retry with different tool or proceed with partial info

**Example**:
```python
try:
    result = await asyncio.wait_for(tool.fn(**params), timeout=tool.timeout_seconds)
except asyncio.TimeoutError:
    return ToolResult(status="timeout", error_message=f"Exceeded {tool.timeout_seconds}s")
except Exception as e:
    logger.exception(f"Tool {tool.name} failed")
    return ToolResult(status="error", error_message=str(e))
```


#### 4. Token Budget Errors

**Scenario**: Tool results exceed token budget, truncation required

**Handling**:
- **Soft limit (80% threshold)**: Log warning, continue without truncation
- **Hard limit (budget exhausted)**: Truncate oldest tool results, add truncation notice
- **Truncation notice doesn't fit**: Return all results unchanged (no truncation)
- **Never fail request due to token budget**: Truncation is best-effort optimization

**Example**:
```python
if not token_budget.fits(results, messages):
    logger.warning(f"Token budget exceeded, truncating {len(results)} tool results")
    results = token_budget.truncate(results)
```


#### 5. Web Search Errors

**Scenario**: Tavily API key invalid, network failure, DuckDuckGo scraping blocked

**Handling**:
- **Tavily failure + no API key configured**: Silently fall back to DuckDuckGo
- **Tavily failure + API key configured**: Log warning, fall back to DuckDuckGo
- **Both Tavily and DuckDuckGo fail**: Return `ToolResult(status="error", error_message="Web search unavailable")`
- **Empty results (no error)**: Return `ToolResult(status="success", data={"results": [], "total_found": 0})`

**Example**:
```python
try:
    results = await tavily_search(query)
except TavilyError as e:
    logger.warning(f"Tavily failed: {e}, falling back to DuckDuckGo")
    try:
        results = await duckduckgo_search(query)
    except Exception as e2:
        return ToolResult(status="error", error_message="Web search unavailable")
```


#### 6. Database Errors

**Scenario**: Conversation not found, message save fails, connection timeout

**Handling**:
- **Conversation not found**: Create new conversation automatically (existing behavior)
- **Message save failure**: Raise exception, return 500 to client (critical failure—response generated but not persisted)
- **Connection pool exhausted**: FastAPI handles via dependency injection; request times out at HTTP layer

**Example**:
```python
try:
    db.add(assistant_msg)
    db.commit()
except Exception as e:
    logger.error(f"Failed to save message: {e}")
    db.rollback()
    raise HTTPException(status_code=500, detail="Failed to save conversation")
```


#### 7. Streaming Errors

**Scenario**: Client disconnects mid-stream, LLM stream fails partway

**Handling**:
- **Client disconnect**: Catch `asyncio.CancelledError`, log gracefully, stop streaming
- **LLM stream fails**: Send `StreamChunk(error="...", done=True)` as final chunk
- **Tool execution during stream**: Tool calls made in non-streaming mode (decision phase), streaming resumes for final response

**Example**:
```python
try:
    async for chunk in llm_stream():
        yield StreamChunk(content=chunk)
except asyncio.CancelledError:
    logger.info("Client disconnected during stream")
    return
except Exception as e:
    yield StreamChunk(error=str(e), done=True)
```


### Error Logging Strategy

All errors are logged with structured context:

```python
logger.error(
    "Tool execution failed",
    extra={
        "tool_name": tool_name,
        "call_id": call_id,
        "conversation_id": conversation_id,
        "execution_time_ms": execution_time_ms,
        "error": str(e),
    }
)
```

**Sensitive Data**: Never log API keys, full API responses, or user message content (except in DEBUG mode with explicit configuration).


### User-Facing Error Messages

Errors presented to users are sanitized:

- ❌ **Bad**: `"ToolResult(status='error', error_message='Tavily API key invalid: 401 Unauthorized')"`
- ✅ **Good**: `"I wasn't able to search the web right now, but I can answer based on my training data."`

The LLM receives error context (so it can explain limitations), but internal error details are removed before inclusion in the response.


## Testing Strategy

This section outlines the testing approach for verifying system correctness, including unit tests, integration tests, and end-to-end tests.

### Testing Approach

Given the integration-heavy nature of this system (external APIs, LLM calls, database operations, async execution), we employ a **layered testing strategy**:

1. **Unit Tests**: Test individual components in isolation with mocks
2. **Integration Tests**: Test component interactions with real implementations (but mocked external services)
3. **End-to-End Tests**: Test complete user flows with mocked LLM/API responses

**No Property-Based Testing**: This feature is not suitable for PBT due to non-deterministic LLM outputs, side effects (database writes, API calls), and orchestration complexity. Instead, we use example-based tests with representative scenarios and edge cases.


### Test Coverage Requirements

| Component | Unit Test Coverage | Integration Test Coverage |
|-----------|-------------------|---------------------------|
| ToolRegistry | 100% (all methods) | N/A |
| DecisionEngine | 90%+ | JSON parsing, fallback logic |
| ToolExecutor | 95%+ | Timeout, concurrency, error handling |
| TokenBudgetManager | 100% | Truncation edge cases |
| CitationTracker | 100% | Deduplication logic |
| AgentOrchestrator | 80%+ | Multi-step tool loop, database integration |
| Web Search Tool | 85%+ | Tavily/DuckDuckGo fallback |


### Unit Tests

**Tool Registry** (`tests/unit/test_tool_registry.py`):
- ✅ Register tool with valid schema succeeds
- ✅ Register tool with duplicate name raises ValueError
- ✅ Register tool with invalid JSON schema raises ValueError
- ✅ Enable/disable tool updates enabled flag
- ✅ get_enabled() returns only enabled tools
- ✅ Decorator registration creates ToolDefinition correctly

**Decision Engine** (`tests/unit/test_decision_engine.py`):
- ✅ Valid LLM JSON response parsed into ToolCalls
- ✅ Invalid JSON response returns empty tool_calls with proceed_without_tools=True
- ✅ Unrecognized tool name filtered out, valid calls preserved
- ✅ Empty tool list returns proceed_without_tools=True
- ✅ LLM decides no tools needed returns empty tool_calls
- ✅ Fallback router provider failover handled gracefully

**Tool Executor** (`tests/unit/test_tool_executor.py`):
- ✅ Successful tool execution returns status="success" with data
- ✅ Tool timeout returns status="timeout" with error message
- ✅ Tool exception returns status="error" with exception message
- ✅ Invalid parameters fail schema validation before execution
- ✅ Concurrent execution with asyncio.gather preserves order
- ✅ Partial failures in batch don't block other tools
- ✅ Cancellation propagates to running and queued tools
- ✅ Sync tools executed in thread pool (asyncio.to_thread)

**Token Budget Manager** (`tests/unit/test_token_budget.py`):
- ✅ estimate_tokens() returns reasonable token count
- ✅ fits() correctly detects budget overflow
- ✅ truncate() keeps recent results and adds notice
- ✅ truncate() returns all results if notice doesn't fit
- ✅ Truncation threshold (80%) respected

**Citation Tracker** (`tests/unit/test_citations.py`):
- ✅ ingest() extracts sources from ToolResults
- ✅ Duplicate URLs deduplicated
- ✅ format_citations() generates numbered Markdown list
- ✅ Empty citations returns empty string


### Integration Tests

**Agent Orchestrator Tool Loop** (`tests/integration/test_agent_tool_loop.py`):
- ✅ Single tool call: web_search → LLM response with citation
- ✅ Multi-round tool calling: web_search → decision → API call → response
- ✅ Max rounds limit enforced (stops after 3 rounds)
- ✅ Tool error → LLM sees error context → generates response with limitation note
- ✅ Token budget exceeded → truncation applied → response generated
- ✅ No tools available → proceeds directly to response
- ✅ Tool disabled → excluded from available tools list

**Web Search Tool** (`tests/integration/test_web_search.py`):
- ✅ Tavily API success returns formatted results with sources
- ✅ Tavily failure falls back to DuckDuckGo
- ✅ Both Tavily and DuckDuckGo fail returns error ToolResult
- ✅ Empty search results returns success with empty list
- ✅ Timeout enforced for slow searches

**Decision Engine + Fallback Router** (`tests/integration/test_decision_fallback.py`):
- ✅ Primary LLM provider failure triggers fallback
- ✅ All providers fail → empty tool_calls returned
- ✅ Streaming mode decision phase uses non-streaming call


### End-to-End Tests

**Complete Agent Flow** (`tests/e2e/test_agent_e2e.py`):

Mock the LLM and external APIs to simulate realistic scenarios:

- ✅ **User asks about current event** → Decision engine requests web_search → Tool executed → LLM generates response with citation
- ✅ **User asks factual question** → Decision engine decides no tools needed → Direct LLM response
- ✅ **Web search fails** → Error ToolResult → LLM generates response noting limitation
- ✅ **Multi-step reasoning** → User asks complex question → Tool call 1 → Tool call 2 → Final response
- ✅ **Token budget exceeded** → Large tool results truncated → Response includes truncation notice
- ✅ **Streaming mode** → Tool calls made → Streaming response delivered → Citations appended at end

**Test Data**:
```python
# Example mocked LLM decision response
mock_llm_decision = {
    "tools_needed": True,
    "reasoning": "User asked about current weather, need real-time data",
    "tool_calls": [
        {"tool": "web_search", "parameters": {"query": "weather in Seattle"}}
    ]
}

# Example mocked web search response
mock_search_result = {
    "results": [
        {"title": "Seattle Weather", "url": "https://weather.com/seattle", "snippet": "..."}
    ],
    "total_found": 1
}
```


### Test Execution

**Run all tests**:
```bash
pytest tests/ --cov=app/services --cov-report=html
```

**Run specific test category**:
```bash
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/e2e/           # End-to-end tests only
```

**Continuous Integration**:
- Run full test suite on every commit (GitHub Actions / GitLab CI)
- Fail build if coverage drops below 85%
- Run integration tests against staging environment before deployment


### Manual Testing Checklist

Before deployment, manually verify:

- [ ] Web search works with valid Tavily API key
- [ ] Web search falls back to DuckDuckGo when Tavily key missing
- [ ] Multi-round tool calling completes successfully
- [ ] Citations appear in response for web search results
- [ ] Streaming mode delivers chunks correctly with citations at end
- [ ] Tool timeout enforced (simulate slow tool with `asyncio.sleep(60)`)
- [ ] Error responses user-friendly (no stack traces exposed)
- [ ] Database persistence: tool results don't appear in message table, only final response


### Test Fixtures

**Mock Tool** (for testing executor):
```python
@pytest.fixture
def mock_tool():
    async def mock_fn(param: str) -> dict:
        await asyncio.sleep(0.1)  # Simulate I/O
        return {"result": f"processed {param}"}
    
    return ToolDefinition(
        name="mock_tool",
        description="A mock tool for testing",
        input_schema={"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]},
        output_schema={"type": "object"},
        fn=mock_fn,
        timeout_seconds=5.0
    )
```

**Mock LLM Decision** (for testing decision engine):
```python
@pytest.fixture
def mock_llm_response(mocker):
    return mocker.patch(
        "app.services.fallback_router.route_chat",
        return_value=ChatResponse(
            content='{"tools_needed": true, "tool_calls": [{"tool": "web_search", "parameters": {"query": "test"}}]}',
            model="mock-model",
            platform="mock-platform"
        )
    )
```


### Performance Testing

**Load Test Scenarios**:
1. **Concurrent tool calls**: 50 simultaneous requests, each triggering 2-3 tool calls
2. **Token budget stress**: Requests generating tool results exceeding 100k tokens
3. **Timeout stress**: 20% of tool calls configured to timeout, verify system stability

**Metrics to track**:
- Average tool execution time (target: <2s for web_search)
- P95 total request time including tool calls (target: <5s)
- Token budget truncation frequency (target: <5% of requests)
- Tool failure rate (target: <1% under normal conditions)

