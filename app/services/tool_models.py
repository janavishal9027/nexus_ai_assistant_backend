"""
Data models for the agent tool system.

This module defines the core data structures used throughout the tool orchestration
system, including tool definitions, tool calls, tool results, decision results, and citations.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """
    Represents a registered tool with its metadata, schemas, and configuration.
    
    Attributes:
        name: Unique identifier (e.g., "web_search")
        description: Human-readable purpose for LLM prompt
        input_schema: JSON Schema for parameters (must be valid JSON Schema Draft 7)
        output_schema: JSON Schema for results (must be valid JSON Schema Draft 7)
        fn: The actual async or sync callable
        enabled: Runtime enable/disable flag
        timeout_seconds: Max execution time (must be > 0)
        rate_limit_rpm: Rate limit (requests per minute, optional)
        requires_auth: Whether tool needs API credentials
        examples: Example inputs for LLM
    """
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    fn: Callable
    enabled: bool = True
    timeout_seconds: float = 30.0
    rate_limit_rpm: int | None = None
    requires_auth: bool = False
    examples: list[dict] = field(default_factory=list)


@dataclass
class ToolCall:
    """
    Represents a request from the LLM to execute a specific tool.
    
    Attributes:
        tool_name: References ToolDefinition.name
        parameters: Must conform to tool's input_schema
        call_id: UUID for correlation tracking (generated)
    
    Lifecycle: Created by DecisionEngine → validated by ToolExecutor → executed → produces ToolResult
    """
    tool_name: str
    parameters: dict[str, Any]
    call_id: str
    # Correlation ID threaded through the Tool_Router so every downstream
    # call/result can be traced to its originating request (req 4.10).
    correlation_id: str | None = None


@dataclass
class ToolResult:
    """
    Represents the outcome of a tool execution (success, error, or timeout).
    
    Attributes:
        call_id: Matches ToolCall.call_id
        tool_name: Tool that was executed
        status: "success" | "error" | "timeout"
        data: Actual result data (if success)
        error_message: Error details (if error/timeout)
        execution_time_ms: Duration in milliseconds
        sources: URLs + metadata for citation tracking
    
    Status Values:
        - "success": Tool executed successfully, data contains result
        - "error": Tool raised exception, error_message contains details
        - "timeout": Execution exceeded timeout_seconds, operation cancelled
    
    Sources Structure (for citation tracking):
        [
            {"url": "https://...", "title": "...", "snippet": "..."},
            ...
        ]
    """
    call_id: str
    tool_name: str
    status: str  # "success" | "error" | "timeout"
    data: Any | None
    error_message: str | None
    execution_time_ms: float
    sources: list[dict] | None = None
    # Correlation ID copied from the originating ToolCall (req 4.10).
    correlation_id: str | None = None


@dataclass
class DecisionResult:
    """
    Represents the LLM's decision about which tools (if any) to invoke.
    
    Attributes:
        tool_calls: Empty list = no tools needed
        reasoning: LLM's reasoning (for logging/debugging)
        proceed_without_tools: True when no tools available but LLM wanted them
    """
    tool_calls: list[ToolCall]
    reasoning: str
    proceed_without_tools: bool = False


@dataclass
class Citation:
    """
    Represents a source citation extracted from tool results.
    
    Attributes:
        url: Source URL
        title: Human-readable title
        tool_name: The tool that provided this citation
    """
    url: str
    title: str
    tool_name: str
