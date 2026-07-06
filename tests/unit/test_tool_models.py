"""
Unit tests for tool data models.
"""

import pytest
from app.services.tool_models import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    DecisionResult,
    Citation,
)


def test_tool_definition_creation():
    """Test that ToolDefinition can be created with all required fields."""
    def dummy_fn():
        pass

    tool_def = ToolDefinition(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        fn=dummy_fn,
    )

    assert tool_def.name == "test_tool"
    assert tool_def.description == "A test tool"
    assert tool_def.enabled is True
    assert tool_def.timeout_seconds == 30.0
    assert tool_def.rate_limit_rpm is None
    assert tool_def.requires_auth is False
    assert tool_def.examples == []


def test_tool_call_creation():
    """Test that ToolCall can be created with all required fields."""
    tool_call = ToolCall(
        tool_name="web_search",
        parameters={"query": "test query"},
        call_id="test-uuid-123",
    )

    assert tool_call.tool_name == "web_search"
    assert tool_call.parameters == {"query": "test query"}
    assert tool_call.call_id == "test-uuid-123"


def test_tool_result_success():
    """Test that ToolResult can be created for a successful execution."""
    result = ToolResult(
        call_id="test-uuid-123",
        tool_name="web_search",
        status="success",
        data={"results": []},
        error_message=None,
        execution_time_ms=150.5,
        sources=[{"url": "https://example.com", "title": "Example"}],
    )

    assert result.status == "success"
    assert result.data == {"results": []}
    assert result.error_message is None
    assert result.execution_time_ms == 150.5
    assert len(result.sources) == 1


def test_tool_result_error():
    """Test that ToolResult can be created for a failed execution."""
    result = ToolResult(
        call_id="test-uuid-123",
        tool_name="web_search",
        status="error",
        data=None,
        error_message="Connection failed",
        execution_time_ms=50.0,
    )

    assert result.status == "error"
    assert result.data is None
    assert result.error_message == "Connection failed"
    assert result.sources is None


def test_decision_result_with_tools():
    """Test that DecisionResult can be created with tool calls."""
    tool_call = ToolCall(
        tool_name="web_search",
        parameters={"query": "test"},
        call_id="test-uuid-123",
    )

    decision = DecisionResult(
        tool_calls=[tool_call],
        reasoning="User needs current information",
        proceed_without_tools=False,
    )

    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].tool_name == "web_search"
    assert decision.reasoning == "User needs current information"
    assert decision.proceed_without_tools is False


def test_decision_result_no_tools():
    """Test that DecisionResult can be created with no tool calls."""
    decision = DecisionResult(
        tool_calls=[],
        reasoning="No tools needed",
        proceed_without_tools=False,
    )

    assert len(decision.tool_calls) == 0
    assert decision.reasoning == "No tools needed"


def test_citation_creation():
    """Test that Citation can be created with all required fields."""
    citation = Citation(
        url="https://example.com",
        title="Example Article",
        tool_name="web_search",
    )

    assert citation.url == "https://example.com"
    assert citation.title == "Example Article"
    assert citation.tool_name == "web_search"
