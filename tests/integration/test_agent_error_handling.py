"""
Integration tests for agent error handling and monitoring.

Tests the enhanced error handling in agent_chat() including:
- Module-level tool failure counters
- Error context injection into messages
- Guaranteed response generation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.agent import (
    _append_tool_results,
    get_tool_failure_rates,
    _tool_success_counts,
    _tool_failure_counts,
)
from app.services.tool_models import ToolResult
from app.models.schemas import MessageDto


def test_append_tool_results_tracks_success():
    """Test that successful tool results increment success counter."""
    # Clear counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    
    messages = []
    results = [
        ToolResult(
            call_id="test-1",
            tool_name="web_search",
            status="success",
            data={"results": []},
            error_message=None,
            execution_time_ms=100.0,
            sources=None
        )
    ]
    
    _append_tool_results(messages, results)
    
    assert _tool_success_counts.get("web_search") == 1
    assert _tool_failure_counts.get("web_search") is None


def test_append_tool_results_tracks_failure():
    """Test that failed tool results increment failure counter."""
    # Clear counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    
    messages = []
    results = [
        ToolResult(
            call_id="test-2",
            tool_name="web_search",
            status="error",
            data=None,
            error_message="Network error",
            execution_time_ms=50.0,
            sources=None
        )
    ]
    
    _append_tool_results(messages, results)
    
    assert _tool_success_counts.get("web_search") is None
    assert _tool_failure_counts.get("web_search") == 1


def test_append_tool_results_tracks_timeout():
    """Test that timeout results increment failure counter."""
    # Clear counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    
    messages = []
    results = [
        ToolResult(
            call_id="test-3",
            tool_name="slow_api",
            status="timeout",
            data=None,
            error_message="Exceeded timeout",
            execution_time_ms=30000.0,
            sources=None
        )
    ]
    
    _append_tool_results(messages, results)
    
    assert _tool_success_counts.get("slow_api") is None
    assert _tool_failure_counts.get("slow_api") == 1


def test_get_tool_failure_rates_single_tool():
    """Test calculating failure rates for a single tool."""
    # Clear and set up counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    _tool_success_counts["web_search"] = 8
    _tool_failure_counts["web_search"] = 2
    
    rates = get_tool_failure_rates()
    
    assert "web_search" in rates
    assert rates["web_search"]["success_count"] == 8
    assert rates["web_search"]["failure_count"] == 2
    assert rates["web_search"]["total_calls"] == 10
    assert rates["web_search"]["failure_rate_percent"] == 20.0


def test_get_tool_failure_rates_multiple_tools():
    """Test calculating failure rates for multiple tools."""
    # Clear and set up counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    _tool_success_counts["web_search"] = 10
    _tool_failure_counts["web_search"] = 0
    _tool_success_counts["weather_api"] = 7
    _tool_failure_counts["weather_api"] = 3
    
    rates = get_tool_failure_rates()
    
    assert len(rates) == 2
    assert rates["web_search"]["failure_rate_percent"] == 0.0
    assert rates["weather_api"]["failure_rate_percent"] == 30.0


def test_get_tool_failure_rates_only_failures():
    """Test calculating rates for tools with only failures."""
    # Clear and set up counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    _tool_failure_counts["broken_api"] = 5
    
    rates = get_tool_failure_rates()
    
    assert "broken_api" in rates
    assert rates["broken_api"]["success_count"] == 0
    assert rates["broken_api"]["failure_count"] == 5
    assert rates["broken_api"]["total_calls"] == 5
    assert rates["broken_api"]["failure_rate_percent"] == 100.0


def test_get_tool_failure_rates_empty():
    """Test calculating rates when no tools have been called."""
    # Clear counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    
    rates = get_tool_failure_rates()
    
    assert rates == {}


def test_append_tool_results_adds_error_context_to_messages():
    """Test that error tool results add properly formatted context to messages."""
    # Clear counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    
    messages = [
        MessageDto(role="system", content="You are a helpful assistant."),
        MessageDto(role="user", content="What's the weather?"),
    ]
    
    results = [
        ToolResult(
            call_id="test-4",
            tool_name="weather_api",
            status="error",
            data=None,
            error_message="API rate limit exceeded",
            execution_time_ms=100.0,
            sources=None
        )
    ]
    
    updated_messages = _append_tool_results(messages, results)
    
    # Should have added a tool message with error context
    assert len(updated_messages) == 3
    assert updated_messages[-1].role == "tool"
    assert "weather_api" in updated_messages[-1].content
    assert "error" in updated_messages[-1].content.lower()
    assert "API rate limit exceeded" in updated_messages[-1].content


def test_append_tool_results_mixed_success_and_failure():
    """Test tracking counters with mixed success and failure results."""
    # Clear counters
    _tool_success_counts.clear()
    _tool_failure_counts.clear()
    
    messages = []
    results = [
        ToolResult(
            call_id="test-5",
            tool_name="web_search",
            status="success",
            data={"results": []},
            error_message=None,
            execution_time_ms=100.0,
            sources=None
        ),
        ToolResult(
            call_id="test-6",
            tool_name="web_search",
            status="error",
            data=None,
            error_message="Network timeout",
            execution_time_ms=5000.0,
            sources=None
        ),
        ToolResult(
            call_id="test-7",
            tool_name="weather_api",
            status="success",
            data={"temperature": 72},
            error_message=None,
            execution_time_ms=150.0,
            sources=None
        ),
    ]
    
    _append_tool_results(messages, results)
    
    assert _tool_success_counts["web_search"] == 1
    assert _tool_failure_counts["web_search"] == 1
    assert _tool_success_counts["weather_api"] == 1
    assert _tool_failure_counts.get("weather_api") is None
    
    rates = get_tool_failure_rates()
    assert rates["web_search"]["failure_rate_percent"] == 50.0
    assert rates["weather_api"]["failure_rate_percent"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
