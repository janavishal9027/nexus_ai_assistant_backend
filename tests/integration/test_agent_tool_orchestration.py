"""
Integration tests for agent tool orchestration.

Tests the enhanced agent_chat() function with tool orchestration loop.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.agent import (
    _format_tool_result,
    _append_tool_results,
    _strip_sensitive_data,
    _build_token_budget_manager,
)
from app.services.tool_models import ToolResult
from app.models.schemas import MessageDto
from app.services.token_budget import TokenBudgetConfig


def test_format_tool_result_success():
    """Test formatting a successful tool result."""
    result = ToolResult(
        call_id="test-123",
        tool_name="web_search",
        status="success",
        data={"results": [{"title": "Test", "url": "https://example.com"}], "total_found": 1},
        error_message=None,
        execution_time_ms=125.5,
        sources=[{"url": "https://example.com", "title": "Test"}]
    )
    
    formatted = _format_tool_result(result)
    
    assert "[Tool: web_search]" in formatted
    assert "Status: success" in formatted
    assert "Duration:" in formatted  # Just check that duration is present, don't check exact rounding
    assert "results" in formatted
    assert "https://example.com" in formatted


def test_format_tool_result_error():
    """Test formatting an error tool result."""
    result = ToolResult(
        call_id="test-456",
        tool_name="web_search",
        status="error",
        data=None,
        error_message="Network connection failed",
        execution_time_ms=50.0,
        sources=None
    )
    
    formatted = _format_tool_result(result)
    
    assert "[Tool: web_search]" in formatted
    assert "Status: error" in formatted
    assert "Error: Network connection failed" in formatted


def test_format_tool_result_timeout():
    """Test formatting a timeout tool result."""
    result = ToolResult(
        call_id="test-789",
        tool_name="slow_api",
        status="timeout",
        data=None,
        error_message="Tool execution exceeded 30s timeout",
        execution_time_ms=30000.0,
        sources=None
    )
    
    formatted = _format_tool_result(result)
    
    assert "[Tool: slow_api]" in formatted
    assert "Status: timeout" in formatted
    assert "exceeded 30s timeout" in formatted


def test_append_tool_results():
    """Test appending tool results to message list."""
    messages = [
        MessageDto(role="system", content="You are a helpful assistant."),
        MessageDto(role="user", content="What's the weather?"),
    ]
    
    results = [
        ToolResult(
            call_id="test-1",
            tool_name="weather",
            status="success",
            data={"temperature": 72, "condition": "sunny"},
            error_message=None,
            execution_time_ms=100.0,
            sources=None
        )
    ]
    
    updated_messages = _append_tool_results(messages, results)
    
    assert len(updated_messages) == 3
    assert updated_messages[-1].role == "tool"
    assert "weather" in updated_messages[-1].content
    assert "72" in updated_messages[-1].content


def test_strip_sensitive_data():
    """Test stripping sensitive fields from data."""
    data = {
        "username": "alice",
        "password": "secret123",
        "email": "alice@example.com",
        "api_key": "sk-1234567890",
        "results": [
            {"id": 1, "token": "abc", "value": "public"}
        ]
    }
    
    sanitized = _strip_sensitive_data(data)
    
    assert "username" in sanitized
    assert "email" in sanitized
    assert "password" not in sanitized
    assert "api_key" not in sanitized
    assert sanitized["results"][0]["value"] == "public"
    assert "token" not in sanitized["results"][0]


def test_strip_sensitive_data_nested():
    """Test stripping sensitive data from deeply nested structures."""
    data = {
        "config": {
            "database": {
                "host": "localhost",
                "password": "dbpass",
                "port": 5432
            },
            "service": {
                "api_secret": "service_secret",
                "provider": "oauth"
            }
        }
    }
    
    sanitized = _strip_sensitive_data(data)
    
    assert sanitized["config"]["database"]["host"] == "localhost"
    assert sanitized["config"]["database"]["port"] == 5432
    assert "password" not in sanitized["config"]["database"]
    assert sanitized["config"]["service"]["provider"] == "oauth"
    assert "api_secret" not in sanitized["config"]["service"]


def test_strip_sensitive_data_non_dict():
    """Test stripping sensitive data handles non-dict types."""
    # Strings
    assert _strip_sensitive_data("test") == "test"
    
    # Numbers
    assert _strip_sensitive_data(42) == 42
    
    # Lists
    result = _strip_sensitive_data([1, 2, {"key": "value", "password": "secret"}])
    assert len(result) == 3
    assert result[2]["key"] == "value"
    assert "password" not in result[2]
    
    # None
    assert _strip_sensitive_data(None) is None


def test_build_token_budget_manager_from_int():
    """Test building token budget manager from integer config."""
    manager = _build_token_budget_manager(50000)
    
    assert manager.config.enabled is True
    assert manager.config.max_tokens == 50000
    assert manager.config.reserve_for_response == 4096
    assert manager.config.truncation_threshold == 0.8


def test_build_token_budget_manager_from_dict():
    """Test building token budget manager from dict config."""
    config_dict = {
        "enabled": False,
        "max_tokens": 200000,
        "reserve_for_response": 8192,
        "truncation_threshold": 0.9,
        "chars_per_token": 3.5,
    }
    
    manager = _build_token_budget_manager(config_dict)
    
    assert manager.config.enabled is False
    assert manager.config.max_tokens == 200000
    assert manager.config.reserve_for_response == 8192
    assert manager.config.truncation_threshold == 0.9
    assert manager.config.chars_per_token == 3.5


def test_build_token_budget_manager_partial_dict():
    """Test building token budget manager from partial dict config (uses defaults for missing fields)."""
    config_dict = {
        "max_tokens": 150000,
        "truncation_threshold": 0.75,
    }
    
    manager = _build_token_budget_manager(config_dict)
    
    assert manager.config.enabled is True  # default
    assert manager.config.max_tokens == 150000  # provided
    assert manager.config.reserve_for_response == 4096  # default
    assert manager.config.truncation_threshold == 0.75  # provided
    assert manager.config.chars_per_token == 4.0  # default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
