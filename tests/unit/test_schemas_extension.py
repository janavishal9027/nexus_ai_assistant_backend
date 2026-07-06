"""
Unit tests for schema extensions (MessageDto tool role support and ChatResponse tool fields).
"""

import pytest
from app.models.schemas import MessageDto, ChatResponse


def test_message_dto_supports_tool_role():
    """Test that MessageDto can be created with role='tool'."""
    message = MessageDto(
        role="tool",
        content="[Tool: web_search] Status: success | Duration: 150ms\nResults: ..."
    )

    assert message.role == "tool"
    assert message.content.startswith("[Tool: web_search]")


def test_message_dto_supports_standard_roles():
    """Test that MessageDto still supports standard roles (user, assistant, system)."""
    user_msg = MessageDto(role="user", content="Hello")
    assistant_msg = MessageDto(role="assistant", content="Hi there")
    system_msg = MessageDto(role="system", content="You are a helpful assistant")

    assert user_msg.role == "user"
    assert assistant_msg.role == "assistant"
    assert system_msg.role == "system"


def test_chat_response_has_tool_fields():
    """Test that ChatResponse has the new tool-related fields with default values."""
    response = ChatResponse(
        conversation_id=1,
        content="Here is your answer",
        model="gpt-4",
        platform="openrouter",
    )

    assert response.tool_calls_made == 0
    assert response.tool_rounds == 0


def test_chat_response_tool_fields_can_be_set():
    """Test that ChatResponse tool fields can be set to non-zero values."""
    response = ChatResponse(
        conversation_id=1,
        content="Here is your answer with sources",
        model="gpt-4",
        platform="openrouter",
        tool_calls_made=2,
        tool_rounds=1,
    )

    assert response.tool_calls_made == 2
    assert response.tool_rounds == 1


def test_chat_response_backward_compatible():
    """Test that ChatResponse is backward compatible (old code doesn't need to pass new fields)."""
    # Old code that doesn't know about new fields should still work
    response = ChatResponse(
        conversation_id=1,
        content="Answer",
        model="gpt-4",
        platform="openrouter",
        fallback_attempts=0,
    )

    assert response.tool_calls_made == 0  # Should default to 0
    assert response.tool_rounds == 0  # Should default to 0
