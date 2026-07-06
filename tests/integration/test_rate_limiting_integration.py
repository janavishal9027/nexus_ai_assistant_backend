"""
Integration tests for rate limiting in the agent tool system.

Tests demonstrate end-to-end rate limiting behavior with real tool execution
scenarios.
"""

import pytest
import asyncio
import uuid
from app.services.tool_executor import ToolExecutor, RateLimiter
from app.services.tool_registry import ToolRegistry
from app.services.tool_models import ToolCall


@pytest.mark.asyncio
async def test_rate_limiting_across_multiple_batches():
    """
    Test that rate limiting persists across multiple execute_batch calls.
    
    This simulates a real scenario where an agent makes multiple rounds of
    tool calls and the rate limiter correctly tracks state between rounds.
    """
    registry = ToolRegistry()
    
    # Register a tool with a strict rate limit
    @registry.tool(
        name="api_tool",
        description="A tool that calls an external API",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        },
        output_schema={"type": "object"},
        rate_limit_rpm=3  # Only 3 calls per minute
    )
    async def api_tool(query: str):
        # Simulate API call
        await asyncio.sleep(0.01)
        return {"result": f"Response for {query}"}
    
    # Create executor with shared rate limiter
    limiter = RateLimiter()
    executor = ToolExecutor(registry=registry, rate_limiter=limiter)
    
    # First batch: 2 calls (should succeed)
    batch1 = [
        ToolCall(tool_name="api_tool", parameters={"query": f"query{i}"}, call_id=str(uuid.uuid4()))
        for i in range(2)
    ]
    results1 = await executor.execute_batch(batch1)
    
    assert all(r.status == "success" for r in results1), "First batch should succeed"
    assert len(results1) == 2
    
    # Second batch: 2 more calls (1 should succeed, 1 should be rate limited)
    batch2 = [
        ToolCall(tool_name="api_tool", parameters={"query": f"query{i}"}, call_id=str(uuid.uuid4()))
        for i in range(2, 4)
    ]
    results2 = await executor.execute_batch(batch2)
    
    assert len(results2) == 2
    
    # One should succeed (the 3rd call), one should be rate limited (the 4th)
    success_count = sum(1 for r in results2 if r.status == "success")
    error_count = sum(1 for r in results2 if r.status == "error")
    
    assert success_count == 1, "Only 1 call in second batch should succeed"
    assert error_count == 1, "1 call should be rate limited"
    
    # Verify the error message
    rate_limited_result = next(r for r in results2 if r.status == "error")
    assert "Rate limit exceeded" in rate_limited_result.error_message
    assert "3 requests per minute" in rate_limited_result.error_message


@pytest.mark.asyncio
async def test_rate_limiting_per_tool_independence():
    """
    Test that different tools have independent rate limits.
    
    Validates that hitting the rate limit for one tool doesn't affect another.
    """
    registry = ToolRegistry()
    
    # Register two tools with different rate limits
    @registry.tool(
        name="slow_api",
        description="API with strict rate limit",
        input_schema={
            "type": "object",
            "properties": {"data": {"type": "string"}},
            "required": ["data"]
        },
        output_schema={"type": "object"},
        rate_limit_rpm=1  # Very strict
    )
    async def slow_api(data: str):
        return {"result": data}
    
    @registry.tool(
        name="fast_api",
        description="API with generous rate limit",
        input_schema={
            "type": "object",
            "properties": {"data": {"type": "string"}},
            "required": ["data"]
        },
        output_schema={"type": "object"},
        rate_limit_rpm=10  # Generous
    )
    async def fast_api(data: str):
        return {"result": data}
    
    limiter = RateLimiter()
    executor = ToolExecutor(registry=registry, rate_limiter=limiter)
    
    # Hit the limit for slow_api (2 calls, limit is 1)
    slow_calls = [
        ToolCall(tool_name="slow_api", parameters={"data": f"test{i}"}, call_id=str(uuid.uuid4()))
        for i in range(2)
    ]
    slow_results = await executor.execute_batch(slow_calls)
    
    # First should succeed, second should be rate limited
    assert slow_results[0].status == "success"
    assert slow_results[1].status == "error"
    assert "Rate limit exceeded" in slow_results[1].error_message
    
    # fast_api should still work fine
    fast_calls = [
        ToolCall(tool_name="fast_api", parameters={"data": f"test{i}"}, call_id=str(uuid.uuid4()))
        for i in range(5)
    ]
    fast_results = await executor.execute_batch(fast_calls)
    
    # All should succeed because fast_api has a limit of 10
    assert all(r.status == "success" for r in fast_results), \
        "fast_api should not be affected by slow_api's rate limit"


@pytest.mark.asyncio
async def test_rate_limiting_with_no_limit_configured():
    """
    Test that tools without rate_limit_rpm set work normally.
    
    Ensures rate limiting doesn't interfere with unlimited tools.
    """
    registry = ToolRegistry()
    
    # Register a tool WITHOUT rate limiting (rate_limit_rpm=None)
    @registry.tool(
        name="unlimited_tool",
        description="A tool with no rate limit",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"]
        },
        output_schema={"type": "object"},
        # Note: rate_limit_rpm is None by default
    )
    async def unlimited_tool(value: int):
        return {"result": value}
    
    limiter = RateLimiter()
    executor = ToolExecutor(registry=registry, rate_limiter=limiter)
    
    # Make many calls - all should succeed
    calls = [
        ToolCall(tool_name="unlimited_tool", parameters={"value": i}, call_id=str(uuid.uuid4()))
        for i in range(20)
    ]
    results = await executor.execute_batch(calls)
    
    # All 20 should succeed
    assert all(r.status == "success" for r in results), \
        "Tools without rate limits should allow unlimited calls"
    assert len(results) == 20


@pytest.mark.asyncio
async def test_rate_limiting_concurrent_execution():
    """
    Test that rate limiting works correctly with concurrent tool execution.
    
    Validates that the rate limiter's lock prevents race conditions.
    """
    registry = ToolRegistry()
    
    @registry.tool(
        name="concurrent_api",
        description="API called concurrently",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"]
        },
        output_schema={"type": "object"},
        rate_limit_rpm=5
    )
    async def concurrent_api(id: int):
        # Simulate some async work
        await asyncio.sleep(0.01)
        return {"id": id}
    
    limiter = RateLimiter()
    executor = ToolExecutor(registry=registry, rate_limiter=limiter)
    
    # Launch 10 concurrent calls (limit is 5)
    calls = [
        ToolCall(tool_name="concurrent_api", parameters={"id": i}, call_id=str(uuid.uuid4()))
        for i in range(10)
    ]
    results = await executor.execute_batch(calls, max_concurrent=10)
    
    # Count successes and failures
    success_count = sum(1 for r in results if r.status == "success")
    error_count = sum(1 for r in results if r.status == "error")
    
    assert success_count == 5, "Should allow exactly 5 concurrent calls (the limit)"
    assert error_count == 5, "Should reject 5 calls due to rate limit"
    
    # Verify error messages
    for result in results:
        if result.status == "error":
            assert "Rate limit exceeded" in result.error_message


@pytest.mark.asyncio
async def test_rate_limiting_error_includes_retry_info():
    """
    Test that rate limit errors include helpful retry information.
    
    Validates the user-facing error message provides actionable information.
    """
    registry = ToolRegistry()
    
    @registry.tool(
        name="rate_test",
        description="Test tool",
        input_schema={
            "type": "object",
            "properties": {},
        },
        output_schema={"type": "object"},
        rate_limit_rpm=1
    )
    async def rate_test():
        return {"ok": True}
    
    limiter = RateLimiter()
    executor = ToolExecutor(registry=registry, rate_limiter=limiter)
    
    # First call succeeds
    call1 = ToolCall(tool_name="rate_test", parameters={}, call_id=str(uuid.uuid4()))
    result1 = await executor._execute_one(call1)
    assert result1.status == "success"
    
    # Second call gets rate limited
    call2 = ToolCall(tool_name="rate_test", parameters={}, call_id=str(uuid.uuid4()))
    result2 = await executor._execute_one(call2)
    
    assert result2.status == "error"
    
    # Verify error message structure
    error_msg = result2.error_message
    assert "Rate limit exceeded" in error_msg
    assert "1 requests per minute" in error_msg or "1 request per minute" in error_msg
    assert "retry after" in error_msg.lower()
    
    # The message should contain a numeric retry time
    import re
    retry_time_match = re.search(r'(\d+\.?\d*)\s*seconds', error_msg)
    assert retry_time_match is not None, "Error should include retry time in seconds"
    retry_seconds = float(retry_time_match.group(1))
    assert 0 < retry_seconds <= 60, "Retry time should be reasonable (0-60 seconds)"
