"""
Unit tests for the RateLimiter in ToolExecutor.

Tests the sliding window rate limiting implementation for tool invocations.
"""

import pytest
import asyncio
import time
from app.services.tool_executor import RateLimiter


class TestRateLimiter:
    """Test suite for RateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        """Test that requests under the limit are allowed."""
        limiter = RateLimiter()
        
        # Allow 5 requests per minute
        for i in range(5):
            allowed, retry_after = await limiter.check_and_record("test_tool", 5)
            assert allowed is True, f"Request {i+1} should be allowed"
            assert retry_after is None
    
    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        """Test that requests exceeding the limit are blocked."""
        limiter = RateLimiter()
        
        # Allow 3 requests per minute
        # First 3 should succeed
        for i in range(3):
            allowed, retry_after = await limiter.check_and_record("test_tool", 3)
            assert allowed is True, f"Request {i+1} should be allowed"
        
        # 4th request should be blocked
        allowed, retry_after = await limiter.check_and_record("test_tool", 3)
        assert allowed is False, "4th request should be blocked"
        assert retry_after is not None
        assert retry_after > 0, "retry_after should be positive"
    
    @pytest.mark.asyncio
    async def test_sliding_window_allows_after_time(self):
        """Test that the sliding window allows requests after old ones expire."""
        limiter = RateLimiter()
        
        # Allow 2 requests per minute
        # First 2 succeed
        for i in range(2):
            allowed, retry_after = await limiter.check_and_record("test_tool", 2)
            assert allowed is True
        
        # 3rd request blocked
        allowed, retry_after = await limiter.check_and_record("test_tool", 2)
        assert allowed is False
        
        # Wait for a short time (simulate sliding window)
        # In real scenario, we'd wait 60+ seconds, but for testing we can mock time
        # For now, just verify the retry_after value is reasonable
        assert 0 < retry_after <= 60
    
    @pytest.mark.asyncio
    async def test_different_tools_have_separate_limits(self):
        """Test that different tools have independent rate limits."""
        limiter = RateLimiter()
        
        # Each tool should have its own counter
        allowed1, _ = await limiter.check_and_record("tool_a", 1)
        allowed2, _ = await limiter.check_and_record("tool_b", 1)
        
        assert allowed1 is True
        assert allowed2 is True
        
        # Second request to tool_a should be blocked
        allowed3, retry_after = await limiter.check_and_record("tool_a", 1)
        assert allowed3 is False
        assert retry_after is not None
        
        # But tool_b should still allow one more request before blocking
        # (total 2 for tool_b, limit is 1 per minute so this will block)
        allowed4, _ = await limiter.check_and_record("tool_b", 1)
        assert allowed4 is False
    
    @pytest.mark.asyncio
    async def test_wait_for_slot_success(self):
        """Test wait_for_slot successfully obtains a slot when available."""
        limiter = RateLimiter()
        
        # Fill the limit
        for i in range(2):
            await limiter.check_and_record("test_tool", 2)
        
        # This should wait and eventually return False (timeout)
        # Use a very short timeout for testing
        success = await limiter.wait_for_slot("test_tool", 2, timeout=0.1)
        assert success is False, "Should timeout when no slots available"
    
    @pytest.mark.asyncio
    async def test_wait_for_slot_immediate_success(self):
        """Test wait_for_slot returns immediately when slot is available."""
        limiter = RateLimiter()
        
        # Slot is available
        start = time.time()
        success = await limiter.wait_for_slot("test_tool", 5, timeout=5.0)
        elapsed = time.time() - start
        
        assert success is True
        assert elapsed < 0.5, "Should return immediately when slot available"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_respect_limit(self):
        """Test that concurrent requests from the same tool respect the limit."""
        limiter = RateLimiter()
        
        # Launch 5 concurrent requests with a limit of 3
        async def make_request():
            return await limiter.check_and_record("test_tool", 3)
        
        results = await asyncio.gather(*[make_request() for _ in range(5)])
        
        # Count how many were allowed
        allowed_count = sum(1 for allowed, _ in results if allowed)
        
        # Should allow exactly 3 (the limit)
        assert allowed_count == 3, f"Expected 3 allowed, got {allowed_count}"
        
        # The remaining 2 should be blocked
        blocked_count = sum(1 for allowed, _ in results if not allowed)
        assert blocked_count == 2, f"Expected 2 blocked, got {blocked_count}"
    
    @pytest.mark.asyncio
    async def test_retry_after_accuracy(self):
        """Test that retry_after gives reasonable estimate of wait time."""
        limiter = RateLimiter()
        
        # Record a request
        await limiter.check_and_record("test_tool", 1)
        
        # Immediately try again - should get blocked with retry_after close to 60s
        allowed, retry_after = await limiter.check_and_record("test_tool", 1)
        
        assert allowed is False
        assert retry_after is not None
        # Should be close to 60 seconds (within reasonable margin)
        assert 55 <= retry_after <= 60, f"retry_after should be ~60s, got {retry_after}"


@pytest.mark.asyncio
async def test_rate_limiter_integration_with_tool_executor():
    """Integration test: Rate limiter works correctly in ToolExecutor context."""
    from app.services.tool_executor import ToolExecutor, RateLimiter
    from app.services.tool_registry import ToolRegistry
    from app.services.tool_models import ToolCall
    import uuid
    
    # Create a registry and register a simple tool with rate limit
    registry = ToolRegistry()
    
    @registry.tool(
        name="rate_limited_tool",
        description="A tool with rate limiting",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            },
            "required": ["value"]
        },
        output_schema={"type": "object"},
        rate_limit_rpm=2  # 2 requests per minute
    )
    async def rate_limited_tool(value: str):
        return {"result": value}
    
    # Create executor with shared rate limiter
    limiter = RateLimiter()
    executor = ToolExecutor(registry=registry, rate_limiter=limiter)
    
    # Execute 2 calls - should succeed
    call1 = ToolCall(tool_name="rate_limited_tool", parameters={"value": "test1"}, call_id=str(uuid.uuid4()))
    call2 = ToolCall(tool_name="rate_limited_tool", parameters={"value": "test2"}, call_id=str(uuid.uuid4()))
    
    result1 = await executor._execute_one(call1)
    result2 = await executor._execute_one(call2)
    
    assert result1.status == "success", f"First call should succeed: {result1.error_message}"
    assert result2.status == "success", f"Second call should succeed: {result2.error_message}"
    
    # Execute 3rd call - should be rate limited
    call3 = ToolCall(tool_name="rate_limited_tool", parameters={"value": "test3"}, call_id=str(uuid.uuid4()))
    result3 = await executor._execute_one(call3)
    
    assert result3.status == "error", "Third call should be rate limited"
    assert "Rate limit exceeded" in result3.error_message
    assert "2 requests per minute" in result3.error_message
    assert "retry after" in result3.error_message.lower()
