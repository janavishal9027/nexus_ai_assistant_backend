# Rate Limiting for API Tools

## Overview

The agent tool system includes built-in rate limiting to prevent excessive API calls and respect external service limits. Rate limiting is enforced per-tool using a sliding window approach that tracks requests over a 60-second window.

## How It Works

### Sliding Window Algorithm

The rate limiter uses a sliding window approach:

1. **Track Timestamps**: Each tool invocation timestamp is recorded in a queue
2. **Sliding Window**: Before each new request, timestamps older than 60 seconds are removed
3. **Limit Enforcement**: If the number of requests in the window exceeds `rate_limit_rpm`, the request is rejected
4. **Retry Information**: Rejected requests receive information about when the next slot will be available

### Per-Tool Tracking

Each tool has independent rate limiting:
- Tool A with 10 req/min and Tool B with 5 req/min are tracked separately
- Hitting the limit for Tool A doesn't affect Tool B
- Tools without `rate_limit_rpm` set have no limit

## Configuration

### Setting Rate Limits

Rate limits are configured when registering a tool:

```python
from app.services.tool_registry import tool_registry

@tool_registry.tool(
    name="weather_api",
    description="Get current weather data",
    input_schema={
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    },
    output_schema={"type": "object"},
    rate_limit_rpm=60  # 60 requests per minute
)
async def weather_api(location: str):
    # ... implementation
    pass
```

### No Rate Limit

Tools without `rate_limit_rpm` (or with `rate_limit_rpm=None`) have no rate limiting:

```python
@tool_registry.tool(
    name="local_calculator",
    description="Perform calculations",
    input_schema={...},
    output_schema={...},
    # No rate_limit_rpm = unlimited
)
async def calculator(expression: str):
    return {"result": eval(expression)}
```

## Behavior

### When Rate Limit Is Exceeded

When a tool call exceeds the rate limit:

1. **Status**: The `ToolResult` has `status="error"`
2. **Error Message**: Contains:
   - Clear explanation that rate limit was exceeded
   - The configured limit (e.g., "60 requests per minute")
   - When to retry (e.g., "retry after 45.3 seconds")
3. **Execution Time**: Tracks the time spent checking the rate limit (typically < 1ms)

Example error message:
```
Rate limit exceeded: 60 requests per minute. Please retry after 45.3 seconds.
```

### Concurrent Execution

The rate limiter is thread-safe and handles concurrent requests correctly:

```python
# If 10 concurrent calls are made to a tool with limit 5
results = await executor.execute_batch(calls, max_concurrent=10)

# Result:
# - Exactly 5 will succeed (status="success")
# - Exactly 5 will be rate limited (status="error")
```

The `asyncio.Lock` ensures no race conditions when multiple tools are executed concurrently.

### Batch Execution

Rate limits persist across multiple `execute_batch()` calls:

```python
# First batch: 3 calls (limit is 5)
batch1 = [call1, call2, call3]
results1 = await executor.execute_batch(batch1)  # All succeed

# Second batch: 3 calls (limit is 5, 3 already used)
batch2 = [call4, call5, call6]
results2 = await executor.execute_batch(batch2)  
# Result: 2 succeed, 1 is rate limited
```

## Implementation Details

### RateLimiter Class

Located in `app/services/tool_executor.py`:

```python
class RateLimiter:
    """
    Sliding window rate limiter for tool invocations.
    
    Attributes:
        _invocations: Dict mapping tool_name to deque of timestamps
        _lock: asyncio.Lock for thread-safe access
    """
    
    async def check_and_record(
        self, 
        tool_name: str, 
        rate_limit_rpm: int
    ) -> tuple[bool, Optional[float]]:
        """
        Check if invocation is allowed and record it if so.
        
        Returns:
            (allowed, retry_after_seconds)
        """
```

### Integration with ToolExecutor

The `ToolExecutor` checks rate limits before executing tools:

```python
async def _execute_one(self, call: ToolCall) -> ToolResult:
    # ... validation ...
    
    # Enforce rate limiting if configured
    if tool.rate_limit_rpm is not None:
        allowed, retry_after = await self.rate_limiter.check_and_record(
            call.tool_name,
            tool.rate_limit_rpm
        )
        
        if not allowed:
            return ToolResult(
                status="error",
                error_message=f"Rate limit exceeded: {tool.rate_limit_rpm} requests per minute. "
                              f"Please retry after {retry_after:.1f} seconds."
            )
    
    # ... execution ...
```

## Best Practices

### Setting Appropriate Limits

1. **API Provider Limits**: Set `rate_limit_rpm` slightly below the actual API limit
   - If API allows 100 req/min, use 90 to leave a safety margin

2. **Cost Control**: Use rate limits to control costs for paid APIs
   ```python
   rate_limit_rpm=30  # Limit to 30 calls/min = 43,200/day
   ```

3. **Testing**: In development, use lower limits to catch rate limit issues early
   ```python
   rate_limit_rpm=5  # Low limit for testing
   ```

### Handling Rate Limit Errors

In the agent orchestrator, rate limit errors are treated like other tool errors:

```python
# Agent sees the error and can:
# 1. Try a different tool
# 2. Inform the user
# 3. Wait and retry (future enhancement)
```

Example agent response:
```
I tried to fetch weather data but hit a rate limit. 
The weather API allows 60 requests per minute and we've 
exceeded that. Please try again in about 45 seconds, or 
I can try a different data source.
```

### Monitoring

Log entries track rate limit events:

```python
logger.warning(
    f"Rate limiter: Tool '{tool_name}' rate limit exceeded "
    f"({current_count}/{rate_limit_rpm} in last 60s). "
    f"Retry after {retry_after:.2f}s"
)
```

Use these logs to:
- Identify tools that frequently hit limits (may need higher limits)
- Detect unexpected usage patterns
- Track rate limit violations for billing/quota management

## Testing

### Unit Tests

See `tests/unit/test_rate_limiter.py` for comprehensive unit tests:

```bash
pytest tests/unit/test_rate_limiter.py -v
```

Tests cover:
- Requests under limit allowed
- Requests over limit blocked
- Sliding window behavior
- Per-tool independence
- Concurrent execution
- Retry time accuracy

### Integration Tests

See `tests/integration/test_rate_limiting_integration.py`:

```bash
pytest tests/integration/test_rate_limiting_integration.py -v
```

Tests validate:
- Rate limiting across multiple batches
- Per-tool independence in realistic scenarios
- Unlimited tools work correctly
- Concurrent execution with rate limits
- Error messages include retry information

## Future Enhancements

Potential improvements:

1. **Automatic Retry**: Wait for available slot instead of immediate error
   ```python
   async def wait_for_slot(self, tool_name: str, timeout: float) -> bool:
       # Already implemented, not yet used in _execute_one
   ```

2. **Token Bucket Algorithm**: Support burst capacity
   ```python
   rate_limit_rpm=60,
   burst_capacity=10  # Allow 10 instant requests
   ```

3. **Per-User Rate Limiting**: Track limits per user instead of globally
   ```python
   await limiter.check_and_record(tool_name, rate_limit, user_id=user_id)
   ```

4. **Dynamic Limits**: Adjust limits based on API response headers
   ```python
   # Read X-RateLimit-Remaining from API response
   # Dynamically adjust internal limit
   ```

## Troubleshooting

### Problem: Tool always returns rate limit error

**Diagnosis**: Check if limit is too low

```python
# View current tool configuration
tool = tool_registry.get("my_tool")
print(f"Rate limit: {tool.rate_limit_rpm} requests/min")
```

**Solution**: Increase the limit or disable rate limiting

```python
# Option 1: Increase limit
tool.rate_limit_rpm = 120

# Option 2: Disable rate limiting
tool.rate_limit_rpm = None
```

### Problem: Rate limiting too aggressive in testing

**Solution**: Use a separate RateLimiter for tests

```python
# In tests, create a fresh limiter for each test
@pytest.fixture
def fresh_executor(registry):
    limiter = RateLimiter()  # Fresh limiter = no history
    return ToolExecutor(registry=registry, rate_limiter=limiter)
```

### Problem: Need to reset rate limit counters

**Solution**: Create a new RateLimiter instance

```python
# Rate limit state is stored in the limiter instance
# Creating a new instance resets all counters
executor.rate_limiter = RateLimiter()
```

## API Reference

### RateLimiter

#### `check_and_record(tool_name: str, rate_limit_rpm: int) -> tuple[bool, Optional[float]]`

Check if a tool invocation is allowed and record it if so.

**Args:**
- `tool_name`: Name of the tool being invoked
- `rate_limit_rpm`: Maximum requests per minute

**Returns:**
- `(True, None)` if allowed and recorded
- `(False, retry_after_seconds)` if rate limited

#### `wait_for_slot(tool_name: str, rate_limit_rpm: int, timeout: float) -> bool`

Wait for a rate limit slot to become available.

**Args:**
- `tool_name`: Name of the tool
- `rate_limit_rpm`: Maximum requests per minute  
- `timeout`: Maximum seconds to wait

**Returns:**
- `True` if slot obtained within timeout
- `False` if timeout expired

### ToolDefinition

#### `rate_limit_rpm: int | None`

Rate limit in requests per minute for this tool.

- `None` or not set: No rate limiting
- Positive integer: Maximum requests per 60-second sliding window

**Example:**
```python
ToolDefinition(
    name="api_tool",
    rate_limit_rpm=60,  # 60 requests per minute
    # ... other fields
)
```
