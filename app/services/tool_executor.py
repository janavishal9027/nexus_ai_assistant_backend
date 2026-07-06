"""
Tool Executor for the agent tool system.

This module provides the execution engine for tool calls, including parameter validation,
timeout enforcement, concurrent execution, and result capture.
"""

import asyncio
import logging
import time
import inspect
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional
import jsonschema
from jsonschema.exceptions import ValidationError
import httpx

from .tool_models import ToolCall, ToolResult, ToolDefinition
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class RetryPolicy:
    """
    Configurable retry policy for HTTP API tool requests.

    Attributes:
        max_attempts: Total number of attempts (1 = no retry). Default is 3.
        backoff_factor: Multiplier applied after each failed attempt. Default is 0.5.
        retry_on_status: HTTP status codes that should trigger a retry. Default
            includes 429 (rate limit) and 5xx server errors.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 0.5,
        retry_on_status: Optional[set[int]] = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        # By default retry on rate-limit and server-side errors; 4xx client
        # errors (except 429) are not retried because resending the same
        # request won't fix a bad request / auth failure.
        self.retry_on_status: set[int] = retry_on_status if retry_on_status is not None else {
            429, 500, 502, 503, 504
        }

    def should_retry(self, status_code: int) -> bool:
        return status_code in self.retry_on_status

    def wait_seconds(self, attempt: int) -> float:
        """Return how long to wait before the given attempt (0-indexed)."""
        if attempt == 0:
            return 0.0
        return self.backoff_factor * (2 ** (attempt - 1))


# Module-level default retry policy – can be overridden per executor instance.
_DEFAULT_RETRY_POLICY = RetryPolicy()


class RateLimiter:
    """
    Sliding window rate limiter for tool invocations.
    
    Tracks invocation timestamps per tool name and enforces rate_limit_rpm
    (requests per minute) using a sliding window approach.
    
    Each tool maintains a deque of timestamps representing recent invocations.
    Before allowing a new invocation, we remove timestamps older than 60 seconds
    and check if we're under the limit.
    """
    
    def __init__(self):
        """Initialize the rate limiter with empty tracking state."""
        # tool_name -> deque of timestamps
        self._invocations: dict[str, deque[float]] = {}
        # Lock for thread-safe access (though in asyncio we may not strictly need this)
        self._lock = asyncio.Lock()
    
    async def check_and_record(self, tool_name: str, rate_limit_rpm: int) -> tuple[bool, Optional[float]]:
        """
        Check if a tool invocation would exceed the rate limit and record it if allowed.
        
        Uses a sliding window: remove timestamps older than 60 seconds, then check
        if the count is below the limit.
        
        Args:
            tool_name: Name of the tool being invoked
            rate_limit_rpm: Maximum requests per minute for this tool
        
        Returns:
            Tuple of (allowed: bool, retry_after_seconds: Optional[float])
            - If allowed is True, the invocation was recorded and can proceed
            - If allowed is False, retry_after_seconds indicates when the next slot opens
        """
        async with self._lock:
            current_time = time.time()
            window_start = current_time - 60.0  # 60 seconds sliding window
            
            # Get or create the invocation queue for this tool
            if tool_name not in self._invocations:
                self._invocations[tool_name] = deque()
            
            invocations = self._invocations[tool_name]
            
            # Remove timestamps outside the sliding window
            while invocations and invocations[0] < window_start:
                invocations.popleft()
            
            # Check if we're under the limit
            if len(invocations) < rate_limit_rpm:
                # Record this invocation and allow it
                invocations.append(current_time)
                logger.debug(
                    f"Rate limiter: Tool '{tool_name}' allowed "
                    f"({len(invocations)}/{rate_limit_rpm} in last 60s)"
                )
                return (True, None)
            else:
                # Calculate when the oldest invocation in the window will expire
                oldest_timestamp = invocations[0]
                retry_after = oldest_timestamp + 60.0 - current_time
                logger.warning(
                    f"Rate limiter: Tool '{tool_name}' rate limit exceeded "
                    f"({len(invocations)}/{rate_limit_rpm} in last 60s). "
                    f"Retry after {retry_after:.2f}s"
                )
                return (False, max(0.0, retry_after))
    
    async def wait_for_slot(self, tool_name: str, rate_limit_rpm: int, timeout: float) -> bool:
        """
        Wait until a rate limit slot is available or timeout expires.
        
        This method will check the rate limit repeatedly (with small delays) until
        either a slot becomes available or the timeout is reached.
        
        Args:
            tool_name: Name of the tool
            rate_limit_rpm: Maximum requests per minute
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if a slot was obtained, False if timeout expired
        """
        start_time = time.time()
        
        while True:
            allowed, retry_after = await self.check_and_record(tool_name, rate_limit_rpm)
            
            if allowed:
                return True
            
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            
            if remaining_timeout <= 0:
                logger.warning(
                    f"Rate limiter: Tool '{tool_name}' timeout while waiting for slot "
                    f"(waited {elapsed:.2f}s)"
                )
                return False
            
            # Wait for the smaller of: retry_after or remaining_timeout
            # Add a small buffer to avoid tight loops
            wait_time = min(retry_after or 1.0, remaining_timeout, 1.0)
            await asyncio.sleep(wait_time)


class ToolExecutor:
    """
    Component responsible for executing tool calls and managing their lifecycle.
    
    Handles:
    - Parameter validation against tool input schemas
    - Timeout enforcement for tool execution
    - Concurrent execution of multiple tool calls
    - Capturing and structuring results
    - Error handling and graceful degradation
    - HTTP API tool execution with retry and API key injection
    - Rate limiting enforcement per tool
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        retry_policy: Optional[RetryPolicy] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize the tool executor.
        
        Args:
            registry: ToolRegistry instance containing registered tools
            retry_policy: Retry policy used for HTTP API tools.  Falls back to
                the module-level default (3 attempts, exponential back-off) when
                not provided.
            rate_limiter: RateLimiter instance for enforcing per-tool rate limits.
                Creates a new instance if not provided.
        """
        self.registry = registry
        self.retry_policy = retry_policy or _DEFAULT_RETRY_POLICY
        self.rate_limiter = rate_limiter or RateLimiter()
    
    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        max_concurrent: int = 5,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls concurrently, respecting the max_concurrent limit.

        Uses a semaphore to cap the number of simultaneously running tools so resource
        usage stays bounded regardless of batch size.  Results are returned in the same
        order as the input tool_calls list.

        Partial-failure isolation: asyncio.gather(return_exceptions=True) is used so
        that an exception escaping _execute_one (which should never happen in normal
        flow, but is defended against here) does not cancel sibling tasks.

        Cancellation propagation: if the caller's coroutine is cancelled (e.g. because
        the HTTP request was disconnected), the CancelledError propagates through
        asyncio.gather and all running/queued tasks are cancelled together.  Tasks that
        are still waiting on the semaphore (i.e. queued) are also cancelled because
        they are Tasks created with asyncio.ensure_future / asyncio.create_task and
        will receive the cancel() call.

        Args:
            tool_calls: List of tool calls to execute.
            max_concurrent: Maximum number of tools running at the same time (default 5).

        Returns:
            List of ToolResults in the same order as tool_calls.
        """
        if not tool_calls:
            return []

        # Semaphore enforces the global concurrent-tool limit.
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _guarded_execute(call: ToolCall) -> ToolResult:
            """Acquire semaphore then run the tool, ensuring cancellation propagates."""
            async with semaphore:
                return await self._execute_one(call)

        # Create explicit Tasks so that cancellation of the gather propagates to each
        # individual task (including those still waiting on the semaphore).
        tasks = [asyncio.ensure_future(_guarded_execute(call)) for call in tool_calls]

        try:
            # return_exceptions=True means a raised exception from any task is returned
            # as a value rather than cancelling the rest of the gather.
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            # The caller was cancelled. Cancel all still-running/queued tasks and
            # wait for them to finish cleanup before re-raising.
            for task in tasks:
                task.cancel()
            # Suppress further CancelledErrors from the tasks themselves while we
            # wait for graceful shutdown.
            await asyncio.gather(*tasks, return_exceptions=True)
            raise  # Re-raise to propagate the cancellation to the caller.

        # Convert any bare exceptions (unexpected; _execute_one should swallow them)
        # to error ToolResults so the return type is always list[ToolResult].
        processed_results: list[ToolResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(
                    f"Unexpected exception escaping _execute_one for tool "
                    f"'{tool_calls[i].tool_name}': {result}",
                    exc_info=result,
                )
                processed_results.append(ToolResult(
                    call_id=tool_calls[i].call_id,
                    tool_name=tool_calls[i].tool_name,
                    status="error",
                    data=None,
                    error_message=f"Unexpected error: {str(result)}",
                    execution_time_ms=0.0,
                ))
            else:
                processed_results.append(result)

        return processed_results
    
    async def _execute_one(self, call: ToolCall) -> ToolResult:
        """
        Execute a single tool call with timeout enforcement.
        
        Validates parameters against the tool's input schema, executes the tool
        (wrapping sync tools in asyncio.to_thread), and captures the result.
        
        Args:
            call: ToolCall to execute
        
        Returns:
            ToolResult with status "success", "error", or "timeout"
        """
        start_time = time.perf_counter()
        
        # Validate tool exists
        tool = self.registry.get(call.tool_name)
        if tool is None:
            logger.error(f"Tool '{call.tool_name}' not found in registry")
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                data=None,
                error_message=f"Tool '{call.tool_name}' not found",
                execution_time_ms=0.0
            )
        
        # Enforce rate limiting if configured for this tool
        if tool.rate_limit_rpm is not None:
            allowed, retry_after = await self.rate_limiter.check_and_record(
                call.tool_name,
                tool.rate_limit_rpm
            )
            
            if not allowed:
                logger.warning(
                    f"Rate limit exceeded for tool '{call.tool_name}' "
                    f"(limit: {tool.rate_limit_rpm} requests/min). "
                    f"Retry after {retry_after:.2f}s"
                )
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    status="error",
                    data=None,
                    error_message=(
                        f"Rate limit exceeded: {tool.rate_limit_rpm} requests per minute. "
                        f"Please retry after {retry_after:.1f} seconds."
                    ),
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
        
        # Validate parameters against input schema
        try:
            jsonschema.validate(instance=call.parameters, schema=tool.input_schema)
        except ValidationError as e:
            logger.error(f"Parameter validation failed for tool '{call.tool_name}': {e.message}")
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                data=None,
                error_message=f"Invalid parameters: {e.message}",
                execution_time_ms=0.0
            )
        
        # Log the invocation with timestamp (requirement 12.1)
        logger.info(
            f"[{datetime.utcnow().isoformat()}Z] Executing tool '{call.tool_name}' "
            f"with call_id={call.call_id}, parameters={call.parameters}"
        )
        
        # Track start time for 50% timeout warning (requirement 12.7)
        timeout_threshold = tool.timeout_seconds * 0.5
        
        # Execute the tool with timeout enforcement
        try:
            # Route HTTP API tools to the dedicated HTTP executor.
            # HTTP tools return a ToolResult directly, so handle separately.
            if self._is_http_tool(tool):
                async def _http_coro():
                    return await self._execute_http_tool(call, tool)

                async def _http_with_half_timeout_warning():
                    task = asyncio.ensure_future(_http_coro())
                    try:
                        return await asyncio.wait_for(
                            asyncio.shield(task),
                            timeout=timeout_threshold,
                        )
                    except asyncio.TimeoutError:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        logger.warning(
                            f"Tool '{call.tool_name}' (call_id={call.call_id}) has used "
                            f"{elapsed_ms:.0f}ms which exceeds 50% of the "
                            f"{tool.timeout_seconds}s timeout threshold"
                        )
                        return await task

                return await asyncio.wait_for(
                    _http_with_half_timeout_warning(),
                    timeout=tool.timeout_seconds,
                )

            # Check if the tool function is async or sync
            if inspect.iscoroutinefunction(tool.fn):
                coro = tool.fn(**call.parameters)
            else:
                # Sync tool - wrap in thread pool
                coro = asyncio.to_thread(tool.fn, **call.parameters)

            # Wrap with timeout; use a background monitor task for the 50% warning
            # so we don't need a separate thread or complex instrumentation.
            async def _run_with_half_timeout_warning():
                """Run the coroutine; emit a WARNING if it exceeds 50% of the timeout."""
                task = asyncio.ensure_future(coro)
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=timeout_threshold,
                    )
                except asyncio.TimeoutError:
                    # 50% elapsed – emit the warning then keep waiting for the full timeout
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.warning(
                        f"Tool '{call.tool_name}' (call_id={call.call_id}) has used "
                        f"{elapsed_ms:.0f}ms which exceeds 50% of the "
                        f"{tool.timeout_seconds}s timeout threshold"
                    )
                    # Wait for the remaining time; if the task is already done it
                    # returns immediately.
                    return await task

            result = await asyncio.wait_for(
                _run_with_half_timeout_warning(),
                timeout=tool.timeout_seconds,
            )
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                f"Tool '{call.tool_name}' (call_id={call.call_id}) completed successfully "
                f"in {execution_time_ms:.2f}ms"
            )
            
            # Extract _sources from result dict if present (for citation tracking)
            sources = None
            if isinstance(result, dict) and "_sources" in result:
                sources = result.pop("_sources")
            
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="success",
                data=result,
                error_message=None,
                execution_time_ms=execution_time_ms,
                sources=sources
            )
        
        except asyncio.TimeoutError:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"Tool '{call.tool_name}' (call_id={call.call_id}) timed out "
                f"after {tool.timeout_seconds}s"
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="timeout",
                data=None,
                error_message=f"Tool execution exceeded {tool.timeout_seconds}s timeout",
                execution_time_ms=execution_time_ms
            )
        
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Tool '{call.tool_name}' (call_id={call.call_id}) raised exception: {e}",
                exc_info=True
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                data=None,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )

    # ------------------------------------------------------------------
    # HTTP API tool support
    # ------------------------------------------------------------------

    @staticmethod
    def _is_http_tool(tool: ToolDefinition) -> bool:
        """
        Return True when the ToolDefinition represents an HTTP API tool.

        An HTTP tool is identified by a ``metadata`` dict (stored either
        directly on the definition or inside ``input_schema["x-metadata"]``)
        that contains an ``endpoint`` key.

        Supported metadata locations (checked in order):
        1. ``tool.metadata`` – a plain ``dict`` attribute on the definition,
           populated by callers that extend ``ToolDefinition`` or store extra
           data there.
        2. ``tool.input_schema.get("x-metadata")`` – a JSON-Schema extension
           field embedded directly in the input schema.
        """
        metadata = getattr(tool, "metadata", None) or tool.input_schema.get("x-metadata", {})
        return isinstance(metadata, dict) and "endpoint" in metadata

    @staticmethod
    def _get_http_metadata(tool: ToolDefinition) -> dict:
        """Return the HTTP metadata dict for an HTTP API tool."""
        return getattr(tool, "metadata", None) or tool.input_schema.get("x-metadata", {})

    @staticmethod
    def _inject_api_keys(headers: dict, metadata: dict) -> dict:
        """
        Inject API key(s) from environment variables into *headers*.

        The metadata dict may contain an ``auth_env_var`` key whose value is
        the name of an environment variable holding the API key.  If found,
        the key is injected according to ``auth_header`` (default
        ``"Authorization"``), using ``auth_scheme`` as a prefix (default
        ``"Bearer"``).

        Example metadata::

            {
                "endpoint": "https://api.example.com/data",
                "method": "GET",
                "auth_env_var": "MY_API_KEY",
                "auth_header": "Authorization",
                "auth_scheme": "Bearer"
            }

        Returns a *copy* of headers with the key injected (original unchanged).
        """
        headers = dict(headers)  # make a copy so we don't mutate the caller's dict

        auth_env_var: Optional[str] = metadata.get("auth_env_var")
        if auth_env_var:
            api_key = os.environ.get(auth_env_var)
            if api_key:
                auth_header: str = metadata.get("auth_header", "Authorization")
                auth_scheme: str = metadata.get("auth_scheme", "Bearer")
                headers[auth_header] = f"{auth_scheme} {api_key}" if auth_scheme else api_key
                logger.debug(f"Injected API key from env var '{auth_env_var}' into header '{auth_header}'")
            else:
                logger.warning(
                    f"Expected API key in environment variable '{auth_env_var}' but it is not set or empty"
                )

        return headers

    async def _execute_http_tool(self, call: ToolCall, tool: ToolDefinition) -> ToolResult:
        """
        Execute an HTTP API tool call using httpx.AsyncClient.

        The method:
        1. Reads ``endpoint``, ``method``, and ``headers`` from tool metadata.
        2. Injects API keys from environment variables.
        3. Sends the HTTP request with the tool call parameters (as query params
           for GET/DELETE, or as a JSON body for POST/PUT/PATCH).
        4. Retries transient failures according to the executor's retry policy.
        5. Returns an error ``ToolResult`` for any non-2xx final response.

        Args:
            call: The ToolCall to execute.
            tool: The resolved ToolDefinition (already retrieved from registry).

        Returns:
            A ToolResult with status "success", "error", or "timeout".
        """
        start_time = time.perf_counter()
        metadata = self._get_http_metadata(tool)

        endpoint: str = metadata["endpoint"]
        method: str = metadata.get("method", "GET").upper()
        base_headers: dict = dict(metadata.get("headers", {}))

        # Inject API keys
        headers = self._inject_api_keys(base_headers, metadata)

        # Determine how to pass parameters
        use_json_body = method in {"POST", "PUT", "PATCH"}

        attempt = 0
        last_error: Optional[str] = None

        async with httpx.AsyncClient(timeout=tool.timeout_seconds) as client:
            while attempt < self.retry_policy.max_attempts:
                wait = self.retry_policy.wait_seconds(attempt)
                if wait > 0:
                    logger.debug(
                        f"HTTP tool '{call.tool_name}' (call_id={call.call_id}): "
                        f"waiting {wait:.2f}s before attempt {attempt + 1}"
                    )
                    await asyncio.sleep(wait)

                attempt += 1
                logger.debug(
                    f"HTTP tool '{call.tool_name}' (call_id={call.call_id}): "
                    f"attempt {attempt}/{self.retry_policy.max_attempts} – "
                    f"{method} {endpoint}"
                )

                try:
                    if use_json_body:
                        response = await client.request(
                            method=method,
                            url=endpoint,
                            headers=headers,
                            json=call.parameters,
                        )
                    else:
                        response = await client.request(
                            method=method,
                            url=endpoint,
                            headers=headers,
                            params=call.parameters,
                        )
                except httpx.TimeoutException as exc:
                    execution_time_ms = (time.perf_counter() - start_time) * 1000
                    logger.warning(
                        f"HTTP tool '{call.tool_name}' (call_id={call.call_id}) timed out "
                        f"after {tool.timeout_seconds}s on attempt {attempt}"
                    )
                    # Timeout is non-retriable (the request already consumed the budget)
                    return ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status="timeout",
                        data=None,
                        error_message=f"HTTP request timed out after {tool.timeout_seconds}s: {exc}",
                        execution_time_ms=execution_time_ms,
                    )
                except httpx.RequestError as exc:
                    # Network-level error (DNS, connection refused, etc.)
                    last_error = f"HTTP request error: {exc}"
                    logger.warning(
                        f"HTTP tool '{call.tool_name}' (call_id={call.call_id}) network error "
                        f"on attempt {attempt}: {exc}"
                    )
                    # Treat like a 503 – retry if budget allows
                    if attempt >= self.retry_policy.max_attempts:
                        break
                    continue

                # Log response details
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"HTTP tool '{call.tool_name}' (call_id={call.call_id}): "
                    f"response status={response.status_code}, "
                    f"content-type={response.headers.get('content-type', 'unknown')}, "
                    f"elapsed={execution_time_ms:.2f}ms"
                )

                # Non-2xx response
                if not (200 <= response.status_code < 300):
                    last_error = (
                        f"HTTP {response.status_code} {response.reason_phrase}: "
                        f"{response.text[:500]}"
                    )
                    logger.warning(
                        f"HTTP tool '{call.tool_name}' (call_id={call.call_id}): "
                        f"non-2xx status {response.status_code} on attempt {attempt}"
                    )
                    if self.retry_policy.should_retry(response.status_code) and attempt < self.retry_policy.max_attempts:
                        continue
                    # Not retriable or exhausted – return error
                    return ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status="error",
                        data=None,
                        error_message=last_error,
                        execution_time_ms=execution_time_ms,
                    )

                # Success – parse response body
                try:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        data: Any = response.json()
                    else:
                        data = response.text
                except Exception as parse_exc:
                    logger.warning(
                        f"HTTP tool '{call.tool_name}' (call_id={call.call_id}): "
                        f"failed to parse response body: {parse_exc}"
                    )
                    data = response.text

                logger.info(
                    f"HTTP tool '{call.tool_name}' (call_id={call.call_id}) succeeded "
                    f"with status {response.status_code} in {execution_time_ms:.2f}ms"
                )
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    status="success",
                    data=data,
                    error_message=None,
                    execution_time_ms=execution_time_ms,
                )

        # All attempts exhausted
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"HTTP tool '{call.tool_name}' (call_id={call.call_id}) failed after "
            f"{attempt} attempt(s). Last error: {last_error}"
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="error",
            data=None,
            error_message=last_error or f"HTTP tool failed after {attempt} attempt(s)",
            execution_time_ms=execution_time_ms,
        )
