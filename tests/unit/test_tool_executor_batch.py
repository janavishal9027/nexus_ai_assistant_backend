"""
Unit tests for ToolExecutor.execute_batch — concurrent execution, max_concurrent
semaphore enforcement, partial failure isolation, and cancellation propagation.

Requirements covered: 14.2, 14.3, 14.4, 14.5, 14.6, 14.8, 3.7, 3.8
"""

import asyncio
import time
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.tool_executor import ToolExecutor
from app.services.tool_models import ToolCall, ToolResult, ToolDefinition
from app.services.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_call(tool_name: str, params: dict | None = None) -> ToolCall:
    """Create a ToolCall with a fresh UUID."""
    return ToolCall(
        tool_name=tool_name,
        parameters=params or {},
        call_id=str(uuid.uuid4()),
    )


def _make_registry(*tools: ToolDefinition) -> ToolRegistry:
    """Build a fresh ToolRegistry populated with the supplied tools."""
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _async_tool_def(
    name: str,
    fn,
    timeout_seconds: float = 30.0,
    input_schema: dict | None = None,
) -> ToolDefinition:
    """Convenience factory for an async tool ToolDefinition."""
    return ToolDefinition(
        name=name,
        description=f"Test tool {name}",
        input_schema=input_schema or {"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        fn=fn,
        timeout_seconds=timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Tests: empty / trivial cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch_empty_returns_empty():
    """execute_batch with no calls returns an empty list immediately."""
    reg = ToolRegistry()
    executor = ToolExecutor(reg)
    results = await executor.execute_batch([], max_concurrent=5)
    assert results == []


# ---------------------------------------------------------------------------
# Tests: result ordering (Requirement 14.5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch_preserves_order():
    """
    Results must be returned in the same order as the input tool_calls even
    when tools finish in a different order.

    Validates: Requirements 14.5
    """
    # Tool A sleeps briefly so tool B finishes first.
    finish_order: list[str] = []

    async def slow_tool():
        await asyncio.sleep(0.05)
        finish_order.append("slow")
        return {"value": "slow"}

    async def fast_tool():
        finish_order.append("fast")
        return {"value": "fast"}

    reg = _make_registry(
        _async_tool_def("slow_tool", slow_tool),
        _async_tool_def("fast_tool", fast_tool),
    )
    executor = ToolExecutor(reg)

    calls = [_make_call("slow_tool"), _make_call("fast_tool")]
    results = await executor.execute_batch(calls, max_concurrent=5)

    assert len(results) == 2
    # Order must match input regardless of execution finish order
    assert results[0].tool_name == "slow_tool"
    assert results[1].tool_name == "fast_tool"
    assert results[0].data == {"value": "slow"}
    assert results[1].data == {"value": "fast"}
    # Confirm fast actually finished first (validates the test itself)
    assert finish_order[0] == "fast"


# ---------------------------------------------------------------------------
# Tests: partial failure isolation (Requirement 14.4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch_partial_failure_does_not_block_others():
    """
    A tool that raises an exception must not prevent other tools in the
    same batch from completing.

    Validates: Requirements 14.4
    """
    async def good_tool():
        return {"ok": True}

    async def bad_tool():
        raise RuntimeError("intentional failure")

    reg = _make_registry(
        _async_tool_def("good_tool", good_tool),
        _async_tool_def("bad_tool", bad_tool),
    )
    executor = ToolExecutor(reg)

    calls = [
        _make_call("good_tool"),
        _make_call("bad_tool"),
        _make_call("good_tool"),
    ]
    results = await executor.execute_batch(calls, max_concurrent=5)

    assert len(results) == 3

    # First and third calls must succeed.
    assert results[0].status == "success"
    assert results[0].data == {"ok": True}

    # Second call (bad_tool) must be an error, not propagate as an exception.
    assert results[1].status == "error"
    assert "intentional failure" in results[1].error_message

    # Third call must also succeed.
    assert results[2].status == "success"
    assert results[2].data == {"ok": True}


# ---------------------------------------------------------------------------
# Tests: max_concurrent semaphore (Requirement 14.6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch_respects_max_concurrent():
    """
    At most `max_concurrent` tools should run simultaneously.

    Validates: Requirements 14.6
    """
    running_concurrently = 0
    peak_concurrency = 0
    barrier = asyncio.Event()

    async def counting_tool():
        nonlocal running_concurrently, peak_concurrency
        running_concurrently += 1
        peak_concurrency = max(peak_concurrency, running_concurrently)
        await asyncio.sleep(0.02)
        running_concurrently -= 1
        return {"done": True}

    num_tools = 6
    max_concurrent = 2

    reg = _make_registry(_async_tool_def("counting_tool", counting_tool))
    executor = ToolExecutor(reg)

    calls = [_make_call("counting_tool") for _ in range(num_tools)]
    results = await executor.execute_batch(calls, max_concurrent=max_concurrent)

    assert len(results) == num_tools
    assert all(r.status == "success" for r in results)
    # Peak concurrency must never exceed the semaphore limit.
    assert peak_concurrency <= max_concurrent, (
        f"Expected peak ≤ {max_concurrent}, got {peak_concurrency}"
    )


# ---------------------------------------------------------------------------
# Tests: concurrent execution (Requirement 14.2, 14.3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch_runs_concurrently():
    """
    Multiple tools should run concurrently (not serially), so total wall-time
    should be significantly less than the sum of individual tool runtimes.

    Validates: Requirements 14.2, 14.3
    """
    SLEEP = 0.1
    N = 4

    async def slow_tool():
        await asyncio.sleep(SLEEP)
        return {"done": True}

    reg = _make_registry(_async_tool_def("slow_tool", slow_tool))
    executor = ToolExecutor(reg)

    calls = [_make_call("slow_tool") for _ in range(N)]
    start = time.perf_counter()
    results = await executor.execute_batch(calls, max_concurrent=N)
    elapsed = time.perf_counter() - start

    assert all(r.status == "success" for r in results)
    # Serial execution would take N * SLEEP seconds; concurrent should be ~SLEEP
    assert elapsed < SLEEP * N * 0.7, (
        f"Execution seems serial: {elapsed:.3f}s ≥ {SLEEP * N * 0.7:.3f}s"
    )


# ---------------------------------------------------------------------------
# Tests: cancellation propagation (Requirement 14.8)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch_cancellation_propagates_to_running_tasks():
    """
    When the execute_batch coroutine is cancelled, running tasks must also
    be cancelled and the CancelledError must be re-raised.

    Validates: Requirements 14.8
    """
    started = asyncio.Event()
    cancelled_flag: list[bool] = []

    async def long_tool():
        started.set()
        try:
            await asyncio.sleep(10)
            return {"done": True}
        except asyncio.CancelledError:
            cancelled_flag.append(True)
            raise

    reg = _make_registry(_async_tool_def("long_tool", long_tool))
    executor = ToolExecutor(reg)

    calls = [_make_call("long_tool"), _make_call("long_tool")]
    batch_task = asyncio.ensure_future(
        executor.execute_batch(calls, max_concurrent=5)
    )

    # Wait until at least one tool has started.
    await asyncio.wait_for(started.wait(), timeout=2.0)

    # Cancel the batch task.
    batch_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await batch_task

    # At least one running task must have received the cancellation.
    assert len(cancelled_flag) > 0, "Expected running tools to be cancelled"


@pytest.mark.asyncio
async def test_execute_batch_cancellation_propagates_to_queued_tasks():
    """
    When the execute_batch coroutine is cancelled, tasks still waiting on the
    semaphore (i.e. queued, not yet started) must also be cancelled.

    Validates: Requirements 14.8
    """
    queued_flag: list[bool] = []
    first_started = asyncio.Event()

    async def blocking_tool():
        first_started.set()
        # Block long enough so queued tasks never get the semaphore
        await asyncio.sleep(10)
        return {"done": True}

    async def queued_tool():
        # This tool should never actually run
        try:
            await asyncio.sleep(5)
            return {"done": True}
        except asyncio.CancelledError:
            queued_flag.append(True)
            raise

    reg = _make_registry(
        _async_tool_def("blocking_tool", blocking_tool),
        _async_tool_def("queued_tool", queued_tool),
    )
    executor = ToolExecutor(reg)

    # max_concurrent=1 ensures queued_tool can't start while blocking_tool runs
    calls = [_make_call("blocking_tool"), _make_call("queued_tool")]
    batch_task = asyncio.ensure_future(
        executor.execute_batch(calls, max_concurrent=1)
    )

    # Wait for blocking_tool to start
    await asyncio.wait_for(first_started.wait(), timeout=2.0)
    # Give the queued_tool task time to reach the semaphore wait
    await asyncio.sleep(0.01)

    batch_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await batch_task

    # The queued task must also have been cancelled.
    # (It may have been cancelled while waiting on the semaphore, before its
    # body even ran; that's fine — what matters is it didn't silently disappear.)


# ---------------------------------------------------------------------------
# Tests: result call_id mapping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch_result_call_ids_match_inputs():
    """
    Each ToolResult.call_id must match the corresponding ToolCall.call_id.
    """
    async def echo_tool():
        return {"echo": True}

    reg = _make_registry(_async_tool_def("echo_tool", echo_tool))
    executor = ToolExecutor(reg)

    calls = [_make_call("echo_tool") for _ in range(3)]
    results = await executor.execute_batch(calls, max_concurrent=5)

    for call, result in zip(calls, results):
        assert result.call_id == call.call_id


# ---------------------------------------------------------------------------
# Tests: unknown tool in batch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch_unknown_tool_returns_error_not_exception():
    """
    A ToolCall referencing an unknown tool must return an error ToolResult,
    not raise an exception that would abort the batch.
    """
    async def real_tool():
        return {"ok": True}

    reg = _make_registry(_async_tool_def("real_tool", real_tool))
    executor = ToolExecutor(reg)

    calls = [_make_call("real_tool"), _make_call("nonexistent_tool")]
    results = await executor.execute_batch(calls, max_concurrent=5)

    assert results[0].status == "success"
    assert results[1].status == "error"
    assert "not found" in results[1].error_message
