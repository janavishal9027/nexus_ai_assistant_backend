"""
Unit tests for HTTP API tool execution capability in ToolExecutor.

Tests cover:
- RetryPolicy: configuration and behaviour helpers
- _is_http_tool: detection of HTTP vs regular tools
- _get_http_metadata: metadata extraction from both locations
- _inject_api_keys: env-var based API key injection
- _execute_http_tool: success, non-2xx error, timeout, retry, network error
- _execute_one routing: HTTP tools are dispatched to _execute_http_tool
"""

import asyncio
import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from app.services.tool_executor import RetryPolicy, ToolExecutor, _DEFAULT_RETRY_POLICY
from app.services.tool_models import ToolCall, ToolDefinition, ToolResult
from app.services.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_call(tool_name: str = "test_http_tool", params: dict | None = None) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        parameters=params or {},
        call_id=str(uuid.uuid4()),
    )


def _make_http_tool(
    endpoint: str = "https://api.example.com/data",
    method: str = "GET",
    headers: dict | None = None,
    auth_env_var: str | None = None,
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
    requires_auth: bool = False,
    timeout_seconds: float = 10.0,
    input_schema: dict | None = None,
) -> ToolDefinition:
    """Build a ToolDefinition whose input_schema carries x-metadata for HTTP dispatch."""
    metadata: dict = {"endpoint": endpoint, "method": method}
    if headers:
        metadata["headers"] = headers
    if auth_env_var:
        metadata["auth_env_var"] = auth_env_var
        metadata["auth_header"] = auth_header
        metadata["auth_scheme"] = auth_scheme

    base_schema = input_schema or {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": [],
    }
    base_schema["x-metadata"] = metadata

    return ToolDefinition(
        name="test_http_tool",
        description="An HTTP API tool for testing",
        input_schema=base_schema,
        output_schema={"type": "object"},
        fn=AsyncMock(return_value={"result": "should-not-be-called"}),
        enabled=True,
        requires_auth=requires_auth,
        timeout_seconds=timeout_seconds,
    )


def _make_registry_with(tool: ToolDefinition) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


# ---------------------------------------------------------------------------
# RetryPolicy tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_defaults(self):
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.backoff_factor == 0.5
        assert 429 in policy.retry_on_status
        assert 500 in policy.retry_on_status
        assert 404 not in policy.retry_on_status

    def test_should_retry_on_known_status(self):
        policy = RetryPolicy(retry_on_status={503})
        assert policy.should_retry(503) is True

    def test_should_not_retry_on_unknown_status(self):
        policy = RetryPolicy(retry_on_status={503})
        assert policy.should_retry(404) is False

    def test_wait_seconds_first_attempt(self):
        policy = RetryPolicy(backoff_factor=1.0)
        # Attempt 0 → no wait
        assert policy.wait_seconds(0) == 0.0

    def test_wait_seconds_exponential(self):
        policy = RetryPolicy(backoff_factor=1.0)
        # attempt 1 → 1.0 * 2^0 = 1.0
        assert policy.wait_seconds(1) == 1.0
        # attempt 2 → 1.0 * 2^1 = 2.0
        assert policy.wait_seconds(2) == 2.0
        # attempt 3 → 1.0 * 2^2 = 4.0
        assert policy.wait_seconds(3) == 4.0

    def test_custom_retry_on_status(self):
        policy = RetryPolicy(retry_on_status={408, 503})
        assert policy.should_retry(408)
        assert policy.should_retry(503)
        assert not policy.should_retry(500)


# ---------------------------------------------------------------------------
# _is_http_tool / _get_http_metadata
# ---------------------------------------------------------------------------


class TestHttpToolDetection:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

    def test_x_metadata_endpoint_detected(self):
        tool = _make_http_tool()
        assert ToolExecutor._is_http_tool(tool) is True

    def test_regular_tool_not_detected(self):
        tool = ToolDefinition(
            name="regular",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object"},
            fn=AsyncMock(),
        )
        assert ToolExecutor._is_http_tool(tool) is False

    def test_metadata_attribute_takes_priority(self):
        """If the tool carries a .metadata attribute, use that."""
        tool = _make_http_tool(endpoint="https://via-attribute.com")
        # Attach metadata dict directly as attribute (simulating extended dataclass)
        object.__setattr__(tool, "metadata", {"endpoint": "https://from-attribute.com/v2"})
        metadata = ToolExecutor._get_http_metadata(tool)
        assert metadata["endpoint"] == "https://from-attribute.com/v2"

    def test_x_metadata_fallback(self):
        tool = _make_http_tool(endpoint="https://api.example.com/test")
        metadata = ToolExecutor._get_http_metadata(tool)
        assert metadata["endpoint"] == "https://api.example.com/test"


# ---------------------------------------------------------------------------
# _inject_api_keys
# ---------------------------------------------------------------------------


class TestInjectApiKeys:
    def test_no_auth_env_var_returns_headers_unchanged(self):
        metadata: dict = {"endpoint": "https://api.example.com"}
        headers = {"Content-Type": "application/json"}
        result = ToolExecutor._inject_api_keys(headers, metadata)
        assert result == {"Content-Type": "application/json"}

    def test_injects_bearer_token(self):
        with patch.dict(os.environ, {"MY_API_KEY": "secret-token"}):
            metadata = {
                "endpoint": "https://api.example.com",
                "auth_env_var": "MY_API_KEY",
                "auth_header": "Authorization",
                "auth_scheme": "Bearer",
            }
            result = ToolExecutor._inject_api_keys({}, metadata)
        assert result["Authorization"] == "Bearer secret-token"

    def test_injects_token_with_custom_header(self):
        with patch.dict(os.environ, {"WEATHER_KEY": "abc123"}):
            metadata = {
                "endpoint": "https://weather.example.com",
                "auth_env_var": "WEATHER_KEY",
                "auth_header": "X-API-Key",
                "auth_scheme": "",  # no scheme prefix
            }
            result = ToolExecutor._inject_api_keys({}, metadata)
        assert result["X-API-Key"] == "abc123"

    def test_missing_env_var_logs_warning_and_skips(self):
        # Ensure the env var is absent
        env_without_key = {k: v for k, v in os.environ.items() if k != "NONEXISTENT_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            metadata = {
                "endpoint": "https://api.example.com",
                "auth_env_var": "NONEXISTENT_KEY",
            }
            result = ToolExecutor._inject_api_keys({}, metadata)
        # Should NOT inject anything
        assert "Authorization" not in result

    def test_does_not_mutate_original_headers(self):
        with patch.dict(os.environ, {"MY_KEY": "val"}):
            metadata = {"auth_env_var": "MY_KEY"}
            original = {"X-Custom": "value"}
            ToolExecutor._inject_api_keys(original, metadata)
        # original headers dict must not be modified
        assert "Authorization" not in original


# ---------------------------------------------------------------------------
# _execute_http_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecuteHttpTool:
    """Tests for the _execute_http_tool method using httpx mock transport."""

    def _make_executor(self, retry_policy: RetryPolicy | None = None) -> ToolExecutor:
        registry = ToolRegistry()
        return ToolExecutor(registry, retry_policy=retry_policy)

    def _mock_response(self, status_code: int, json_body: Any | None = None, text_body: str = "") -> httpx.Response:
        """Create a fake httpx.Response."""
        if json_body is not None:
            import json
            content = json.dumps(json_body).encode()
            headers = {"content-type": "application/json"}
        else:
            content = text_body.encode()
            headers = {"content-type": "text/plain"}
        return httpx.Response(status_code, content=content, headers=headers)

    @pytest.mark.asyncio
    async def test_success_json_response(self):
        tool = _make_http_tool(endpoint="https://api.example.com/data", method="GET")
        call = _make_call()
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=1))

        mock_resp = self._mock_response(200, json_body={"value": 42})
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=mock_resp):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "success"
        assert result.data == {"value": 42}
        assert result.error_message is None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_success_text_response(self):
        tool = _make_http_tool(method="GET")
        call = _make_call()
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=1))

        mock_resp = self._mock_response(200, text_body="plain text response")
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=mock_resp):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "success"
        assert result.data == "plain text response"

    @pytest.mark.asyncio
    async def test_non_2xx_returns_error(self):
        tool = _make_http_tool()
        call = _make_call()
        # Single attempt, 404 is not retriable by default
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=1))

        mock_resp = self._mock_response(404, text_body="Not Found")
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=mock_resp):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "error"
        assert "404" in result.error_message

    @pytest.mark.asyncio
    async def test_5xx_retried_and_eventually_succeeds(self):
        tool = _make_http_tool()
        call = _make_call()
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=3, backoff_factor=0.0))

        fail_resp = self._mock_response(503, text_body="Service Unavailable")
        success_resp = self._mock_response(200, json_body={"ok": True})

        side_effects = [fail_resp, fail_resp, success_resp]
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=side_effects):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "success"
        assert result.data == {"ok": True}

    @pytest.mark.asyncio
    async def test_5xx_all_retries_exhausted_returns_error(self):
        tool = _make_http_tool()
        call = _make_call()
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=2, backoff_factor=0.0))

        fail_resp = self._mock_response(503, text_body="Service Unavailable")
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=fail_resp):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "error"
        assert "503" in result.error_message

    @pytest.mark.asyncio
    async def test_network_error_retried(self):
        tool = _make_http_tool()
        call = _make_call()
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=3, backoff_factor=0.0))

        network_error = httpx.ConnectError("connection refused")
        success_resp = self._mock_response(200, json_body={"data": "ok"})

        side_effects = [network_error, network_error, success_resp]
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=side_effects):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_network_error_all_retries_exhausted(self):
        tool = _make_http_tool()
        call = _make_call()
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=2, backoff_factor=0.0))

        network_error = httpx.ConnectError("connection refused")
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=network_error):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "error"
        assert "HTTP request error" in result.error_message

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_status(self):
        tool = _make_http_tool(timeout_seconds=0.01)
        call = _make_call()
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=1))

        timeout_exc = httpx.ReadTimeout("timed out")
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=timeout_exc):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "timeout"
        assert "timed out" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_api_key_injected_into_request(self):
        """API key from env var should appear in the Authorization header sent."""
        tool = _make_http_tool(
            method="GET",
            auth_env_var="TEST_HTTP_API_KEY",
            auth_header="Authorization",
            auth_scheme="Token",
        )
        call = _make_call()
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=1))
        success_resp = self._mock_response(200, json_body={"ok": True})

        captured_headers: dict = {}

        async def mock_request(method, url, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return success_resp

        with patch.dict(os.environ, {"TEST_HTTP_API_KEY": "my-secret-key"}):
            with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=mock_request):
                result = await executor._execute_http_tool(call, tool)

        assert result.status == "success"
        assert captured_headers.get("Authorization") == "Token my-secret-key"

    @pytest.mark.asyncio
    async def test_post_uses_json_body(self):
        """POST method should send parameters as JSON body, not query params."""
        tool = _make_http_tool(method="POST")
        call = _make_call(params={"query": "hello world"})
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=1))
        success_resp = self._mock_response(200, json_body={"ok": True})

        captured_kwargs: dict = {}

        async def mock_request(method, url, **kwargs):
            captured_kwargs.update(kwargs)
            return success_resp

        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=mock_request):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "success"
        assert captured_kwargs.get("json") == {"query": "hello world"}
        assert "params" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_get_uses_query_params(self):
        """GET method should send parameters as query params, not JSON body."""
        tool = _make_http_tool(method="GET")
        call = _make_call(params={"city": "Berlin"})
        executor = self._make_executor(retry_policy=RetryPolicy(max_attempts=1))
        success_resp = self._mock_response(200, json_body={"temp": 20})

        captured_kwargs: dict = {}

        async def mock_request(method, url, **kwargs):
            captured_kwargs.update(kwargs)
            return success_resp

        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=mock_request):
            result = await executor._execute_http_tool(call, tool)

        assert result.status == "success"
        assert captured_kwargs.get("params") == {"city": "Berlin"}
        assert "json" not in captured_kwargs


# ---------------------------------------------------------------------------
# _execute_one routing – HTTP vs regular tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecuteOneRouting:
    """Verify _execute_one dispatches HTTP tools to _execute_http_tool."""

    @pytest.mark.asyncio
    async def test_http_tool_routed_to_execute_http_tool(self):
        tool = _make_http_tool(endpoint="https://api.example.com/v1")
        registry = _make_registry_with(tool)
        executor = ToolExecutor(registry, retry_policy=RetryPolicy(max_attempts=1))

        import json
        success_resp = httpx.Response(
            200,
            content=json.dumps({"routed": True}).encode(),
            headers={"content-type": "application/json"},
        )
        call = _make_call()

        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=success_resp):
            result = await executor._execute_one(call)

        assert result.status == "success"
        assert result.data == {"routed": True}

    @pytest.mark.asyncio
    async def test_regular_async_tool_not_routed_to_http(self):
        """A non-HTTP async tool should execute its fn directly."""
        async def my_fn(**kwargs):
            return {"answer": 42}

        tool = ToolDefinition(
            name="regular_tool",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object"},
            fn=my_fn,
            enabled=True,
        )
        registry = ToolRegistry()
        registry.register(tool)
        executor = ToolExecutor(registry)

        call = ToolCall(tool_name="regular_tool", parameters={}, call_id=str(uuid.uuid4()))
        result = await executor._execute_one(call)

        assert result.status == "success"
        assert result.data == {"answer": 42}

    @pytest.mark.asyncio
    async def test_schema_validation_runs_before_http_dispatch(self):
        """Invalid parameters should fail validation without making any HTTP request."""
        tool = _make_http_tool(
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
                "x-metadata": {
                    "endpoint": "https://api.example.com",
                    "method": "GET",
                },
            }
        )
        # Overwrite the x-metadata inside input_schema correctly:
        tool.input_schema["x-metadata"] = {"endpoint": "https://api.example.com", "method": "GET"}
        registry = _make_registry_with(tool)
        executor = ToolExecutor(registry, retry_policy=RetryPolicy(max_attempts=1))

        # Missing required 'q' parameter
        call = _make_call(params={})

        # If HTTP was dispatched, we'd need to mock httpx; this patch will
        # raise if any HTTP request is made (ensuring none is).
        with patch("httpx.AsyncClient.request", side_effect=AssertionError("HTTP request should not be made")):
            result = await executor._execute_one(call)

        assert result.status == "error"
        assert "Invalid parameters" in result.error_message
