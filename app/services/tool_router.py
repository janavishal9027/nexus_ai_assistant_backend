"""Tool Router — classifies each Tool_Call and dispatches it (req 4).

Classification is a pure prefix function (never consults the registry, req 4.8).
Dispatch validates registry presence (req 4.7) and enabled status (req 19.5),
logs the routing decision with the resolved handler and correlation id (req 4.9),
then executes the tool through the existing Tool_Executor so all categories share
one battle-tested execution path (schema validation, timeout, concurrency,
error capture). The correlation id is stamped onto the ToolCall and ToolResult
(req 4.10).
"""
import logging
from typing import Literal, Optional

from .tool_registry import ToolRegistry, tool_registry as _default_registry
from .tool_executor import ToolExecutor
from .tool_models import ToolCall, ToolResult
from . import request_context

logger = logging.getLogger(__name__)

ToolCategory = Literal[
    "service_tool", "database_tool", "memory_tool", "realtime_tool", "external_tool"
]

# Static prefix → category map (req 4.1). Order does not matter: prefixes are disjoint.
_PREFIX_MAP: dict[str, ToolCategory] = {
    "user_": "service_tool",
    "task_": "service_tool",
    "query_": "database_tool",
    "memory_": "memory_tool",
    "realtime_": "realtime_tool",
}


def classify_tool(name: str) -> ToolCategory:
    """Pure prefix-based classification. Independent of registry state (req 4.8)."""
    for prefix, category in _PREFIX_MAP.items():
        if name.startswith(prefix):
            return category
    return "external_tool"


def _handler_name(category: ToolCategory, tool_name: str) -> str:
    if category == "service_tool":
        return "UserServiceTool" if tool_name.startswith("user_") else "TaskServiceTool"
    return {
        "database_tool": "DatabaseTool",
        "memory_tool": "MemoryServiceTool",
        "realtime_tool": "RealTimeEventsTool",
        "external_tool": "ToolExecutor",
    }[category]


class ToolRouter:
    """Classifies and dispatches Tool_Calls to the correct handler category."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ) -> None:
        self.registry = registry or _default_registry
        self._executor = tool_executor or ToolExecutor(self.registry)

    async def route(self, call: ToolCall, correlation_id: Optional[str] = None) -> ToolResult:
        category = classify_tool(call.tool_name)
        cid = correlation_id or call.correlation_id or request_context.get_correlation_id()

        tool_def = self.registry.get(call.tool_name)
        if tool_def is None:
            logger.info(
                f"[ToolRouter] tool={call.tool_name} category={category} "
                f"target=N/A correlation_id={cid} NOT_FOUND"
            )
            return ToolResult(
                call_id=call.call_id, tool_name=call.tool_name, status="error",
                data=None, error_message=f"Tool '{call.tool_name}' not found in registry",
                execution_time_ms=0.0, correlation_id=cid,
            )

        if not tool_def.enabled:
            logger.info(
                f"[ToolRouter] tool={call.tool_name} category={category} "
                f"target={_handler_name(category, call.tool_name)} correlation_id={cid} DISABLED"
            )
            return ToolResult(
                call_id=call.call_id, tool_name=call.tool_name, status="error",
                data=None, error_message=f"Tool '{call.tool_name}' is disabled",
                execution_time_ms=0.0, correlation_id=cid,
            )

        logger.info(
            f"[ToolRouter] tool={call.tool_name} category={category} "
            f"target={_handler_name(category, call.tool_name)} correlation_id={cid}"
        )

        call.correlation_id = cid
        token = request_context.set_correlation_id(cid)
        try:
            results = await self._executor.execute_batch([call], max_concurrent=1)
        finally:
            try:
                request_context._correlation_id.reset(token)
            except Exception:
                pass
        result = results[0]
        result.correlation_id = cid
        return result


# Module-level singleton (executor bound to the shared registry).
tool_router = ToolRouter()
