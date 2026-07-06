"""Per-request context propagated to tools via contextvars.

The Agent_Orchestrator sets these at the start of a request. Because the
Tool_Executor runs each tool in a child task (which copies the current context),
tools can read the correlation id, acting user, freshness flag, and session id
without them being threaded through every function signature.
"""
import contextvars
from typing import Optional

_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_correlation_id", default=None
)
_acting_user_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "agent_acting_user_id", default=None
)
_requires_fresh_data: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agent_requires_fresh_data", default=False
)
_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_session_id", default=None
)
# Authenticated account id (used to scope LLM provider keys per user).
_owner_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "agent_owner_id", default=None
)


def set_request_context(
    *,
    correlation_id: Optional[str] = None,
    acting_user_id: Optional[int] = None,
    requires_fresh_data: bool = False,
    session_id: Optional[str] = None,
    owner_id: Optional[int] = None,
) -> None:
    _correlation_id.set(correlation_id)
    _acting_user_id.set(acting_user_id)
    _requires_fresh_data.set(bool(requires_fresh_data))
    _session_id.set(session_id)
    _owner_id.set(owner_id)


def set_owner_id(owner_id: Optional[int]):
    return _owner_id.set(owner_id)


def get_owner_id() -> Optional[int]:
    return _owner_id.get()


def set_correlation_id(correlation_id: Optional[str]):
    return _correlation_id.set(correlation_id)


def get_correlation_id() -> str:
    return _correlation_id.get() or "unknown"


def get_acting_user_id() -> Optional[int]:
    return _acting_user_id.get()


def requires_fresh_data() -> bool:
    return bool(_requires_fresh_data.get())


def get_session_id() -> Optional[str]:
    return _session_id.get()
