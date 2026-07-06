"""Real-Time Events Tool — realtime_get_state / realtime_recent_events (req 18).

Read-only access to live state (Redis) and recent events (Kafka event buffer) so
the agent can reason over what is happening right now. Both operations report the
tool as disabled when the kafka/redis_cache feature flags are off (req 18.8).
"""
import logging
from datetime import datetime, timezone

from ..services.tool_registry import tool_registry
from ..services.feature_flags import get_agent_features

logger = logging.getLogger(__name__)

_DISABLED_MSG = "Real-time features are disabled"
_EVENTS_UNAVAILABLE_MSG = "Real-time event source unavailable"
_STATE_UNAVAILABLE_MSG = "Live state source unavailable"
RECENT_DEFAULT = 20
RECENT_MAX = 100  # req 18.3


def _require_enabled() -> None:
    flags = get_agent_features()
    if not (flags.kafka and flags.redis_cache):
        raise ValueError(_DISABLED_MSG)


@tool_registry.tool(
    name="realtime_get_state",
    description="Read the current live state for a key from the real-time state store.",
    input_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=5.0,
)
async def realtime_get_state(key: str) -> dict:
    from ..services.redis_cache import get_redis_cache

    _require_enabled()
    cache = get_redis_cache()
    if cache is None:
        raise ValueError(_STATE_UNAVAILABLE_MSG)

    value = await cache.get(key)
    as_of = datetime.now(timezone.utc).isoformat()
    logger.info(f"[RealTime] realtime_get_state key={key} found={value is not None} "
                f"correlation_id={_cid()}")
    if value is None:
        raise ValueError(f"No live state for '{key}'")
    return {"key": key, "value": value, "as_of": as_of, "source": "live"}


@tool_registry.tool(
    name="realtime_recent_events",
    description="Return the most recent real-time events for a topic, newest-first.",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        },
        "required": ["topic"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=5.0,
)
async def realtime_recent_events(topic: str, limit: int = RECENT_DEFAULT) -> dict:
    from ..services.kafka_consumer import event_buffer
    from ..services.kafka_producer import kafka_producer

    _require_enabled()
    limit = min(max(int(limit), 1), RECENT_MAX)

    # If the Kafka consumer never started there is no live event source (req 18.7).
    if kafka_producer is None and not event_buffer.topics():
        raise ValueError(_EVENTS_UNAVAILABLE_MSG)

    events = event_buffer.get_recent(topic, limit)
    logger.info(f"[RealTime] realtime_recent_events topic={topic} count={len(events)} "
                f"correlation_id={_cid()}")
    return {"topic": topic, "events": events, "count": len(events),
            "source": "live", "fetched_at": datetime.now(timezone.utc).isoformat()}


def _cid() -> str:
    from ..services import request_context
    return request_context.get_correlation_id()
