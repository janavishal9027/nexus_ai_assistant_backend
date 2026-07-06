"""Agent feature flags parsed from the AGENT_FEATURES environment variable.

`AGENT_FEATURES` is a comma-separated list drawn from: planner, redis_cache,
kafka, fcm, websocket. When a flag is absent the corresponding feature is
disabled and the system falls back to the equivalent existing behavior
(req 15.7, 15.8).
"""
from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class AgentFeatures:
    planner: bool
    redis_cache: bool
    kafka: bool
    fcm: bool
    websocket: bool

    def any_enabled(self) -> bool:
        return any((self.planner, self.redis_cache, self.kafka, self.fcm, self.websocket))


def _parse(raw: str) -> AgentFeatures:
    enabled = {f.strip().lower() for f in (raw or "").split(",") if f.strip()}
    return AgentFeatures(
        planner="planner" in enabled,
        redis_cache="redis_cache" in enabled,
        kafka="kafka" in enabled,
        fcm="fcm" in enabled,
        websocket="websocket" in enabled,
    )


@lru_cache(maxsize=1)
def get_agent_features() -> AgentFeatures:
    """Parse AGENT_FEATURES once per process. Reads the env var directly so it
    works even before Settings is constructed."""
    raw = os.environ.get("AGENT_FEATURES")
    if raw is None:
        # Fall back to Settings (which loads .env) when the raw env var is unset.
        try:
            from ..config import get_settings
            raw = get_settings().agent_features
        except Exception:
            raw = ""
    return _parse(raw)
