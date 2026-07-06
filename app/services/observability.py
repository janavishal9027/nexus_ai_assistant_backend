"""Observability counters and Prometheus text exposition (req 16.5, 16.6).

A single process-wide `observability` collector accumulates counters that are
rendered at GET /api/agent/metrics in Prometheus text format.
"""
from dataclasses import dataclass


@dataclass
class Counters:
    ws_sessions_active: int = 0
    kafka_events_published: int = 0
    redis_hits: int = 0
    redis_misses: int = 0
    memory_chunks_stored: int = 0
    memory_searches: int = 0


class ObservabilityCollector:
    """Accumulates counters exposed to Prometheus.

    Mutations happen on the asyncio event loop thread (single-threaded), so
    plain integer increments are sufficient.
    """

    def __init__(self) -> None:
        self.counters = Counters()

    def inc_ws_sessions(self, delta: int = 1) -> None:
        self.counters.ws_sessions_active += delta

    def inc_kafka_events(self) -> None:
        self.counters.kafka_events_published += 1

    def inc_redis_hit(self) -> None:
        self.counters.redis_hits += 1

    def inc_redis_miss(self) -> None:
        self.counters.redis_misses += 1

    def inc_memory_stored(self, count: int = 1) -> None:
        self.counters.memory_chunks_stored += count

    def inc_memory_search(self) -> None:
        self.counters.memory_searches += 1

    def to_prometheus_text(self) -> str:
        c = self.counters
        lines = [
            "# HELP agent_ws_sessions_active Active WebSocket sessions",
            "# TYPE agent_ws_sessions_active gauge",
            f"agent_ws_sessions_active {c.ws_sessions_active}",
            "# HELP agent_kafka_events_published_total Kafka events published",
            "# TYPE agent_kafka_events_published_total counter",
            f"agent_kafka_events_published_total {c.kafka_events_published}",
            "# HELP agent_redis_hits_total Redis cache hits",
            "# TYPE agent_redis_hits_total counter",
            f"agent_redis_hits_total {c.redis_hits}",
            "# HELP agent_redis_misses_total Redis cache misses",
            "# TYPE agent_redis_misses_total counter",
            f"agent_redis_misses_total {c.redis_misses}",
            "# HELP agent_memory_chunks_stored_total Memory chunks stored",
            "# TYPE agent_memory_chunks_stored_total counter",
            f"agent_memory_chunks_stored_total {c.memory_chunks_stored}",
            "# HELP agent_memory_searches_total Memory searches performed",
            "# TYPE agent_memory_searches_total counter",
            f"agent_memory_searches_total {c.memory_searches}",
        ]
        return "\n".join(lines) + "\n"


# Process-wide singleton
observability = ObservabilityCollector()
