"""Redis live-state cache with a hard 500 ms per-operation timeout (req 9).

Every public method is failure-transparent: on any error (connection loss,
timeout, auth, pool exhaustion, protocol error) it logs a WARNING and returns
None/False so callers treat it as a cache miss and fall back to the database
(req 9.7, 9.9, 9.11). `redis` is imported lazily so this module always imports
even when the package is absent (req 15.8).
"""
import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

OPERATION_TIMEOUT_S = 0.5     # req 9.9
SESSION_TTL = 1800            # 30 min (req 9.1)
USER_TTL = 300               # 5 min (req 9.4)
TOOLS_TTL = 60               # 1 min (req 9.6)

try:
    import redis.asyncio as aioredis  # type: ignore
    HAS_REDIS = True
except Exception:  # pragma: no cover
    aioredis = None  # type: ignore
    HAS_REDIS = False


class RedisCache:
    """Async Redis wrapper. Pool max 20 connections (req 9.8)."""

    def __init__(self, url: str) -> None:
        if not HAS_REDIS:
            raise RuntimeError(
                "redis package is not installed but the redis_cache feature is enabled"
            )
        # redis-py pools are lazy (no eager min-size); cap at 20 per req 9.8.
        self._pool = aioredis.ConnectionPool.from_url(
            url, max_connections=20, decode_responses=True
        )
        self._client = aioredis.Redis(connection_pool=self._pool)

    async def ping(self) -> bool:
        try:
            return bool(await asyncio.wait_for(self._client.ping(), timeout=OPERATION_TIMEOUT_S))
        except Exception as exc:
            logger.warning(f"[Redis] ping failed: {exc}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        try:
            raw = await asyncio.wait_for(self._client.get(key), timeout=OPERATION_TIMEOUT_S)
            return json.loads(raw) if raw else None
        except asyncio.TimeoutError:
            logger.warning(f"[Redis] get timeout for key={key}; treating as cache miss")
            return None
        except Exception as exc:
            logger.warning(f"[Redis] get error for key={key}: {exc}; treating as cache miss")
            return None

    async def set(self, key: str, value: Any, ttl: int) -> bool:
        try:
            serialized = json.dumps(value, default=str)
            await asyncio.wait_for(
                self._client.set(key, serialized, ex=ttl), timeout=OPERATION_TIMEOUT_S
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(f"[Redis] set timeout for key={key}")
            return False
        except Exception as exc:
            logger.warning(f"[Redis] set error for key={key}: {exc}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            await asyncio.wait_for(self._client.delete(key), timeout=OPERATION_TIMEOUT_S)
            return True
        except Exception as exc:
            logger.warning(f"[Redis] delete error for key={key}: {exc}")
            return False

    async def close(self) -> None:
        try:
            await self._pool.disconnect()
        except Exception:
            pass

    # --- Key helpers ---
    @staticmethod
    def session_key(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def user_key(user_id: int) -> str:
        return f"user:{user_id}"

    TOOLS_ENABLED_KEY = "tools:enabled"


# Singleton — assigned in main.py lifespan when the redis_cache flag is enabled.
redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> Optional[RedisCache]:
    return redis_cache
