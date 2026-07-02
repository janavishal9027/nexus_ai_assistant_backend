import time
import logging

logger = logging.getLogger(__name__)

# Cooldowns: key -> expiry timestamp
_cooldowns: dict[str, float] = {}
# Failure counts for escalation
_failure_counts: dict[str, int] = {}
# Penalty scores (decay over time)
_penalties: dict[str, tuple[int, float]] = {}  # key -> (penalty, last_hit_time)

TRANSIENT_COOLDOWN_S = 90
ESCALATED_COOLDOWN_S = 600
MAX_COOLDOWN_S = 86400


def _key(platform: str, model_id: str, key_id: int) -> str:
    return f"{platform}:{model_id}:{key_id}"


def is_on_cooldown(platform: str, model_id: str, key_id: int) -> bool:
    k = _key(platform, model_id, key_id)
    expiry = _cooldowns.get(k)
    if expiry is None:
        return False
    if time.time() > expiry:
        _cooldowns.pop(k, None)
        return False
    return True


def set_cooldown(platform: str, model_id: str, key_id: int):
    k = _key(platform, model_id, key_id)
    failures = _failure_counts.get(k, 0) + 1
    _failure_counts[k] = failures

    if failures <= 1:
        duration = TRANSIENT_COOLDOWN_S
    elif failures <= 3:
        duration = ESCALATED_COOLDOWN_S
    else:
        duration = MAX_COOLDOWN_S

    _cooldowns[k] = time.time() + duration
    logger.info(f"[RateLimit] Cooldown set for {k} ({duration}s, failures: {failures})")


def record_success(platform: str, model_id: str, key_id: int):
    k = _key(platform, model_id, key_id)
    count = _failure_counts.get(k, 0)
    if count <= 1:
        _failure_counts.pop(k, None)
    else:
        _failure_counts[k] = count - 1


def record_rate_limit_hit(platform: str, model_id: str, key_id: int):
    k = _key(platform, model_id, key_id)
    penalty, _ = _penalties.get(k, (0, 0.0))
    penalty = min(penalty + 3, 10)
    _penalties[k] = (penalty, time.time())


def get_penalty(platform: str, model_id: str, key_id: int) -> int:
    k = _key(platform, model_id, key_id)
    entry = _penalties.get(k)
    if entry is None:
        return 0
    penalty, last_hit = entry
    elapsed = time.time() - last_hit
    decay_steps = int(elapsed / 120)
    decayed = max(0, penalty - decay_steps)
    if decayed == 0:
        _penalties.pop(k, None)
        return 0
    return decayed


def is_retryable_error(error: Exception) -> bool:
    msg = str(error).lower()
    retryable_patterns = [
        "429", "rate limit", "too many requests", "quota",
        "timeout", "timed out", "503", "unavailable",
        "500", "internal server error", "connection",
        "404", "not found", "402", "payment",
        "empty", "no content",
    ]
    return any(p in msg for p in retryable_patterns)
