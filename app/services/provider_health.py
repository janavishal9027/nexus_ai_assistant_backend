"""Provider key health — classify router failures and persist them.

`ApiKey.status` has existed since the beginning and nothing ever wrote it: it
was set to "unknown" when a key was added and stayed there forever, while the
UI faithfully displayed "unknown" and every request 402'd. This module is the
missing half.

Two things depend on it:

1. **The user** — Settings can say "OpenRouter: out of credits" instead of a
   permanently-unknown dot, which is the difference between "the app is broken"
   and "top up your account".
2. **The router** — it already filters keys on `status != "error"`, so once
   health is recorded a dead provider is skipped *before* it burns any of the
   20s fallback budget. That budget is why a chat could exhaust OpenRouter's
   402s and never reach the 9 other providers with working keys.

Classification is deliberately conservative: only auth/credit failures (which
persist until a human acts) mark a key `error`. Rate limits are transient and
recorded as `limited` without taking the provider out of rotation. Anything else
(a bad model id, a 5xx, a timeout) says nothing about the key and is ignored.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..models.db_models import ApiKey

logger = logging.getLogger(__name__)

HEALTHY, ERROR, LIMITED, UNKNOWN = "healthy", "error", "limited", "unknown"

# The key itself is broken/unfunded — a human must fix it. Note "402" and the
# credit wording: OpenRouter answers 402 "requires more credits", which matched
# none of the router's original auth checks, so it was treated as a per-model
# retry and walked all 351 models until the budget died.
_KEY_FATAL = (
    "401", "402", "403",
    "unauthorized", "forbidden", "permission", "invalid api key",
    "authentication", "api key not valid", "reported as leaked",
    "payment required", "billing", "credit card", "customer_verification",
    "more credits", "insufficient credit", "insufficient_quota",
    "out of credit", "exceeded your current quota",
)

# Transient: the key is fine, we're just going too fast / out of period quota.
_RATE_LIMITED = (
    "429", "rate limit", "rate-limit", "too many requests",
    "resource_exhausted", "quota",
)


def classify(error: object) -> Optional[str]:
    """ERROR / LIMITED for failures that say something about the KEY, else None.

    None means "this tells us nothing about the key" — a bad model id or a
    provider 500 must not condemn an otherwise-good key.
    """
    msg = str(error or "").lower()
    if not msg:
        return None
    # Rate-limit first: "exceeded your current quota" contains both "quota"
    # (transient) and credit wording, but a 429 is not a dead key.
    if any(t in msg for t in ("429", "too many requests", "rate limit",
                             "rate-limit", "resource_exhausted")):
        return LIMITED
    if any(t in msg for t in _KEY_FATAL):
        return ERROR
    if any(t in msg for t in _RATE_LIMITED):
        return LIMITED
    return None


def _short(error: object, limit: int = 300) -> str:
    text = " ".join(str(error or "").split())
    return text[:limit]


def record_failure(db: Session, key_id: int, error: object) -> Optional[str]:
    """Persist what a failed attempt says about the key. Returns the new status
    (or None if the failure was uninformative). Never raises — health bookkeeping
    must not break a chat."""
    status = classify(error)
    if status is None:
        return None
    try:
        key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
        if key is None:
            return None
        key.status = status
        key.last_error = _short(error)
        key.last_checked_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"[Health] key #{key_id} ({key.platform}) → {status}: "
                    f"{key.last_error[:80]}")
        return status
    except Exception as exc:
        db.rollback()
        logger.warning(f"[Health] could not record failure for key #{key_id}: {exc}")
        return None


def record_success(db: Session, key_id: int) -> None:
    """A working call clears any past error. Never raises."""
    try:
        key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
        if key is None:
            return
        if key.status == HEALTHY and key.last_error is None:
            key.last_checked_at = datetime.now(timezone.utc)
        else:
            key.status = HEALTHY
            key.last_error = None
            key.last_checked_at = datetime.now(timezone.utc)
            logger.info(f"[Health] key #{key_id} ({key.platform}) → healthy")
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning(f"[Health] could not record success for key #{key_id}: {exc}")


def usable_filter(cooldown_minutes: int = 30):
    """SQLAlchemy condition for "this key is worth trying".

    A key marked `error` is skipped — but only until the cooldown lapses, then
    it gets one more chance. Without that, a single blip (or a top-up the app
    can't see) would bench a provider permanently with no way back except the
    UI's Test button.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, cooldown_minutes))
    return or_(
        ApiKey.status != ERROR,
        ApiKey.last_checked_at.is_(None),
        ApiKey.last_checked_at < cutoff,
    )


def summary(db: Session, owner_id: Optional[int] = None) -> list[dict]:
    """Per-key health for the settings view. Never exposes the key itself."""
    q = db.query(ApiKey)
    if owner_id is not None:
        q = q.filter(or_(ApiKey.owner_id == owner_id, ApiKey.owner_id.is_(None)))
    out = []
    for k in q.order_by(ApiKey.platform).all():
        out.append({
            "id": k.id,
            "platform": k.platform,
            "label": k.label or "",
            "enabled": bool(k.enabled),
            "status": k.status or UNKNOWN,
            "last_error": k.last_error,
            "last_checked_at": k.last_checked_at.isoformat() if k.last_checked_at else None,
        })
    return out
