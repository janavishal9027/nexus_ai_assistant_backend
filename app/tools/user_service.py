"""User Service Tool — user_get / user_list / user_create / user_update (req 5).

Registered functions receive validated kwargs from the Tool_Executor. Error
conditions are signalled by raising with the exact required message; the executor
converts them into a ToolResult(status="error"). Success returns a JSON-safe dict.
UserDto serialization guarantees no password/api_key fields leak (req 5.9 / 19.4).
"""
import asyncio
import logging
from datetime import datetime, timezone

from ..services.tool_registry import tool_registry
from ..services import request_context
from ..services.audit import write_audit_log

logger = logging.getLogger(__name__)

DB_SOFT_TIMEOUT_S = 2.0  # req 5.2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_dict(user) -> dict:
    """Serialize a User ORM row through UserDto → JSON-safe dict (no secrets)."""
    from ..models.schemas import UserDto
    from ..services.agent import _strip_sensitive_data
    data = UserDto.model_validate(user).model_dump(mode="json")
    return _strip_sensitive_data(data)


@tool_registry.tool(
    name="user_get",
    description="Retrieve a user record by user_id.",
    input_schema={
        "type": "object",
        "properties": {"user_id": {"type": "integer", "description": "User ID"}},
        "required": ["user_id"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def user_get(user_id: int) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import User
    from ..services.redis_cache import get_redis_cache, USER_TTL
    from ..services.observability import observability

    cache = get_redis_cache()
    key = f"user:{user_id}"
    fresh = request_context.requires_fresh_data()

    # Cache read (bypassed when fresh data is required — req 20.1/20.3)
    if cache is not None and not fresh:
        cached = await cache.get(key)
        if cached is not None:
            observability.inc_redis_hit()
            fetched_at = cached.get("fetched_at")
            age = None
            if fetched_at:
                try:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds()
                except Exception:
                    age = None
            return {**cached, "source": "cache", "age_seconds": age}
        observability.inc_redis_miss()

    def _query() -> dict | None:
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return _user_dict(user) if user else None
        finally:
            db.close()

    # Soft 2 s budget: on overrun return available data with degraded=True (req 5.2)
    try:
        data = await asyncio.wait_for(asyncio.to_thread(_query), timeout=DB_SOFT_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(f"[UserService] user_get({user_id}) exceeded {DB_SOFT_TIMEOUT_S}s; degraded")
        return {"id": user_id, "degraded": True, "source": "live", "fetched_at": _now_iso()}

    if data is None:
        raise ValueError(f"User {user_id} not found")

    data = {**data, "source": "live", "fetched_at": _now_iso()}
    if cache is not None:
        await cache.set(key, data, ttl=USER_TTL)
    return data


@tool_registry.tool(
    name="user_list",
    description="List users with pagination.",
    input_schema={
        "type": "object",
        "properties": {
            "page": {"type": "integer", "minimum": 1, "default": 1},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        },
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def user_list(page: int = 1, page_size: int = 20) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import User

    def _query():
        db = SessionLocal()
        try:
            total = db.query(User).count()
            rows = (
                db.query(User).order_by(User.id.asc())
                .offset((page - 1) * page_size).limit(page_size).all()
            )
            return total, [_user_dict(u) for u in rows]
        finally:
            db.close()

    total, items = await asyncio.to_thread(_query)
    return {"items": items, "page": page, "page_size": page_size, "total_count": total,
            "source": "live", "fetched_at": _now_iso()}


@tool_registry.tool(
    name="user_create",
    description="Create a new user with name, email, and role.",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "role": {"type": "string"},
        },
        "required": ["name", "email", "role"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def user_create(name: str, email: str, role: str) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import User

    def _create():
        db = SessionLocal()
        try:
            if db.query(User).filter(User.email == email).first() is not None:
                raise ValueError(f"Email {email} is already registered")
            user = User(name=name, email=email, role=role)
            db.add(user)
            db.commit()
            db.refresh(user)
            data = _user_dict(user)
            write_audit_log("user_create", user.id, "success")
            return data
        finally:
            db.close()

    return await asyncio.to_thread(_create)


@tool_registry.tool(
    name="user_update",
    description="Update a user's name, email, and/or role. Only provided fields change.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "integer"},
            "name": {"type": "string"},
            "email": {"type": "string"},
            "role": {"type": "string"},
        },
        "required": ["user_id"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def user_update(user_id: int, **fields) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import User
    from ..services.redis_cache import get_redis_cache

    def _update():
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                raise ValueError(f"User {user_id} not found")
            for k in ("name", "email", "role"):
                if k in fields and fields[k] is not None:
                    setattr(user, k, fields[k])
            db.commit()
            db.refresh(user)
            data = _user_dict(user)
            write_audit_log("user_update", user_id, "success")
            return data
        finally:
            db.close()

    data = await asyncio.to_thread(_update)
    cache = get_redis_cache()
    if cache is not None:
        await cache.delete(f"user:{user_id}")  # invalidate (req 9.5)
    return {**data, "source": "live", "fetched_at": _now_iso()}
