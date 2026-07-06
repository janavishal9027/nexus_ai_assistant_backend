"""Task Service Tool — task_get / task_list / task_create / task_update /
task_complete (req 6). Same conventions as the User Service Tool: raise with the
exact message on error, return JSON-safe dicts on success, audit every write.
"""
import asyncio
import base64
import logging
from datetime import datetime, timezone
from typing import Optional

from ..services.tool_registry import tool_registry
from ..services.audit import write_audit_log

logger = logging.getLogger(__name__)

LIST_HARD_CAP = 100  # req 6.8


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_dict(task) -> dict:
    from ..models.schemas import TaskDto
    return TaskDto.model_validate(task).model_dump(mode="json")


def _parse_dt(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


@tool_registry.tool(
    name="task_get",
    description="Retrieve a task by task_id.",
    input_schema={
        "type": "object",
        "properties": {"task_id": {"type": "integer"}},
        "required": ["task_id"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def task_get(task_id: int) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import Task

    def _query():
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            return _task_dict(task)
        finally:
            db.close()

    data = await asyncio.to_thread(_query)
    return {**data, "source": "live", "fetched_at": _now_iso()}


@tool_registry.tool(
    name="task_list",
    description="List tasks with optional filters and pagination.",
    input_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "assignee_id": {"type": "integer"},
            "due_date_from": {"type": "string"},
            "due_date_to": {"type": "string"},
            "priority": {"type": "string"},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 100, "default": 100},
        },
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def task_list(
    status: Optional[str] = None,
    assignee_id: Optional[int] = None,
    due_date_from: Optional[str] = None,
    due_date_to: Optional[str] = None,
    priority: Optional[str] = None,
    page_size: int = 100,
) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import Task

    page_size = min(page_size, LIST_HARD_CAP)

    def _query():
        db = SessionLocal()
        try:
            q = db.query(Task)
            if status is not None:
                q = q.filter(Task.status == status)
            if assignee_id is not None:
                q = q.filter(Task.assignee_id == assignee_id)
            if priority is not None:
                q = q.filter(Task.priority == priority)
            dfrom = _parse_dt(due_date_from)
            dto = _parse_dt(due_date_to)
            if dfrom is not None:
                q = q.filter(Task.due_date >= dfrom)
            if dto is not None:
                q = q.filter(Task.due_date <= dto)
            total = q.count()
            rows = q.order_by(Task.id.asc()).limit(page_size).all()
            return total, [_task_dict(t) for t in rows]
        finally:
            db.close()

    total, items = await asyncio.to_thread(_query)
    # Always include pagination metadata (req 6.8 / Q8). Provide an opaque cursor
    # only when the result set was truncated at the hard cap.
    next_token = None
    if total > LIST_HARD_CAP:
        next_token = base64.urlsafe_b64encode(f"offset:{page_size}".encode()).decode()
    return {
        "items": items,
        "page_size": page_size,
        "total_count": total,
        "next_page_token": next_token,
        "source": "live",
        "fetched_at": _now_iso(),
    }


@tool_registry.tool(
    name="task_create",
    description="Create a task. 'title' is required.",
    input_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "assignee_id": {"type": "integer"},
            "due_date": {"type": "string"},
            "priority": {"type": "string"},
        },
        "required": ["title"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def task_create(
    title: str,
    description: Optional[str] = None,
    assignee_id: Optional[int] = None,
    due_date: Optional[str] = None,
    priority: str = "medium",
) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import Task

    missing = [f for f, v in (("title", title),) if not v]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    def _create():
        db = SessionLocal()
        try:
            task = Task(
                title=title,
                description=description,
                assignee_id=assignee_id,
                due_date=_parse_dt(due_date),
                priority=priority or "medium",
                status="pending",
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            data = _task_dict(task)
            write_audit_log("task_create", task.id, "success")
            return data
        finally:
            db.close()

    data = await asyncio.to_thread(_create)
    return {**data, "source": "live", "fetched_at": _now_iso()}


@tool_registry.tool(
    name="task_update",
    description="Update a task. Only provided fields change.",
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "status": {"type": "string"},
            "assignee_id": {"type": "integer"},
            "due_date": {"type": "string"},
            "priority": {"type": "string"},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def task_update(task_id: int, **fields) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import Task

    def _update():
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            for k in ("title", "description", "status", "assignee_id", "priority"):
                if k in fields and fields[k] is not None:
                    setattr(task, k, fields[k])
            if fields.get("due_date"):
                task.due_date = _parse_dt(fields["due_date"])
            db.commit()
            db.refresh(task)
            data = _task_dict(task)
            write_audit_log("task_update", task_id, "success")
            return data
        finally:
            db.close()

    data = await asyncio.to_thread(_update)
    return {**data, "source": "live", "fetched_at": _now_iso()}


@tool_registry.tool(
    name="task_complete",
    description="Mark a task complete, setting completed_at to now.",
    input_schema={
        "type": "object",
        "properties": {"task_id": {"type": "integer"}},
        "required": ["task_id"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=10.0,
)
async def task_complete(task_id: int) -> dict:
    from ..database import SessionLocal
    from ..models.db_models import Task

    def _complete():
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(task)
            data = _task_dict(task)
            write_audit_log("task_complete", task_id, "success")
            return data
        finally:
            db.close()

    data = await asyncio.to_thread(_complete)
    return {**data, "source": "live", "fetched_at": _now_iso()}
