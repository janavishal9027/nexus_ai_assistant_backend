"""Audit logging for privileged write operations (req 19.7).

Every create/update/complete/delete tool invocation records an AuditLog row with
the correlation id, tool name, acting user, target resource, and outcome. Audit
failures are logged but never propagated so they cannot break the operation.
"""
import logging
from typing import Optional

from . import request_context

logger = logging.getLogger(__name__)


def write_audit_log(
    tool_name: str,
    target_resource,
    outcome: str,
    acting_user_id: Optional[int] = None,
    correlation_id: Optional[str] = None,
) -> None:
    from ..database import SessionLocal
    from ..models.db_models import AuditLog

    db = SessionLocal()
    try:
        entry = AuditLog(
            correlation_id=correlation_id or request_context.get_correlation_id(),
            tool_name=tool_name,
            acting_user_id=acting_user_id if acting_user_id is not None else request_context.get_acting_user_id(),
            target_resource=str(target_resource) if target_resource is not None else None,
            outcome=outcome,
        )
        db.add(entry)
        db.commit()
    except Exception as exc:
        logger.warning(f"[Audit] Failed to write audit log for {tool_name}: {exc}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()
