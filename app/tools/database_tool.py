"""Database Tool — query_database (req 7).

The LLM turns a natural-language description into a parameterized SELECT. The tool
enforces, in order: syntax sanity → SELECT-only (DML/DDL rejection) → no
unparameterized string literals → execution via SQLAlchemy text() with empty
params. Errors return exact messages; DB exceptions are sanitized and never leak
partial results (req 7.4 / 19.2 / 19.3).
"""
import logging
import re
from datetime import datetime, timezone

from ..services.tool_registry import tool_registry

logger = logging.getLogger(__name__)

MAX_ROWS_HARD_CAP = 500  # req 7.5

_FORBIDDEN_KEYWORDS = frozenset({
    "insert", "update", "delete", "drop", "truncate", "alter",
    "create", "replace", "merge", "exec", "execute", "grant", "revoke",
})
# Any single-quoted string literal is treated as unparameterized interpolation.
_INTERPOLATION_PATTERN = re.compile(r"'[^']*'")
_ONLY_SELECT_MSG = "Only SELECT queries are permitted"
_INTERP_MSG = "Parameterization required: direct value interpolation detected"
_PARSE_FAIL_MSG = "Could not generate a valid SQL query from the provided description"
_SYNTAX_MSG = "Generated SQL is syntactically invalid"


def _validate_sql_keywords(sql: str) -> tuple[bool, str | None]:
    """Return (allowed, error). Rejects any forbidden DML/DDL keyword (req 7.3)."""
    lowered = sql.lower()
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            return False, _ONLY_SELECT_MSG
    return True, None


def _validate_interpolation(sql: str) -> tuple[bool, str | None]:
    """Return (allowed, error). Rejects unparameterized string literals (req 7.7)."""
    if _INTERPOLATION_PATTERN.search(sql):
        return False, _INTERP_MSG
    return True, None


def _validate_syntax(sql: str) -> tuple[bool, str | None]:
    """Light structural check: must be a non-empty SELECT/WITH statement."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False, _SYNTAX_MSG
    head = stripped.lower()
    if not (head.startswith("select") or head.startswith("with")):
        return False, _SYNTAX_MSG
    return True, None


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


async def _llm_generate_sql(db, query_description: str) -> str:
    from ..services.fallback_router import route_chat
    from ..models.schemas import MessageDto

    schema_summary = (
        "Tables:\n"
        "- users(id, name, email, role, created_at, updated_at)\n"
        "- tasks(id, title, description, status, assignee_id, due_date, priority, completed_at, created_at)\n"
        "- conversations(id, title, created_at, updated_at)\n"
        "- messages(id, conversation_id, role, content, created_at)\n"
    )
    prompt = (
        "Translate the request into a single read-only SQL SELECT statement for PostgreSQL.\n"
        "Rules: SELECT only; no INSERT/UPDATE/DELETE/DDL; do NOT inline literal values "
        "as quoted strings; return ONLY the SQL with no commentary.\n\n"
        f"{schema_summary}\nRequest: {query_description}\n"
    )
    result = await route_chat(
        db=db, messages=[MessageDto(role="user", content=prompt)],
        temperature=0.0, max_tokens=512,
    )
    return _strip_code_fence(result.content or "")


@tool_registry.tool(
    name="query_database",
    description="Translate a natural-language description into a parameterized SELECT and return rows.",
    input_schema={
        "type": "object",
        "properties": {
            "query_description": {"type": "string"},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
        },
        "required": ["query_description"],
        "additionalProperties": False,
    },
    output_schema={"type": "object"},
    timeout_seconds=30.0,
)
async def query_database(query_description: str, max_rows: int = 100) -> dict:
    from sqlalchemy import text
    from ..database import SessionLocal
    import time

    max_rows = min(max_rows, MAX_ROWS_HARD_CAP)
    db = SessionLocal()
    try:
        sql = await _llm_generate_sql(db, query_description)
        if not sql:
            raise ValueError(_PARSE_FAIL_MSG)

        for validator in (_validate_syntax, _validate_sql_keywords, _validate_interpolation):
            ok, err = validator(sql)
            if not ok:
                raise ValueError(err)

        started = time.perf_counter()
        try:
            result = db.execute(text(sql), {})
            mappings = result.mappings().all()
        except Exception as exc:
            # Sanitized: omit table names / connection details, no partial rows (req 7.4)
            logger.error(f"[DatabaseTool] Query execution error: {exc}")
            raise ValueError("Database query failed while executing the generated SELECT")
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(f"[DatabaseTool] Executed SQL in {duration_ms:.0f}ms: {sql}")

        rows = [dict(m) for m in mappings]
        truncated = len(rows) > max_rows
        out = {
            "rows": rows[:max_rows],
            "row_count": min(len(rows), max_rows),
            "truncated": truncated,
            "source": "live",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        if truncated:
            out["total_available"] = len(rows)
        return out
    finally:
        db.close()
