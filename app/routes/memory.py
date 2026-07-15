"""Memory lifecycle endpoints (Part D Phase 3) — view / export / clear the user's
memory. All owner-scoped by the authenticated account.
"""
import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from ..models.db_models import Account
from ..services.auth import get_current_account
from ..memory import data_lifecycle, semantic

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("")
@router.get("/")
def memory_summary(account: Account = Depends(get_current_account)):
    """Counts + the user's top skills (for a Memory & Privacy settings view)."""
    return {
        "counts": data_lifecycle.summary(account.id),
        "skills": semantic.list_skills(account.id, limit=50),
    }


@router.get("/export")
def memory_export(account: Account = Depends(get_current_account)):
    """Download the user's full memory as JSON (data portability)."""
    data = data_lifecycle.export_memory(account.id)
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": 'attachment; filename="nexus-memory.json"'},
    )


@router.delete("")
@router.delete("/")
def memory_purge(scope: str = Query("all"),
                 account: Account = Depends(get_current_account)):
    """Clear the user's memory. scope ∈ {all, episodic, skills, feedback}."""
    return {"purged": data_lifecycle.purge_memory(account.id, scope)}
