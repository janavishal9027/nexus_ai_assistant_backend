"""Memory lifecycle endpoints (Part D Phase 3) — view / export / clear the user's
memory. All owner-scoped by the authenticated account.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models.db_models import Account
from ..services.auth import get_current_account
from ..memory import data_lifecycle, memory_graph, memory_prefs, semantic

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/memory", tags=["memory"])


class MemoryPrefsRequest(BaseModel):
    """Partial update — only the switches present are changed."""
    recall_enabled: Optional[bool] = None
    record_enabled: Optional[bool] = None
    reflect_enabled: Optional[bool] = None
    graph_enabled: Optional[bool] = None


@router.get("")
@router.get("/")
def memory_summary(account: Account = Depends(get_current_account)):
    """Counts + the user's top skills (for a Memory & Privacy settings view)."""
    return {
        "counts": data_lifecycle.summary(account.id),
        "skills": semantic.list_skills(account.id, limit=50),
        "prefs": memory_prefs.get_prefs(account.id),
    }


@router.get("/prefs")
def memory_prefs_get(account: Account = Depends(get_current_account)):
    """The user's memory switches. `prefs` are their raw choices; `effective` is
    what the chat path actually does (their choice AND the operator's flag)."""
    return {
        "prefs": memory_prefs.get_prefs(account.id),
        "effective": memory_prefs.effective(account.id),
    }


@router.patch("/prefs")
def memory_prefs_set(body: MemoryPrefsRequest,
                     account: Account = Depends(get_current_account)):
    """Turn memory layers on/off for this account. Partial: omitted switches are
    left as they are."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No preferences supplied")
    try:
        prefs = memory_prefs.set_prefs(account.id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save preferences: {e}")
    return {"prefs": prefs, "effective": memory_prefs.effective(account.id)}


@router.get("/graph")
def memory_graph_view(limit: int = Query(300, ge=1, le=2000),
                      account: Account = Depends(get_current_account)):
    """The user's personal memory graph (Part D Phase 5): people/orgs + tools/tech
    the assistant has learned about them, as nodes + edges (strongest first)."""
    return memory_graph.graph(account.id, limit=limit)


@router.get("/graph/neighbors")
def memory_graph_neighbors(entity: str = Query(..., min_length=1),
                           limit: int = Query(50, ge=1, le=500),
                           account: Account = Depends(get_current_account)):
    """All edges touching a specific entity in the user's personal graph."""
    return {"entity": entity, "edges": memory_graph.neighbors(account.id, entity, limit=limit)}


@router.get("/graph/query")
async def memory_graph_query(q: str = Query(..., min_length=1),
                             limit: int = Query(12, ge=1, le=100),
                             account: Account = Depends(get_current_account)):
    """Graph edges relevant to a query (semantic, falling back to keyword)."""
    return {"query": q, "edges": await memory_graph.query(account.id, q, limit=limit)}


@router.delete("/graph/{edge_id}")
def memory_graph_delete_edge(edge_id: int,
                             account: Account = Depends(get_current_account)):
    """Forget ONE edge — "that's wrong about me" without clearing the graph.
    404 when the edge doesn't exist or isn't yours (same response either way, so
    this can't be used to probe for other accounts' edge ids)."""
    if not memory_graph.delete_edge(account.id, edge_id):
        raise HTTPException(status_code=404, detail="Edge not found")
    return {"deleted": edge_id}


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
    """Clear the user's memory. scope ∈ {all, episodic, skills, feedback, graph}.

    An unrecognized scope is a 422, never a silent wipe: this endpoint is
    destructive and irreversible, so a client typo must not be read as "all".
    """
    if scope not in data_lifecycle.SCOPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown scope '{scope}'. Expected one of: "
                   f"{', '.join(data_lifecycle.SCOPES)}")
    return {"purged": data_lifecycle.purge_memory(account.id, scope)}
