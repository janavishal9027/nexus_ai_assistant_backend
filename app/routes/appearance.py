"""Appearance / personalization sync (Settings → Personalization).

The client is local-first: it reads SharedPreferences before the first frame so
the theme never flashes, and it only talks to this endpoint to keep a second
device in step. That means every failure here must be silent to the user — a
sync hiccup should never change how the app looks.

Last write wins. Two devices edited at once is not worth a merge protocol for
a text-size slider.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import Account, AppearancePrefs
from ..services.auth import get_current_account

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings", tags=["settings"])

DEFAULTS = {
    "theme_mode": "dark",
    "accent": "#10A37F",
    "text_size": 15,
    "corner_radius": 16,
    "density": "comfortable",
    "reduce_animations": False,
    "wallpaper": "none",
}

_THEME_MODES = {"system", "light", "dark"}
_DENSITIES = {"comfortable", "compact"}


class AppearanceRequest(BaseModel):
    """Partial update — only the fields present are changed.

    Bounds are enforced here as well as in the UI: the sliders can't send a
    silly value, but a stale client or a hand-rolled request could, and a
    text_size of 900 would render the app unusable on every other device.
    """
    theme_mode: Optional[str] = None
    accent: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    text_size: Optional[int] = Field(None, ge=11, le=24)
    corner_radius: Optional[int] = Field(None, ge=0, le=28)
    density: Optional[str] = None
    reduce_animations: Optional[bool] = None
    wallpaper: Optional[str] = Field(None, max_length=32)


def _as_dict(row: Optional[AppearancePrefs]) -> dict:
    if row is None:
        return dict(DEFAULTS)
    return {
        "theme_mode": row.theme_mode or DEFAULTS["theme_mode"],
        "accent": row.accent or DEFAULTS["accent"],
        "text_size": row.text_size if row.text_size is not None else DEFAULTS["text_size"],
        "corner_radius": (row.corner_radius if row.corner_radius is not None
                          else DEFAULTS["corner_radius"]),
        "density": row.density or DEFAULTS["density"],
        "reduce_animations": bool(row.reduce_animations),
        "wallpaper": row.wallpaper or DEFAULTS["wallpaper"],
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


@router.get("/appearance")
def get_appearance(db: Session = Depends(get_db),
                   account: Account = Depends(get_current_account)):
    """This account's look-and-feel. Defaults when never set."""
    row = (db.query(AppearancePrefs)
           .filter(AppearancePrefs.owner_id == account.id).first())
    return _as_dict(row)


@router.patch("/appearance")
def set_appearance(body: AppearanceRequest, db: Session = Depends(get_db),
                   account: Account = Depends(get_current_account)):
    """Update the fields supplied; leave the rest alone."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No settings supplied")
    if "theme_mode" in updates and updates["theme_mode"] not in _THEME_MODES:
        raise HTTPException(status_code=422,
                            detail=f"theme_mode must be one of {sorted(_THEME_MODES)}")
    if "density" in updates and updates["density"] not in _DENSITIES:
        raise HTTPException(status_code=422,
                            detail=f"density must be one of {sorted(_DENSITIES)}")
    try:
        row = (db.query(AppearancePrefs)
               .filter(AppearancePrefs.owner_id == account.id).first())
        if row is None:
            row = AppearancePrefs(owner_id=account.id)
            db.add(row)
        for k, v in updates.items():
            setattr(row, k, v)
        db.commit()
        db.refresh(row)
    except Exception as e:
        db.rollback()
        logger.warning(f"[Appearance] save failed for owner={account.id}: {e}")
        raise HTTPException(status_code=500, detail="Could not save appearance")
    return _as_dict(row)
