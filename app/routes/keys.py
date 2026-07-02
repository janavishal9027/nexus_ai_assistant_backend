from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from ..database import get_db
from ..models.db_models import ApiKey
from ..models.schemas import AddKeyRequest
from ..providers.registry import provider_registry

router = APIRouter(prefix="/api/keys", tags=["keys"])


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "****"
    return key[:4] + "..." + key[-4:]


@router.get("/")
def get_all(db: Session = Depends(get_db)):
    keys = db.query(ApiKey).order_by(ApiKey.created_at.desc()).all()
    return [
        {
            "id": k.id,
            "platform": k.platform,
            "label": k.label or "",
            "maskedKey": _mask_key(k.api_key),
            "enabled": k.enabled,
            "status": k.status,
            "createdAt": k.created_at.isoformat() if k.created_at else "",
        }
        for k in keys
    ]


@router.post("/")
def add_key(request: AddKeyRequest, db: Session = Depends(get_db)):
    if not provider_registry.has(request.platform):
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {request.platform}")

    if not request.key.strip():
        raise HTTPException(status_code=400, detail="API key is required")

    existing = (
        db.query(ApiKey)
        .filter(ApiKey.platform == request.platform, ApiKey.api_key == request.key)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Key already exists for this platform")

    api_key = ApiKey(
        platform=request.platform,
        api_key=request.key.strip(),
        label=request.label or "",
        enabled=True,
        status="unknown",
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return {
        "id": api_key.id,
        "platform": api_key.platform,
        "maskedKey": _mask_key(api_key.api_key),
        "status": "unknown",
    }


@router.delete("/{key_id}")
def delete_key(key_id: int, db: Session = Depends(get_db)):
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    db.delete(key)
    db.commit()
    return {"success": True}


class ToggleKey(BaseModel):
    enabled: Optional[bool] = None


@router.patch("/{key_id}")
def toggle_key(key_id: int, body: ToggleKey, db: Session = Depends(get_db)):
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    if body.enabled is not None:
        key.enabled = body.enabled
        db.commit()
    return {"success": True, "enabled": key.enabled}


@router.get("/platforms")
def get_platforms():
    return provider_registry.all_platforms()
