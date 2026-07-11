from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from ..database import get_db
from ..models.db_models import ApiKey, Account
from ..models.schemas import AddKeyRequest
from ..providers.registry import provider_registry
from ..services.auth import get_current_account
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/keys", tags=["keys"])

# Non-LLM API keys accepted here (e.g. web-search providers). These aren't in the
# LLM provider_registry but are valid keys the user configures in the UI.
SEARCH_PLATFORMS = {"tavily"}


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "****"
    return key[:4] + "..." + key[-4:]


@router.get("/")
def get_all(db: Session = Depends(get_db), account: Account = Depends(get_current_account)):
    # Only the account's own keys are shown/managed; shared/global (owner_id NULL)
    # keys remain usable for chat but aren't editable per-user.
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.owner_id == account.id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
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
async def add_key(request: AddKeyRequest, db: Session = Depends(get_db),
                  account: Account = Depends(get_current_account)):
    if not provider_registry.has(request.platform) and request.platform not in SEARCH_PLATFORMS:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {request.platform}")

    if not request.key.strip():
        raise HTTPException(status_code=400, detail="API key is required")

    existing = (
        db.query(ApiKey)
        .filter(
            ApiKey.platform == request.platform,
            ApiKey.api_key == request.key,
            ApiKey.owner_id == account.id,
        )
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
        owner_id=account.id,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    # Auto-sync this provider's model catalog so the models appear immediately
    # after adding the key. Only LLM providers have models; search keys (Tavily)
    # do not, so skip the sync for those.
    models_synced = 0
    sync_error = None
    if provider_registry.has(request.platform):
        try:
            from ..services.model_sync import sync_provider_models
            result = await sync_provider_models(db, request.platform)
            if isinstance(result, dict):
                sync_error = result.get("error")
                # Prefer the provider's total model count (informative even when
                # the models were already present from a previous sync).
                models_synced = result.get("total", result.get("added", 0))
            logger.info(
                f"[Keys] Added {request.platform} key for account {account.id}; "
                f"models_synced={models_synced}"
                + (f", sync_error={sync_error}" if sync_error else "")
            )
        except Exception as e:
            # Key is still saved even if the model fetch fails — but surface it
            # instead of failing silently.
            sync_error = str(e)
            logger.warning(
                f"[Keys] Model sync failed for {request.platform}: {e}")

    return {
        "id": api_key.id,
        "platform": api_key.platform,
        "maskedKey": _mask_key(api_key.api_key),
        "status": "unknown",
        "models_synced": models_synced,
        "sync_error": sync_error,
    }


@router.delete("/{key_id}")
def delete_key(key_id: int, db: Session = Depends(get_db),
               account: Account = Depends(get_current_account)):
    key = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.owner_id == account.id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    db.delete(key)
    db.commit()
    return {"success": True}


class ToggleKey(BaseModel):
    enabled: Optional[bool] = None


@router.patch("/{key_id}")
def toggle_key(key_id: int, body: ToggleKey, db: Session = Depends(get_db),
               account: Account = Depends(get_current_account)):
    key = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.owner_id == account.id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    if body.enabled is not None:
        key.enabled = body.enabled
        db.commit()
    return {"success": True, "enabled": key.enabled}


@router.get("/platforms")
def get_platforms():
    return provider_registry.all_platforms()
