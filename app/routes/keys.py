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
            # Real health now that the router writes it (see provider_health):
            # healthy | error | limited | unknown, plus why and when.
            "status": k.status or "unknown",
            "lastError": k.last_error,
            "lastCheckedAt": k.last_checked_at.isoformat() if k.last_checked_at else None,
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


@router.post("/{key_id}/test")
async def test_key(key_id: int, db: Session = Depends(get_db),
                   account: Account = Depends(get_current_account)):
    """Check a key against its provider right now and record the result.

    Without this the only way to learn a key is dead is to send a chat and watch
    it silently fall back — and the only way to clear an `error` is to wait out
    the cooldown.

    max_tokens is deliberately generous: reasoning models (gpt-oss-120b on
    cerebras/groq, magistral on mistral) spend their budget thinking and return
    EMPTY content on a tight cap, which reads as a failure and would report a
    perfectly good key as broken.
    """
    from ..models.db_models import ChatModel
    from ..services import provider_health
    from ..services.fallback_router import _is_chatty
    from ..models.schemas import MessageDto

    key = db.query(ApiKey).filter(ApiKey.id == key_id,
                                  ApiKey.owner_id == account.id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")

    provider = provider_registry.get(key.platform)
    if provider is None:
        # Non-LLM keys (e.g. Tavily) have no chat endpoint to probe.
        raise HTTPException(status_code=400,
                            detail=f"'{key.platform}' keys can't be tested here")

    model = next((m for m in db.query(ChatModel)
                  .filter(ChatModel.platform == key.platform,
                          ChatModel.enabled == True)
                  .order_by(ChatModel.priority.asc()).all() if _is_chatty(m)), None)
    if model is None:
        raise HTTPException(status_code=400,
                            detail=f"No chat model available for {key.platform}")

    try:
        await provider.chat_completion(
            api_key=key.api_key,
            messages=[MessageDto(role="user", content="Say hello")],
            model_id=model.model_id, temperature=0.0, max_tokens=200)
    except Exception as e:
        status = provider_health.record_failure(db, key.id, e)
        if status is None:
            # The failure says nothing about the key (bad model id, provider 5xx).
            return {"ok": False, "status": key.status or "unknown",
                    "error": f"Could not verify the key: {e}"[:300]}
        db.refresh(key)
        return {"ok": False, "status": key.status, "error": key.last_error}

    provider_health.record_success(db, key.id)
    db.refresh(key)
    return {"ok": True, "status": key.status, "error": None}


@router.get("/platforms")
def get_platforms():
    return provider_registry.all_platforms()
