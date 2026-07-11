"""
Config endpoint — serves the unified providers config to the frontend.
The frontend uses this to dynamically render providers, models, and status.
"""
from fastapi import APIRouter, Depends
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import ApiKey, ChatModel, Account
from ..services.agent import get_config, reload_config
from ..services.auth import get_current_account
from ..services.model_sync import sync_all_providers, sync_provider_models
from ..services.fallback_router import _is_chatty, _param_billions

router = APIRouter(prefix="/api/config", tags=["config"])

# Providers whose synced models are free to use (free tier / no per-token cost).
_FREE_PLATFORMS = {"groq", "google", "nvidia", "huggingface", "cerebras", "sambanova"}


def _cost_label(platform: str, model_id: str) -> str:
    """Best-effort billing hint for a model: 'free', 'credit' or 'paid'. This is
    a heuristic (real pricing varies per model) meant as an at-a-glance
    indicator, not a billing guarantee."""
    mid = (model_id or "").lower()
    if ":free" in mid or "(free)" in mid:
        return "free"
    if platform in _FREE_PLATFORMS:
        return "free"
    if platform in ("openrouter", "vercel"):
        return "credit"  # prepaid-credit gateways
    return "paid"  # mistral, zai and other pay-per-use APIs


@router.get("/")
def get_app_config(db: Session = Depends(get_db), account: Account = Depends(get_current_account)):
    """
    Returns the full app config enriched with key status and DB models.
    The frontend uses this single endpoint to know:
    - Which providers exist
    - Which providers have active keys (and are therefore usable)
    - Which models are available per provider (from DB, not just static config)
    - Agent settings
    """
    config = get_config()

    # Count active keys per platform visible to this account (own + shared/global)
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.enabled == True,
                or_(ApiKey.owner_id == account.id, ApiKey.owner_id.is_(None)))
        .all()
    )
    key_counts: dict[str, int] = {}
    for k in keys:
        key_counts[k.platform] = key_counts.get(k.platform, 0) + 1

    # Get all models from DB grouped by platform
    all_models = db.query(ChatModel).filter(ChatModel.enabled == True).order_by(ChatModel.priority.asc()).all()
    models_by_platform: dict[str, list] = {}
    for m in all_models:
        # Hide non-chat models (safety guards, embeddings, whisper/TTS, rerankers)
        # so they can't be picked as a chat model.
        if not _is_chatty(m):
            continue
        if m.platform not in models_by_platform:
            models_by_platform[m.platform] = []
        models_by_platform[m.platform].append({
            "id": m.model_id,
            "name": m.display_name,
            "tier": m.size_label,
            "context": m.context_window,
            "vision": m.supports_vision,
            "tools": m.supports_tools,
            "param_billions": _param_billions(m),
            "cost": _cost_label(m.platform, m.model_id),
        })
    # Order each provider's models smallest → largest by parameter count
    # (models whose size is unknown go last).
    for _ms in models_by_platform.values():
        _ms.sort(key=lambda x: (x["param_billions"] is None, x["param_billions"] or 0.0))

    # Enrich providers with active status and DB models
    providers = []
    for p in config.get("providers", []):
        platform_id = p["id"]
        provider_data = {
            **p,
            "active": key_counts.get(platform_id, 0) > 0,
            "key_count": key_counts.get(platform_id, 0),
            # Use DB models (includes synced ones) instead of static config
            "models": models_by_platform.get(platform_id, p.get("models", [])),
        }
        providers.append(provider_data)

    return {
        "providers": providers,
        "agent": config.get("agent", {}),
        "fallback": config.get("fallback", {}),
    }


@router.post("/reload")
def reload():
    """Reload the config from disk (after editing providers_config.json)."""
    new_config = reload_config()
    return {"success": True, "providers": len(new_config.get("providers", []))}


@router.post("/sync-models")
async def sync_models(db: Session = Depends(get_db)):
    """Fetch all models from all providers that have API keys and sync to database."""
    result = await sync_all_providers(db)
    return result


@router.post("/sync-models/{platform}")
async def sync_platform_models(platform: str, db: Session = Depends(get_db)):
    """Fetch models for a specific provider and sync to database."""
    result = await sync_provider_models(db, platform)
    return result
