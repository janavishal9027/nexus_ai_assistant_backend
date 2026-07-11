"""
Dynamically fetch ALL models from provider APIs using the user's API keys.
No hardcoded model lists — everything is fetched live from the provider.
"""
import httpx
import logging
from sqlalchemy.orm import Session

from ..models.db_models import ChatModel, ApiKey
from ..providers.registry import provider_registry

logger = logging.getLogger(__name__)

# Provider model list endpoints
PROVIDER_MODEL_ENDPOINTS = {
    "openrouter": "https://openrouter.ai/api/v1/models",
    "groq": "https://api.groq.com/openai/v1/models",
    "nvidia": "https://integrate.api.nvidia.com/v1/models",
    "huggingface": "https://router.huggingface.co/v1/models",
    "mistral": "https://api.mistral.ai/v1/models",
    "cerebras": "https://api.cerebras.ai/v1/models",
    "sambanova": "https://api.sambanova.ai/v1/models",
    "vercel": "https://ai-gateway.vercel.sh/v1/models",
    "zai": "https://api.z.ai/api/paas/v4/models",
}


async def sync_all_providers(db: Session) -> dict:
    """Sync models from ALL providers that have active API keys."""
    results = {}

    # Get all platforms with active keys
    active_keys = db.query(ApiKey).filter(ApiKey.enabled == True).all()
    keys_by_platform: dict[str, str] = {}
    for k in active_keys:
        if k.platform not in keys_by_platform:
            keys_by_platform[k.platform] = k.api_key

    for platform, api_key in keys_by_platform.items():
        if platform == "google":
            result = await _sync_google_models(db, api_key)
        elif platform in PROVIDER_MODEL_ENDPOINTS:
            result = await _sync_openai_compat_models(db, platform, api_key)
        else:
            result = {"skipped": True}
        results[platform] = result

    return results


async def sync_provider_models(db: Session, platform: str) -> dict:
    """Sync models for a specific provider."""
    key = (
        db.query(ApiKey)
        .filter(ApiKey.platform == platform, ApiKey.enabled == True)
        .first()
    )
    if not key:
        return {"error": f"No API key for {platform}", "synced": 0}

    if platform == "google":
        return await _sync_google_models(db, key.api_key)
    elif platform in PROVIDER_MODEL_ENDPOINTS:
        return await _sync_openai_compat_models(db, platform, key.api_key)
    else:
        return {"error": f"Sync not supported for {platform}", "synced": 0}


async def _sync_openai_compat_models(db: Session, platform: str, api_key: str) -> dict:
    """Fetch models from an OpenAI-compatible /models endpoint."""
    url = PROVIDER_MODEL_ENDPOINTS.get(platform)
    if not url:
        return {"error": "No endpoint configured", "synced": 0}

    provider = provider_registry.get(platform)
    extra_headers = {}
    if hasattr(provider, '_extra_headers'):
        extra_headers = provider._extra_headers

    headers = {
        "Authorization": f"Bearer {api_key}",
        **extra_headers,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)

            if resp.status_code != 200:
                return {"error": f"API error {resp.status_code}: {resp.text[:200]}", "synced": 0}

            data = resp.json()
            models = data.get("data", [])

            # Include ALL models accessible with the API key (not just free)
            added = 0
            updated = 0
            for m in models:
                model_id = m.get("id", "")
                if not model_id:
                    continue

                name = m.get("name", model_id.split("/")[-1])
                context_length = m.get("context_length", 4096)

                existing = db.query(ChatModel).filter(
                    ChatModel.platform == platform,
                    ChatModel.model_id == model_id,
                ).first()

                if existing:
                    # Update context window and name if changed
                    if existing.context_window != context_length or existing.display_name != name:
                        existing.context_window = context_length
                        existing.display_name = name
                        updated += 1
                else:
                    tier = _guess_tier(name, context_length, model_id)
                    db.add(ChatModel(
                        platform=platform,
                        model_id=model_id,
                        display_name=name,
                        size_label=tier,
                        intelligence_rank=_rank_for_tier(tier),
                        speed_rank=5,
                        context_window=context_length,
                        supports_vision=_is_vision_model(name, model_id, m),
                        supports_tools=True,
                        enabled=True,
                        priority=_rank_for_tier(tier),
                    ))
                    added += 1

            if added > 0 or updated > 0:
                db.commit()

            total = db.query(ChatModel).filter(ChatModel.platform == platform).count()
            logger.info(f"[ModelSync] {platform}: {len(models)} models fetched, {added} added, {updated} updated, {total} total")

            return {
                "fetched": len(models),
                "added": added,
                "updated": updated,
                "total": total,
            }

    except Exception as e:
        logger.error(f"[ModelSync] Error fetching {platform} models: {e}")
        return {"error": str(e), "synced": 0}


async def _sync_google_models(db: Session, api_key: str) -> dict:
    """Fetch models from Google AI Studio's model list API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)

            if resp.status_code != 200:
                return {"error": f"Google API error {resp.status_code}", "synced": 0}

            data = resp.json()
            models = data.get("models", [])

            added = 0
            for m in models:
                # Google model name format: "models/gemini-2.0-flash"
                full_name = m.get("name", "")
                model_id = full_name.replace("models/", "")
                if not model_id:
                    continue

                # Only include generative models
                supported_methods = m.get("supportedGenerationMethods", [])
                if "generateContent" not in supported_methods:
                    continue

                display_name = m.get("displayName", model_id)
                context_length = m.get("inputTokenLimit", 32000)

                existing = db.query(ChatModel).filter(
                    ChatModel.platform == "google",
                    ChatModel.model_id == model_id,
                ).first()

                if not existing:
                    tier = _guess_tier(display_name, context_length, model_id)
                    db.add(ChatModel(
                        platform="google",
                        model_id=model_id,
                        display_name=display_name,
                        size_label=tier,
                        intelligence_rank=_rank_for_tier(tier),
                        speed_rank=3,
                        context_window=context_length,
                        supports_vision="vision" in model_id.lower() or "gemini" in model_id.lower(),
                        supports_tools=True,
                        enabled=True,
                        priority=_rank_for_tier(tier),
                    ))
                    added += 1

            if added > 0:
                db.commit()

            total = db.query(ChatModel).filter(ChatModel.platform == "google").count()
            logger.info(f"[ModelSync] google: {len(models)} models fetched, {added} added, {total} total")

            return {"fetched": len(models), "added": added, "total": total}

    except Exception as e:
        logger.error(f"[ModelSync] Error fetching Google models: {e}")
        return {"error": str(e), "synced": 0}


def _filter_free_openrouter_models(models: list) -> list:
    """Filter OpenRouter models to only include free ones (pricing == 0)."""
    free = []
    for m in models:
        pricing = m.get("pricing", {})
        prompt_price = str(pricing.get("prompt", "1"))
        completion_price = str(pricing.get("completion", "1"))

        if prompt_price == "0" and completion_price == "0":
            free.append(m)

    return free


def _is_vision_model(name: str, model_id: str, raw: dict) -> bool:
    """Detect if a model supports vision/images."""
    name_lower = (name + model_id).lower()
    if any(x in name_lower for x in ["vision", "vl", "visual", "image"]):
        return True
    # OpenRouter marks it in architecture
    arch = raw.get("architecture", {})
    if arch.get("modality", "") in ("multimodal", "text+image->text"):
        return True
    return False


def _guess_tier(name: str, context_length: int, model_id: str = "") -> str:
    """Guess the model tier from its name and context."""
    combined = (name + model_id).lower()
    if any(x in combined for x in ["405b", "671b", "235b", "120b", "ultra", "550b"]):
        return "Frontier"
    if any(x in combined for x in ["70b", "72b", "80b", "super", "pro"]):
        return "Large"
    if any(x in combined for x in ["27b", "24b", "32b", "31b", "medium"]):
        return "Medium"
    if any(x in combined for x in ["1b", "3b", "7b", "8b", "9b", "nano", "small", "mini", "lite"]):
        return "Small"
    if context_length >= 500000:
        return "Frontier"
    if context_length >= 100000:
        return "Large"
    return "Medium"


def _rank_for_tier(tier: str) -> int:
    return {"Frontier": 2, "Large": 5, "Medium": 8, "Small": 12}.get(tier, 10)
