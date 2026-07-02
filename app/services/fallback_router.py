import logging
from dataclasses import dataclass
from typing import AsyncGenerator
from sqlalchemy.orm import Session

from ..models.db_models import ChatModel, ApiKey
from ..models.schemas import MessageDto
from ..providers.registry import provider_registry
from . import rate_limit

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    content: str
    model_id: str
    platform: str
    display_name: str
    attempts: int


@dataclass
class StreamRouteResult:
    stream: AsyncGenerator[str, None]
    model_id: str
    platform: str
    display_name: str
    attempts: int


# ─── Complexity-aware auto routing ──────────────────────────────────────────
# Maps the size tier to a numeric "power" level so we can prefer a tier and
# fall back to the nearest ones.
_TIER_LEVEL = {"Small": 0, "Medium": 1, "Large": 2, "Frontier": 3}

# Signals in the user's message that call for a bigger, smarter model.
_HEAVY_SIGNALS = (
    "architecture", "end-to-end", "end to end", "system design", "design a",
    "design an", "scalable", "microservice", "distributed", "infrastructure",
    "prove", "proof", "derive", "optimi", "refactor", "algorithm",
    "trade-off", "tradeoff", "in-depth", "in depth", "comprehensive",
    "step by step", "step-by-step", "production", "entire project",
    "whole project", "full implementation", "detailed", "diagram", "roadmap",
)
# Signals for a moderate task (explanations, code, analysis).
_MODERATE_SIGNALS = (
    "explain", "why", "how does", "how do", "compare", "difference between",
    "analyze", "analyse", "summarize", "summarise", "write", "code",
    "function", "debug", "error", "fix", "example", "implement",
    "pros and cons", "best practice", "review",
)
# Tiny conversational turns that only need a small, fast model.
_SIMPLE_GREETINGS = {
    "hi", "hii", "hey", "hello", "yo", "sup", "thanks", "thank you", "thx",
    "ok", "okay", "cool", "nice", "great", "bye", "good morning",
    "good evening", "good night", "how are you", "hru", "wsg", "whats up",
    "what's up", "lol", "haha",
}
# Non-chat models we don't want auto-routing to answer text with.
_NON_CHAT_HINTS = (
    "image", "embed", "embedding", "rerank", "moderation", "tts", "whisper",
    "audio", "speech", "stable-diffusion", "sdxl", "flux", "dall-e", "video",
)


def _estimate_tier(messages: list[MessageDto]) -> str:
    """Heuristically map the latest user request to a model size tier."""
    user_text = ""
    for m in reversed(messages):
        if m.role == "user":
            user_text = (m.content or "").strip()
            break
    if not user_text:
        return "Medium"

    lower = user_text.lower()
    words = len(user_text.split())
    heavy = any(s in lower for s in _HEAVY_SIGNALS)
    moderate = any(s in lower for s in _MODERATE_SIGNALS)

    # Greeting or a couple of words with no task -> smallest, fastest model.
    if lower.strip("!?.,… ") in _SIMPLE_GREETINGS or (
        words <= 3 and not heavy and not moderate
    ):
        return "Small"
    if heavy or words >= 120:
        return "Frontier"
    if moderate or words >= 40:
        return "Large"
    if words >= 12:
        return "Medium"
    return "Small"


def _is_chatty(model: ChatModel) -> bool:
    """False for models that clearly aren't text chat (image/embed/audio...)."""
    text = f"{model.display_name} {model.model_id}".lower()
    return not any(h in text for h in _NON_CHAT_HINTS)


def _get_ordered_models(
    db: Session,
    requested_model: str | None,
    messages: list[MessageDto] | None = None,
) -> list[ChatModel]:
    """Get models to try, in order. A specific requested model goes first;
    otherwise ("auto") pick the tier that matches the request's complexity and
    fall back to the nearest tiers. Only models from platforms with active keys."""
    # Get platforms with active keys
    active_platforms = set(
        row[0] for row in
        db.query(ApiKey.platform).filter(ApiKey.enabled == True, ApiKey.status != "error").distinct().all()
    )

    models = (
        db.query(ChatModel)
        .filter(ChatModel.enabled == True, ChatModel.platform.in_(active_platforms))
        .order_by(ChatModel.priority.asc())
        .all()
    )

    if requested_model and requested_model.lower() != "auto":
        ordered = []
        rest = []
        for m in models:
            if m.model_id == requested_model:
                ordered.append(m)
            else:
                rest.append(m)
        return ordered + rest

    # AUTO: choose a tier from the request complexity, prefer real chat models,
    # then order by distance from the desired tier (nearest tiers are fallbacks).
    desired = _estimate_tier(messages) if messages else "Large"
    target = _TIER_LEVEL.get(desired, 2)
    models = sorted(
        models,
        key=lambda m: (
            0 if _is_chatty(m) else 1,
            abs(_TIER_LEVEL.get(m.size_label, 1) - target),
            m.priority,
        ),
    )
    if models:
        logger.info(
            f"[Router] Auto complexity → tier '{desired}'; "
            f"top pick: {models[0].display_name} ({models[0].size_label})"
        )
    return models


async def route_chat(
    db: Session,
    messages: list[MessageDto],
    requested_model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 10,
) -> RouteResult:
    """Route a chat request with automatic fallback across models and keys."""
    models = _get_ordered_models(db, requested_model, messages)
    skip_keys: set[str] = set()
    skip_models: set[int] = set()
    last_error: Exception | None = None

    for attempt in range(max_retries):
        for model in models:
            if model.id in skip_models:
                continue

            provider = provider_registry.get(model.platform)
            if provider is None:
                continue

            keys = (
                db.query(ApiKey)
                .filter(
                    ApiKey.platform == model.platform,
                    ApiKey.enabled == True,
                    ApiKey.status != "error",
                )
                .all()
            )
            if not keys:
                continue

            for key in keys:
                skip_id = f"{model.platform}:{model.model_id}:{key.id}"
                if skip_id in skip_keys:
                    continue
                if rate_limit.is_on_cooldown(model.platform, model.model_id, key.id):
                    continue

                try:
                    logger.info(
                        f"[Router] Attempt {attempt + 1}: {model.display_name} "
                        f"via {model.platform} (key #{key.id})"
                    )

                    content = await provider.chat_completion(
                        api_key=key.api_key,
                        messages=messages,
                        model_id=model.model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    if not content or not content.strip():
                        raise RuntimeError(f"Empty response from {model.display_name}")

                    rate_limit.record_success(model.platform, model.model_id, key.id)
                    logger.info(f"[Router] Success: {model.display_name} via {model.platform}")

                    return RouteResult(
                        content=content,
                        model_id=model.model_id,
                        platform=model.platform,
                        display_name=model.display_name,
                        attempts=attempt,
                    )

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"[Router] Error from {model.display_name} ({model.platform}): {e}"
                    )
                    skip_keys.add(skip_id)

                    if rate_limit.is_retryable_error(e):
                        rate_limit.set_cooldown(model.platform, model.model_id, key.id)
                        rate_limit.record_rate_limit_hit(model.platform, model.model_id, key.id)

                        msg = str(e).lower()
                        if "404" in msg or "not found" in msg or "403" in msg:
                            skip_models.add(model.id)
                            break
                    else:
                        skip_models.add(model.id)
                        break

    raise RuntimeError(
        f"All models exhausted after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


async def route_stream_chat(
    db: Session,
    messages: list[MessageDto],
    requested_model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 10,
) -> StreamRouteResult:
    """Route a streaming chat request with fallback."""
    models = _get_ordered_models(db, requested_model, messages)
    skip_keys: set[str] = set()
    skip_models: set[int] = set()
    last_error: Exception | None = None

    for attempt in range(max_retries):
        for model in models:
            if model.id in skip_models:
                continue

            provider = provider_registry.get(model.platform)
            if provider is None:
                continue

            keys = (
                db.query(ApiKey)
                .filter(
                    ApiKey.platform == model.platform,
                    ApiKey.enabled == True,
                    ApiKey.status != "error",
                )
                .all()
            )
            if not keys:
                continue

            for key in keys:
                skip_id = f"{model.platform}:{model.model_id}:{key.id}"
                if skip_id in skip_keys:
                    continue
                if rate_limit.is_on_cooldown(model.platform, model.model_id, key.id):
                    continue

                try:
                    logger.info(
                        f"[Router/Stream] Attempt {attempt + 1}: {model.display_name} "
                        f"via {model.platform} (key #{key.id})"
                    )

                    stream = provider.stream_chat_completion(
                        api_key=key.api_key,
                        messages=messages,
                        model_id=model.model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    rate_limit.record_success(model.platform, model.model_id, key.id)

                    return StreamRouteResult(
                        stream=stream,
                        model_id=model.model_id,
                        platform=model.platform,
                        display_name=model.display_name,
                        attempts=attempt,
                    )

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"[Router/Stream] Error from {model.display_name} ({model.platform}): {e}"
                    )
                    skip_keys.add(skip_id)
                    if rate_limit.is_retryable_error(e):
                        rate_limit.set_cooldown(model.platform, model.model_id, key.id)
                        rate_limit.record_rate_limit_hit(model.platform, model.model_id, key.id)
                        msg = str(e).lower()
                        if "404" in msg or "not found" in msg or "403" in msg:
                            skip_models.add(model.id)
                            break
                    else:
                        skip_models.add(model.id)
                        break

    raise RuntimeError(f"All models exhausted for streaming. Last error: {last_error}")
