import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import json
from typing import AsyncGenerator
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..models.db_models import ChatModel, ApiKey
from ..models.schemas import MessageDto
from ..providers.registry import provider_registry
from . import rate_limit
from . import provider_health
from . import request_context

logger = logging.getLogger(__name__)

_config_path = Path(__file__).parent.parent / "providers_config.json"

# Hard wall-clock cap on how long the router spends trying models before giving
# up. Bounds worst-case latency when many models fail (bad key, quota, non-chat
# models). The good case (a working model early) is unaffected.
OVERALL_BUDGET_S = 20.0

# After this many quota/rate-limit (429) errors from one provider in a single
# request, treat it as a project-level limit and stop trying that provider.
_PLATFORM_429_LIMIT = 3


def _get_max_retries() -> int:
    """Read max_retries from providers_config.json. Defaults to 3."""
    try:
        data = json.loads(_config_path.read_text(encoding="utf-8"))
        return int(data.get("fallback", {}).get("max_retries", 3))
    except Exception:
        return 3


def _apply_owner_scope(query):
    """Restrict ApiKey rows to the current account's own keys plus shared/global
    (owner_id NULL) keys. No-op when no authenticated owner is in context, so
    internal/system calls keep working."""
    owner_id = request_context.get_owner_id()
    if owner_id is None:
        return query
    return query.filter(or_(ApiKey.owner_id == owner_id, ApiKey.owner_id.is_(None)))


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
    # Safety / moderation / classifier / reward / OCR / retrieval models: these
    # emit verdicts, scores or vectors, NOT conversation, so a chat routed to
    # them comes back as e.g. {"User Safety": "safe"} instead of a reply.
    "guard", "safety", "shieldgemma", "classifier", "detector", "reward",
    "ocr", "retriever",
    # Groq's "compound" agentic systems return 200 with no streamed chat text
    # through our path (empty reply), so keep them out of the auto pool.
    "compound",
    # Google (and others) non-text families that error out on chat completions.
    "imagen", "veo", "lyria", "music", "gemini-embedding", "text-embedding", "aqa",
    # Mistral's Voxtral family is audio (ASR/TTS/realtime). Only the "-tts-"
    # variants matched a hint above, so plain `voxtral-mini-2602` looked chatty
    # and every auto route burned a round-trip on it for a 400 Invalid model —
    # eating the fallback budget before reaching providers that work.
    "voxtral", "asr", "transcribe", "diariz", "realtime",
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


# ─── Deep Research: large-model (>=400B parameter) gating ────────────────────
# Minimum parameter count (in billions) a model must have to be used in Deep
# Research mode. Anything below this is excluded from that mode entirely.
_MIN_DEEP_RESEARCH_B = 400.0

# Parameter counts (billions) for big models whose size isn't in the name
# (closed or oddly-named). Only consulted when no explicit "<n>b" token is
# present, so distilled variants like "deepseek-r1-distill-llama-70b" are still
# read as 70B (from the explicit token) rather than the base model's size.
_KNOWN_PARAMS_B = {
    "deepseek-v3": 671, "deepseek-r1": 671, "deepseek-chat": 671,
    "deepseek-reasoner": 671, "deepseek-v3.1": 671, "deepseek-v3.2": 671,
    "kimi-k2": 1000,
    "llama-4-maverick": 400, "llama-4-behemoth": 2000,
}

import re as _re
_PARAM_RE = _re.compile(r"(\d+(?:\.\d+)?)\s*b\b")


def _param_billions(model: ChatModel) -> float | None:
    """Best-effort parameter count (in billions) for a model, or None if the
    size can't be determined. Prefers an explicit '<n>b' token in the id/name;
    falls back to a small known-map for closed/oddly-named big models."""
    text = f"{model.model_id} {model.display_name or ''}".lower()
    best: float | None = None
    for m in _PARAM_RE.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if best is None or v > best:
            best = v
    if best is not None:
        return best
    for key, b in _KNOWN_PARAMS_B.items():
        if key in text:
            return float(b)
    return None


def _is_deep_research_model(model: ChatModel) -> bool:
    """True only for large (>=400B parameter) chat models — the tier Deep
    Research mode is allowed to use."""
    if not _is_chatty(model):
        return False
    b = _param_billions(model)
    return b is not None and b >= _MIN_DEEP_RESEARCH_B


class DeepResearchUnavailableError(RuntimeError):
    """Raised when Deep Research mode is requested but no >=400B model is
    available with the current keys. Carries a user-facing, actionable message."""


def _get_ordered_models(
    db: Session,
    requested_model: str | None,
    messages: list[MessageDto] | None = None,
    deep_research: bool = False,
    require_vision: bool = False,
) -> list[ChatModel]:
    """Get models to try, in order. A specific requested model goes first;
    otherwise ("auto") pick the tier that matches the request's complexity and
    fall back to the nearest tiers. Only models from platforms with active keys.

    When deep_research is True, restrict the pool to large (>=400B parameter)
    models and order them biggest-first, ignoring the requested model/tier."""
    # Get platforms with active keys
    active_platforms = set(
        row[0] for row in
        _apply_owner_scope(
            db.query(ApiKey.platform).filter(ApiKey.enabled == True,
                                             provider_health.usable_filter())
        ).distinct().all()
    )

    models = (
        db.query(ChatModel)
        .filter(ChatModel.enabled == True, ChatModel.platform.in_(active_platforms))
        .order_by(ChatModel.priority.asc())
        .all()
    )

    if require_vision:
        # Images are attached → only vision-capable chat models can handle them.
        # Honor a specific requested model only if it too supports vision.
        vision = [m for m in models if m.supports_vision and _is_chatty(m)]
        if requested_model and requested_model.lower() not in ("", "auto") \
                and not requested_model.lower().startswith("auto:"):
            first = [m for m in vision if m.model_id == requested_model]
            rest = [m for m in vision if m.model_id != requested_model]
            if first:
                return first + rest
        desired = _estimate_tier(messages) if messages else "Large"
        target = _TIER_LEVEL.get(desired, 2)
        vision.sort(key=lambda m: (abs(_TIER_LEVEL.get(m.size_label, 1) - target), m.priority))
        if vision:
            logger.info(
                f"[Router] Vision required → {len(vision)} vision model(s); "
                f"top pick: {vision[0].display_name}")
        else:
            logger.warning(
                "[Router] Vision required but no vision-capable model is available "
                "with the current keys")
        return vision

    if deep_research:
        # Only >=400B models, ordered by parameter count (largest first) then
        # priority. Requested model / complexity tier are intentionally ignored.
        big = [m for m in models if _is_deep_research_model(m)]
        big.sort(key=lambda m: (-(_param_billions(m) or 0), m.priority))
        if big:
            logger.info(
                f"[Router] Deep Research → {len(big)} model(s) >= "
                f"{_MIN_DEEP_RESEARCH_B:.0f}B; top pick: {big[0].display_name} "
                f"(~{_param_billions(big[0]):.0f}B)"
            )
        else:
            logger.warning("[Router] Deep Research requested but no >=400B model is available")
        return big

    # Provider-scoped auto ("auto:<platform>"): let the app auto-pick within a
    # single provider (e.g. the user chose "Groq · Auto"). Order by complexity
    # tier like global auto, but only among that platform's chat models.
    if requested_model and requested_model.lower().startswith("auto:"):
        plat = requested_model.split(":", 1)[1].strip().lower()
        scoped = [m for m in models if m.platform == plat and _is_chatty(m)]
        desired = _estimate_tier(messages) if messages else "Large"
        target = _TIER_LEVEL.get(desired, 2)
        scoped.sort(
            key=lambda m: (abs(_TIER_LEVEL.get(m.size_label, 1) - target), m.priority))
        if scoped:
            logger.info(
                f"[Router] Provider-auto '{plat}' → {len(scoped)} model(s); "
                f"top pick: {scoped[0].display_name}")
        else:
            logger.warning(f"[Router] Provider-auto '{plat}' but no chat model available")
        return scoped

    if requested_model and requested_model.lower() != "auto":
        ordered = []
        rest = []
        for m in models:
            if m.model_id == requested_model:
                ordered.append(m)
            elif _is_chatty(m):  # non-chat models are dead-ends for chat fallback
                rest.append(m)
        return ordered + rest

    # AUTO: exclude non-chat models entirely (trying an image/audio/embedding
    # model for chat wastes a full round-trip each), then order by tier proximity.
    models = [m for m in models if _is_chatty(m)]
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
    max_retries: int | None = None,
    deep_research: bool = False,
) -> RouteResult:
    """Route a chat request with automatic fallback across models and keys."""
    if max_retries is None:
        max_retries = _get_max_retries()
    models = _get_ordered_models(db, requested_model, messages, deep_research=deep_research)
    if deep_research and not models:
        raise DeepResearchUnavailableError(
            "Deep Research needs a large model (400B+ parameters), but none are "
            "available with your current keys. Add a provider that offers one — "
            "e.g. NVIDIA NIM (Llama-3.1-405B or DeepSeek-R1) — then try again."
        )
    skip_keys: set[str] = set()
    skip_models: set[int] = set()
    skip_platforms: set[str] = set()   # providers whose key failed auth or hit quota
    platform_429: dict[str, int] = {}  # per-provider 429 count this request
    last_error: Exception | None = None
    deadline = time.monotonic() + OVERALL_BUDGET_S

    for attempt in range(max_retries):
        for model in models:
            if time.monotonic() > deadline:
                logger.warning(f"[Router] Overall {OVERALL_BUDGET_S:.0f}s budget exceeded; stopping fallback")
                break  # wall-clock budget exhausted; stop trying more models
            if model.id in skip_models or model.platform in skip_platforms:
                continue

            provider = provider_registry.get(model.platform)
            if provider is None:
                continue

            keys = (
                _apply_owner_scope(
                    db.query(ApiKey).filter(
                        ApiKey.platform == model.platform,
                        ApiKey.enabled == True,
                        provider_health.usable_filter(),
                    )
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
                    provider_health.record_success(db, key.id)
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
                    msg = str(e).lower()
                    # Persist what this says about the key, so the next request
                    # skips a dead provider outright instead of re-discovering it
                    # (and so Settings can explain why).
                    health = provider_health.record_failure(db, key.id, e)
                    # Provider-level failure (bad, forbidden or uncredited key) →
                    # skip the WHOLE provider immediately, regardless of whether the
                    # error is "retryable" (e.g. Vercel's 403 "requires a valid
                    # credit card" is not, and OpenRouter's 402 "requires more
                    # credits" once fell through to a per-model retry that walked
                    # every model until the budget died).
                    if health == provider_health.ERROR:
                        skip_platforms.add(model.platform)
                        skip_models.add(model.id)
                        break

                    if rate_limit.is_retryable_error(e):
                        rate_limit.set_cooldown(model.platform, model.model_id, key.id)
                        rate_limit.record_rate_limit_hit(model.platform, model.model_id, key.id)
                        # Repeated quota/rate-limit from a provider → treat as a
                        # project-level limit and skip the rest of that provider.
                        if any(t in msg for t in (
                            "429", "quota", "rate limit", "rate-limit",
                            "resource_exhausted", "too many requests",
                        )):
                            platform_429[model.platform] = platform_429.get(model.platform, 0) + 1
                            # Abandon the whole provider on repeated 429s only if
                            # there's another provider to fall back to; if it's the
                            # only one, keep trying its models (some may have quota).
                            other_platforms = {m.platform for m in models} - skip_platforms - {model.platform}
                            if platform_429[model.platform] >= _PLATFORM_429_LIMIT and other_platforms:
                                skip_platforms.add(model.platform)
                            skip_models.add(model.id)
                            break
                        if "404" in msg or "not found" in msg:
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
    max_retries: int | None = None,
    deep_research: bool = False,
    require_vision: bool = False,
) -> StreamRouteResult:
    """Route a streaming chat request with fallback."""
    if max_retries is None:
        max_retries = _get_max_retries()
    models = _get_ordered_models(db, requested_model, messages,
                                 deep_research=deep_research, require_vision=require_vision)
    if deep_research and not models:
        raise DeepResearchUnavailableError(
            "Deep Research needs a large model (400B+ parameters), but none are "
            "available with your current keys. Add a provider that offers one — "
            "e.g. NVIDIA NIM (Llama-3.1-405B or DeepSeek-R1) — then try again."
        )
    skip_keys: set[str] = set()
    skip_models: set[int] = set()
    skip_platforms: set[str] = set()   # providers whose key failed auth or hit quota
    platform_429: dict[str, int] = {}  # per-provider 429 count this request
    last_error: Exception | None = None
    deadline = time.monotonic() + OVERALL_BUDGET_S

    for attempt in range(max_retries):
        for model in models:
            if time.monotonic() > deadline:
                logger.warning(f"[Router] Overall {OVERALL_BUDGET_S:.0f}s budget exceeded; stopping fallback")
                break  # wall-clock budget exhausted; stop trying more models
            if model.id in skip_models or model.platform in skip_platforms:
                continue

            provider = provider_registry.get(model.platform)
            if provider is None:
                continue

            keys = (
                _apply_owner_scope(
                    db.query(ApiKey).filter(
                        ApiKey.platform == model.platform,
                        ApiKey.enabled == True,
                        provider_health.usable_filter(),
                    )
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

                    # Prime the stream: pull the first chunk here so provider errors
                    # (e.g. a 403 auth failure) are raised *inside* this retry loop
                    # and fall back to the next model — instead of after the router
                    # has already returned, which would surface as a hard failure.
                    agen = stream.__aiter__()
                    first_chunk, has_first = None, False
                    try:
                        # Pull the first *non-empty* chunk. This surfaces provider
                        # errors inside the retry loop AND detects models that 200
                        # with no real content (e.g. Groq's "compound"), so we can
                        # fall back instead of returning an empty reply.
                        while True:
                            chunk = await agen.__anext__()
                            if chunk:
                                first_chunk, has_first = chunk, True
                                break
                    except StopAsyncIteration:
                        pass

                    if not has_first:
                        logger.warning(
                            f"[Router/Stream] {model.display_name} ({model.platform}) "
                            f"streamed no content; falling back to the next model"
                        )
                        skip_keys.add(skip_id)
                        continue

                    async def _primed_stream(_agen=agen, _first=first_chunk):
                        yield _first
                        async for chunk in _agen:
                            yield chunk

                    rate_limit.record_success(model.platform, model.model_id, key.id)
                    provider_health.record_success(db, key.id)

                    return StreamRouteResult(
                        stream=_primed_stream(),
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
                    msg = str(e).lower()
                    health = provider_health.record_failure(db, key.id, e)
                    # Provider-level failure (bad, forbidden or uncredited key) →
                    # skip the WHOLE provider immediately so we don't waste a
                    # round-trip on every other model behind the same failing key.
                    # Checked regardless of whether the error is "retryable" (e.g.
                    # Vercel's 403 "requires a valid credit card" is not).
                    if health == provider_health.ERROR:
                        skip_platforms.add(model.platform)
                        skip_models.add(model.id)
                        break
                    if rate_limit.is_retryable_error(e):
                        rate_limit.set_cooldown(model.platform, model.model_id, key.id)
                        rate_limit.record_rate_limit_hit(model.platform, model.model_id, key.id)
                        # Repeated quota/rate-limit from a provider → treat as a
                        # project-level limit and skip the rest of that provider.
                        if any(t in msg for t in (
                            "429", "quota", "rate limit", "rate-limit",
                            "resource_exhausted", "too many requests",
                        )):
                            platform_429[model.platform] = platform_429.get(model.platform, 0) + 1
                            # Abandon the whole provider on repeated 429s only if
                            # there's another provider to fall back to; if it's the
                            # only one, keep trying its models (some may have quota).
                            other_platforms = {m.platform for m in models} - skip_platforms - {model.platform}
                            if platform_429[model.platform] >= _PLATFORM_429_LIMIT and other_platforms:
                                skip_platforms.add(model.platform)
                            skip_models.add(model.id)
                            break
                        if "404" in msg or "not found" in msg:
                            skip_models.add(model.id)
                            break
                    else:
                        skip_models.add(model.id)
                        break

    raise RuntimeError(f"All models exhausted for streaming. Last error: {last_error}")


# ─── Deep Research: multi-model auto-continuation orchestrator ───────────────
# Hard caps so a runaway build can't loop forever or blow the wall clock.
DEEP_RESEARCH_MAX_ITERS = 12        # max model calls stitched into one answer
DEEP_RESEARCH_BUDGET_S = 240.0      # wall-clock cap for a full multi-model build
_DR_CONTEXT_CHARS = 80000           # tail of prior work handed to the next model
_DR_STALL_TIMEOUT_S = 30.0          # no token for this long → model is "stuck", switch


def _dr_continue_messages(base_messages: list[MessageDto], accumulated: str) -> list[MessageDto]:
    """Messages for a continuation segment: the base request, the work produced
    so far (so the next model can analyze it), and an instruction to resume
    exactly where it stopped without repeating anything."""
    if not accumulated:
        return list(base_messages)
    ctx = accumulated if len(accumulated) <= _DR_CONTEXT_CHARS else accumulated[-_DR_CONTEXT_CHARS:]
    resume = MessageDto(role="user", content=(
        "The assistant response above is your work so far but it was cut off. "
        "Continue the implementation EXACTLY where it stopped. Do NOT repeat, "
        "re-list, or restate any file or code already produced above; if you were "
        "in the middle of a file, resume from the exact point (even mid-line). "
        "Keep the same project structure and file-path headers, and keep going "
        "until the entire project is fully implemented."
    ))
    return list(base_messages) + [
        MessageDto(role="assistant", content=ctx),
        resume,
    ]


def _dr_footer(models_used: list[str]) -> str:
    """Trailing note listing every model that contributed to the answer."""
    if not models_used:
        return ""
    if len(models_used) == 1:
        return f"\n\n---\n*🔬 Deep Research · generated by **{models_used[0]}***"
    chain = " → ".join(models_used)
    return (f"\n\n---\n*🔬 Deep Research · assembled across "
            f"**{len(models_used)} models**: {chain}*")


_DR_EMPTY_MSG = (
    "I couldn't generate a Deep Research response right now — every "
    "large (400B+) model was unavailable or rate-limited. Please try again shortly."
)


async def _deep_research_relay(
    db: Session,
    models: list[ChatModel],
    messages: list[MessageDto],
    temperature: float | None,
    max_tokens: int | None,
) -> AsyncGenerator[dict, None]:
    """Core multi-model continuation used by BOTH the streaming and non-streaming
    Deep Research entry points, so the relay behaves identically either way.

    A single model keeps going while it only hits the token cap (finish_reason
    "length"); when a model rate-limits, stalls, or errors, the next eligible
    model receives the work so far and resumes exactly where it stopped. Yields
    dict events so each caller can render them as it likes:

      {"type": "handoff", "model": <display_name>}  — about to switch models
      {"type": "content", "text": <str>}            — a token delta
      {"type": "done", "models_used": [...], "produced_any": <bool>}  — final
    """
    accumulated = ""
    models_used: list[str] = []
    skip_platforms: set[str] = set()
    skip_model_ids: set[int] = set()
    idx = 0
    iters = 0
    deadline = time.monotonic() + DEEP_RESEARCH_BUDGET_S

    while iters < DEEP_RESEARCH_MAX_ITERS and time.monotonic() < deadline:
        iters += 1

        # Find the next usable model (with a live, non-cooled key) from idx.
        model = provider = key = None
        while idx < len(models):
            cand = models[idx]
            if cand.platform in skip_platforms or cand.id in skip_model_ids:
                idx += 1
                continue
            cand_provider = provider_registry.get(cand.platform)
            if cand_provider is None:
                skip_model_ids.add(cand.id)
                idx += 1
                continue
            keys = _apply_owner_scope(
                db.query(ApiKey).filter(
                    ApiKey.platform == cand.platform,
                    ApiKey.enabled == True,
                    ApiKey.status != "error",
                )
            ).all()
            cand_key = next(
                (k for k in keys
                 if not rate_limit.is_on_cooldown(cand.platform, cand.model_id, k.id)),
                None,
            )
            if cand_key is None:
                skip_model_ids.add(cand.id)
                idx += 1
                continue
            model, provider, key = cand, cand_provider, cand_key
            break

        if model is None:
            break  # no more usable large models

        # Announce a genuine hand-off so callers can show the relay (and so the
        # frontend's inter-event timeout keeps getting fed during the switch).
        if accumulated and models_used and model.display_name != models_used[-1]:
            yield {"type": "handoff", "model": model.display_name}

        seg_msgs = _dr_continue_messages(messages, accumulated)
        produced = ""
        finish_reason: str | None = None
        try:
            logger.info(f"[DeepResearch] Segment {iters}: {model.display_name} ({model.platform})")
            agen = provider.stream_chat_completion_ex(
                api_key=key.api_key,
                messages=seg_msgs,
                model_id=model.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            ).__aiter__()
            try:
                while True:
                    # Bound each token wait so a hung/"stuck" model (open
                    # connection, no output) hands off instead of blocking.
                    try:
                        ev = await asyncio.wait_for(
                            agen.__anext__(), timeout=_DR_STALL_TIMEOUT_S)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"{model.display_name} stalled — no output for "
                            f"{_DR_STALL_TIMEOUT_S:.0f}s")
                    if ev.get("type") == "content":
                        text = ev.get("text", "")
                        if text:
                            produced += text
                            accumulated += text
                            yield {"type": "content", "text": text}
                    elif ev.get("type") == "finish":
                        finish_reason = ev.get("reason")
            finally:
                try:
                    await agen.aclose()
                except Exception:
                    pass
            rate_limit.record_success(model.platform, model.model_id, key.id)
        except Exception as e:
            logger.warning(f"[DeepResearch] {model.display_name} error: {e}")
            if rate_limit.is_retryable_error(e):
                rate_limit.set_cooldown(model.platform, model.model_id, key.id)
                rate_limit.record_rate_limit_hit(model.platform, model.model_id, key.id)
                if any(t in str(e).lower() for t in (
                    "401", "403", "unauthorized", "forbidden",
                    "permission", "invalid api key", "authentication",
                )):
                    skip_platforms.add(model.platform)
            # Hand off to the next model; any partial text already produced stays
            # in `accumulated`, so the next model resumes from it.
            skip_model_ids.add(model.id)
            idx += 1
            continue

        if produced.strip():
            if not models_used or models_used[-1] != model.display_name:
                models_used.append(model.display_name)
        else:
            # Nothing produced and no error → drop this model, try the next.
            skip_model_ids.add(model.id)
            idx += 1
            continue

        if finish_reason == "length":
            # Truncated: same healthy model keeps going (most coherent).
            continue
        # Natural stop (or unknown reason) → the project is complete.
        break

    yield {
        "type": "done",
        "models_used": models_used,
        "produced_any": bool(accumulated.strip()),
    }


async def route_deep_research_stream(
    db: Session,
    messages: list[MessageDto],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> StreamRouteResult:
    """Stream a Deep Research answer, auto-continuing across large (>=400B)
    models so the user never has to type "continue".

    A single model keeps going while it only hits the token cap (finish_reason
    "length"); when a model rate-limits or errors, the next eligible model
    analyzes the work so far and resumes it. The answer ends with a footer
    listing every model that contributed."""
    models = _get_ordered_models(db, None, messages, deep_research=True)
    if not models:
        raise DeepResearchUnavailableError(
            "Deep Research needs a large model (400B+ parameters), but none are "
            "available with your current keys. Add a provider that offers one — "
            "e.g. NVIDIA NIM (Llama-3.1-405B or DeepSeek-R1) — then try again."
        )

    async def _gen() -> AsyncGenerator[str, None]:
        async for ev in _deep_research_relay(db, models, messages, temperature, max_tokens):
            kind = ev.get("type")
            if kind == "content":
                yield ev["text"]
            elif kind == "handoff":
                yield f"\n\n> ↻ *Continuing with **{ev['model']}**…*\n\n"
            elif kind == "done":
                if ev["models_used"]:
                    yield _dr_footer(ev["models_used"])
                elif not ev["produced_any"]:
                    yield _DR_EMPTY_MSG

    first = models[0]
    return StreamRouteResult(
        stream=_gen(),
        model_id=first.model_id,
        platform=first.platform,
        display_name="Deep Research",
        attempts=0,
    )


async def route_deep_research(
    db: Session,
    messages: list[MessageDto],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> RouteResult:
    """Non-streaming Deep Research: run the SAME multi-model relay to completion,
    then return the fully assembled answer (with the models-used footer) as one
    RouteResult. This is what makes non-streaming Deep Research requests continue
    across models instead of stopping at the first model's token cap."""
    models = _get_ordered_models(db, None, messages, deep_research=True)
    if not models:
        raise DeepResearchUnavailableError(
            "Deep Research needs a large model (400B+ parameters), but none are "
            "available with your current keys. Add a provider that offers one — "
            "e.g. NVIDIA NIM (Llama-3.1-405B or DeepSeek-R1) — then try again."
        )

    parts: list[str] = []
    models_used: list[str] = []
    produced_any = False
    async for ev in _deep_research_relay(db, models, messages, temperature, max_tokens):
        kind = ev.get("type")
        if kind == "content":
            parts.append(ev["text"])
        elif kind == "done":
            models_used = ev["models_used"]
            produced_any = ev["produced_any"]

    content = "".join(parts)
    if models_used:
        content += _dr_footer(models_used)
    elif not produced_any:
        content = _DR_EMPTY_MSG

    first = models[0]
    return RouteResult(
        content=content,
        model_id=first.model_id,
        platform=first.platform,
        display_name="Deep Research",
        attempts=len(models_used),
    )
