"""
Real-time web search for the agent.

Gives the LLM up-to-date information it can't have from training data.
Uses Tavily (LLM-optimized, needs a free key) when TAVILY_API_KEY is set,
otherwise falls back to keyless DuckDuckGo search.
"""
import asyncio
import logging

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

# Phrases that unambiguously signal the user wants live/current data.
# Kept intentionally narrow — generic words like "current" or bare year numbers
# ("2025") matched too broadly and triggered unnecessary searches for every
# coding or writing prompt.
_REALTIME_SIGNALS = (
    "today", "today's", "tonight", "right now",
    "this week", "this month", "as of", "up to date", "up-to-date",
    "breaking news", "live score", "live update",
    "news", "weather", "temperature", "forecast",
    "stock price", "share price", "exchange rate",
    "who won", "who is the current president", "who is the current ceo",
    "how much is", "what time is it",
    "just announced", "just released", "just launched",
    "trending", "happening now", "update on",
    "status of",
)

# Words that look like real-time signals but usually aren't (programming, writing, etc.)
_FALSE_POSITIVE_SIGNALS = (
    "code", "function", "class", "api", "implement", "write", "create",
    "design", "explain", "how to", "what is", "what are", "difference between",
    "example", "tutorial", "fix", "debug", "error", "build",
)


def needs_web_search(query: str) -> bool:
    """Heuristic: does this query actually need real-time/current information?

    Uses a narrow allowlist of unambiguous real-time signals and suppresses
    false positives from programming/writing prompts, which were the main
    source of unnecessary pre-response LLM round-trips.
    """
    q = query.lower()
    # Bail out fast if this looks like a coding/writing prompt
    if any(fp in q for fp in _FALSE_POSITIVE_SIGNALS):
        return False
    return any(sig in q for sig in _REALTIME_SIGNALS)


def get_tavily_key() -> str:
    """Resolve the Tavily API key for the current request.

    Source of truth is the frontend: a `tavily` key added by the authenticated
    user (or a shared/global one) in the API Keys UI, stored in the database and
    scoped by account. Falls back to the optional TAVILY_API_KEY env var for
    backward compatibility. Returns '' when none is configured (→ DuckDuckGo).
    """
    try:
        from sqlalchemy import or_
        from ..database import SessionLocal
        from ..models.db_models import ApiKey
        from . import request_context

        owner_id = request_context.get_owner_id()
        db = SessionLocal()
        try:
            q = db.query(ApiKey).filter(ApiKey.platform == "tavily", ApiKey.enabled == True)
            if owner_id is not None:
                q = q.filter(or_(ApiKey.owner_id == owner_id, ApiKey.owner_id.is_(None)))
            # Prefer the user's own key over a shared/global one.
            row = q.order_by(ApiKey.owner_id.isnot(None).desc()).first()
            if row and row.api_key:
                return row.api_key
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"[WebSearch] Tavily key lookup failed: {e}")
    return getattr(get_settings(), "tavily_api_key", "") or ""


async def web_search(query: str, max_results: int = 5) -> str | None:
    """Return formatted search results for the query, or None on failure."""
    logger.info(f"[WebSearch] Starting search for: '{query[:80]}'")
    try:
        tavily_key = get_tavily_key()
        if tavily_key:
            logger.info("[WebSearch] Using Tavily API")
            return await _tavily_search(query, tavily_key, max_results)
        logger.info("[WebSearch] Using DuckDuckGo (no Tavily key)")
        return await _ddg_search(query, max_results)
    except Exception as e:  # never let search break the chat
        logger.error(f"[WebSearch] '{query[:60]}' failed: {e}", exc_info=True)
        return None


async def _tavily_search(query: str, api_key: str, max_results: int) -> str | None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    parts: list[str] = []
    if data.get("answer"):
        parts.append(f"Answer summary: {data['answer']}")
    for i, r in enumerate(data.get("results", []), 1):
        parts.append(
            f"[{i}] {r.get('title', '')}\n"
            f"{(r.get('content') or '')[:500]}\n"
            f"Source: {r.get('url', '')}"
        )
    logger.info(f"[WebSearch] Tavily returned {len(data.get('results', []))} results")
    return "\n\n".join(parts) if parts else None


async def _ddg_search(query: str, max_results: int) -> str | None:
    """Keyless DuckDuckGo search (sync lib run off the event loop)."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("[WebSearch] ddgs package not installed; pip install ddgs")
        return None

    logger.info(f"[WebSearch/DDG] Searching for '{query[:60]}', max={max_results}")
    
    def _run() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        results = await asyncio.to_thread(_run)
        logger.info(f"[WebSearch/DDG] Retrieved {len(results)} results")
        
        if not results:
            logger.warning("[WebSearch/DDG] No results found")
            return None
            
        parts = [
            f"[{i}] {r.get('title', 'No title')}\n"
            f"{(r.get('body') or 'No description')[:500]}\n"
            f"Source: {r.get('href', '')}"
            for i, r in enumerate(results, 1)
        ]
        result_text = "\n\n".join(parts)
        logger.info(f"[WebSearch/DDG] Formatted {len(result_text)} chars")
        return result_text
    except Exception as e:
        logger.error(f"[WebSearch/DDG] Error during search: {e}", exc_info=True)
        return None
