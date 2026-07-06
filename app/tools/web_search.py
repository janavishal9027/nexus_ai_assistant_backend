"""
Web Search Tool for Agent Real-Time Data Access

This tool provides intelligent web search capabilities with automatic provider fallback:
- Primary: Tavily API (LLM-optimized, requires TAVILY_API_KEY)
- Fallback: DuckDuckGo (keyless, uses ddgs package)

The tool is registered with the tool_registry and returns structured results with
source URLs for citation tracking.
"""

import asyncio
import logging
from typing import Any

import httpx

from ..services.tool_registry import tool_registry
from ..services.web_search import get_tavily_key

logger = logging.getLogger(__name__)


@tool_registry.tool(
    name="web_search",
    description=(
        "Search the web for real-time information. Use when the user asks about "
        "current events, news, prices, weather, stock data, or anything requiring "
        "up-to-date information from the internet. Returns a list of search results "
        "with titles, snippets, and URLs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to execute"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
                "default": 5,
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["query"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "snippet": {"type": "string"},
                        "url": {"type": "string"}
                    }
                }
            },
            "total_found": {"type": "integer"}
        },
        "required": ["results", "total_found"]
    },
    timeout_seconds=15.0,
    requires_auth=False
)
async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Execute a web search using Tavily API or DuckDuckGo fallback.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        Dict containing:
        - results: List of dicts with title, snippet, url
        - total_found: Number of results found
        - _sources: List of source dicts for citation tracking (extracted by executor)
    
    The return value format allows the ToolExecutor to extract _sources into
    ToolResult.sources for citation tracking.
    """
    logger.info(f"[web_search] Starting search for: '{query[:80]}' (max_results={max_results})")

    # Try Tavily first if the user configured a Tavily key in the app (per-user,
    # stored in the DB) — no longer read from a hardcoded backend env var.
    tavily_api_key = get_tavily_key()
    if tavily_api_key:
        logger.info("[web_search] Attempting Tavily API search")
        try:
            result = await _tavily_search(query, tavily_api_key, max_results)
            logger.info(f"[web_search] Tavily succeeded with {result['total_found']} results")
            return result
        except Exception as e:
            logger.warning(f"[web_search] Tavily failed: {e}, falling back to DuckDuckGo")
    else:
        logger.info("[web_search] No Tavily API key, using DuckDuckGo")
    
    # Fallback to DuckDuckGo
    try:
        result = await _ddg_search(query, max_results)
        logger.info(f"[web_search] DuckDuckGo succeeded with {result['total_found']} results")
        return result
    except Exception as e:
        logger.error(f"[web_search] DuckDuckGo failed: {e}", exc_info=True)
        # Both providers failed - raise exception to be caught by ToolExecutor
        raise RuntimeError("Web search unavailable: both Tavily and DuckDuckGo failed")


async def _tavily_search(query: str, api_key: str, max_results: int) -> dict[str, Any]:
    """
    Execute search using Tavily API.
    
    Args:
        query: Search query
        api_key: Tavily API key
        max_results: Maximum results to return
    
    Returns:
        Dict with results, total_found, and _sources
        Returns {"results": [], "total_found": 0} when no results found
    
    Raises:
        Exception: On network error or API failure
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
            },
        )
        response.raise_for_status()
        data = response.json()
    
    # Transform Tavily results to standard format
    results = []
    sources = []
    
    tavily_results = data.get("results", [])
    
    # Handle no results case
    if not tavily_results:
        logger.info("[web_search/Tavily] No results found")
        return {
            "results": [],
            "total_found": 0,
            "_sources": []
        }
    
    for item in tavily_results:
        title = item.get("title", "No title")
        snippet = item.get("content", "")[:500]  # Limit snippet length
        url = item.get("url", "")
        
        results.append({
            "title": title,
            "snippet": snippet,
            "url": url
        })
        
        # Add to sources for citation tracking
        sources.append({
            "url": url,
            "title": title,
            "snippet": snippet[:200]  # Shorter snippet for citations
        })
    
    return {
        "results": results,
        "total_found": len(results),
        "_sources": sources  # Special key for ToolExecutor to extract
    }


async def _ddg_search(query: str, max_results: int) -> dict[str, Any]:
    """
    Execute search using DuckDuckGo (keyless fallback).
    
    Uses the ddgs package which is synchronous, so we wrap it in asyncio.to_thread()
    to avoid blocking the event loop.
    
    Args:
        query: Search query
        max_results: Maximum results to return
    
    Returns:
        Dict with results, total_found, and _sources
        Returns {"results": [], "total_found": 0} when no results found
    
    Raises:
        ImportError: If ddgs package is not installed
        Exception: On search failure
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("[web_search] ddgs package not installed; install with: pip install ddgs")
        raise ImportError("ddgs package required for DuckDuckGo search")
    
    logger.info(f"[web_search/DDG] Searching for '{query[:60]}', max={max_results}")
    
    def _run_ddg_search() -> list[dict]:
        """Synchronous DDG search wrapped for asyncio.to_thread()"""
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    
    # Run synchronous search in thread pool
    raw_results = await asyncio.to_thread(_run_ddg_search)
    logger.info(f"[web_search/DDG] Retrieved {len(raw_results)} raw results")
    
    # Handle no results - return empty result structure instead of failing
    if not raw_results:
        logger.info("[web_search/DDG] No results found")
        return {
            "results": [],
            "total_found": 0,
            "_sources": []
        }
    
    # Transform DuckDuckGo results to standard format
    results = []
    sources = []
    
    for item in raw_results:
        title = item.get("title", "No title")
        snippet = item.get("body", "")[:500]  # Limit snippet length
        url = item.get("href", "")
        
        results.append({
            "title": title,
            "snippet": snippet,
            "url": url
        })
        
        # Add to sources for citation tracking
        sources.append({
            "url": url,
            "title": title,
            "snippet": snippet[:200]  # Shorter snippet for citations
        })
    
    return {
        "results": results,
        "total_found": len(results),
        "_sources": sources  # Special key for ToolExecutor to extract
    }
