"""Memory Manager — auto-search at turn start, auto-store at turn end (req 8.7, 8.8).

Both wrap the registered memory tools. Failures are logged at WARNING and never
raised so memory issues cannot interrupt a conversation.
"""
import logging
from typing import Optional

from .tool_registry import ToolRegistry, tool_registry as _default_registry

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or _default_registry

    async def auto_search(self, conversation_id: int, query: str) -> Optional[str]:
        """Return a formatted '## Relevant Memory' block, or None."""
        tool = self.registry.get("memory_search")
        if tool is None or not tool.enabled:
            return None
        try:
            result = await tool.fn(query=query, conversation_id=conversation_id, top_k=5)
            chunks = (result or {}).get("chunks", [])
            if not chunks:
                return None
            texts = "\n".join(f"- {c['text']}" for c in chunks)
            return f"## Relevant Memory\n{texts}"
        except Exception as exc:
            logger.warning(f"[Memory] auto_search failed: {exc}")
            return None

    async def auto_store(
        self,
        conversation_id: int,
        user_message: str,
        assistant_message: str,
        user_id: Optional[int] = None,
    ) -> None:
        """Persist the user + assistant turns; log WARNING on failure (req 8.7)."""
        tool = self.registry.get("memory_store_batch")
        if tool is None or not tool.enabled:
            return
        try:
            await tool.fn(
                items=[
                    {"text": user_message, "conversation_id": conversation_id},
                    {"text": assistant_message, "conversation_id": conversation_id},
                ],
                user_id=user_id,
            )
        except Exception as exc:
            logger.warning(f"[Memory] auto_store failed: {exc}")


# Module-level singleton
memory_manager = MemoryManager()
