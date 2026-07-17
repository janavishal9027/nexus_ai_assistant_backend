"""Memory Manager — auto-search at turn start, auto-store at turn end (req 8.7, 8.8).

Part D: delegates to the layered memory modules — episodic (durable, per-user,
real embeddings) for recall/persistence, and working (in-session ring buffer) for
the current conversation's scratch. Owner-scoped by the authenticated account id.
Failures are logged at WARNING and never raised so memory can't interrupt a turn.
"""
import logging
from typing import Optional

from ..config import get_settings
from ..memory import (episodic, memory_graph, memory_prefs, project_brain,
                      semantic, skills_extractor, working)
from ..rag import knowledge_graph

logger = logging.getLogger(__name__)


class MemoryManager:
    async def auto_search(self, conversation_id: int, query: str,
                          owner_id: Optional[int] = None) -> Optional[str]:
        """Return the turn-start memory blocks — '## Relevant Memory' (episodic)
        and '## About the user' (semantic skills). Owner-scoped; None if empty.
        Honors the user's Settings → Memory switches."""
        if owner_id is None:
            return None
        prefs = memory_prefs.effective(owner_id)
        if not prefs["recall_enabled"]:
            return None            # user turned recall off — answer with no memory
        blocks: list[str] = []
        try:
            chunks = await episodic.search(owner_id, query,
                                           conversation_id=conversation_id,
                                           top_k=5, scope="user")
            if chunks:
                blocks.append("## Relevant Memory\n" +
                              "\n".join(f"- {c['text']}" for c in chunks))
        except Exception as exc:
            logger.warning(f"[Memory] auto_search episodic failed: {exc}")
        try:
            if prefs["semantic_recall_enabled"]:
                skills = await semantic.search(
                    owner_id, query, top_k=get_settings().memory_skill_recall_top_k)
                if skills:
                    blocks.append("## About the user\n" +
                                  "\n".join(f"- {s['content']}" for s in skills))
        except Exception as exc:
            logger.warning(f"[Memory] auto_search semantic failed: {exc}")
        # Personal memory graph (Part D Phase 5): the people/orgs + tools/tech
        # relevant to this message — query-matched, so low token cost.
        try:
            if prefs["graph_enabled"]:
                facts = await memory_graph.render(owner_id, query)
                if facts:
                    blocks.append("## What I know about you\n" + facts)
        except Exception as exc:
            logger.warning(f"[Memory] auto_search graph failed: {exc}")
        return "\n\n".join(blocks) if blocks else None

    async def auto_store(self, conversation_id: int, user_message: str,
                         assistant_message: str, owner_id: Optional[int] = None,
                         user_id: Optional[int] = None) -> None:
        """Push the turn into working memory (always — it's in-process and dies
        with the conversation) and persist it to the owner's durable episodic
        memory. Honors the user's Settings → Memory switches: with recording off
        nothing durable is written, and the layers that derive FROM the durable
        log (reflection, graph) are skipped too."""
        try:
            working.remember(conversation_id, "user", user_message)
            working.remember(conversation_id, "assistant", assistant_message)
        except Exception:
            pass
        if owner_id is None:
            return
        prefs = memory_prefs.effective(owner_id)
        if not prefs["record_enabled"]:
            return                 # user turned recording off — learn nothing
        try:
            await episodic.store(owner_id, conversation_id,
                                 [user_message, assistant_message], user_id=user_id)
        except Exception as exc:
            logger.warning(f"[Memory] auto_store failed: {exc}")
        # Debounced background reflection: distil episodes+feedback → skills,
        # project knowledge → project brain, and content → knowledge graph.
        try:
            if prefs["reflect_enabled"]:
                skills_extractor.maybe_reflect(owner_id, conversation_id)
        except Exception:
            pass
        try:
            project_brain.maybe_reflect(owner_id, conversation_id)
        except Exception:
            pass
        try:
            # async: must be awaited or the coroutine is created and discarded.
            await knowledge_graph.maybe_extract(owner_id, conversation_id)
        except Exception:
            pass
        # Personal memory graph: people/orgs + tools/tech about the user.
        try:
            if prefs["graph_enabled"]:
                await memory_graph.maybe_extract(owner_id, conversation_id)
        except Exception:
            pass


# Module-level singleton
memory_manager = MemoryManager()
