"""
Agent Orchestration Layer.

Manages the conversation context, system prompt, and structures the LLM
interaction. The agent:
1. Prepends the system prompt from config
2. Trims conversation history to fit context limits
3. Sends to the fallback router for model selection
4. Returns structured response with metadata
"""
import json
import logging
from pathlib import Path
from sqlalchemy.orm import Session

from ..models.schemas import MessageDto, ChatRequest
from ..models.db_models import Conversation, Message
from .fallback_router import route_chat, route_stream_chat, RouteResult, StreamRouteResult
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Load config once
_config_path = Path(__file__).parent.parent / "providers_config.json"
_config: dict = {}


def get_config() -> dict:
    global _config
    if not _config:
        _config = json.loads(_config_path.read_text(encoding="utf-8"))
    return _config


def reload_config():
    """Reload config from disk (call after edits)."""
    global _config
    _config = json.loads(_config_path.read_text(encoding="utf-8"))
    return _config


def _build_agent_messages(
    db: Session,
    conversation_id: int,
    user_message: str,
    history: list[MessageDto] | None = None,
) -> list[MessageDto]:
    """Build the message list with system prompt and context window trimming."""
    config = get_config()
    agent_cfg = config.get("agent", {})
    system_prompt = agent_cfg.get("system_prompt", "You are a helpful assistant.")
    max_context = agent_cfg.get("max_context_messages", 20)

    messages: list[MessageDto] = []

    # System prompt always first
    messages.append(MessageDto(role="system", content=system_prompt))

    # Load conversation history
    if history:
        context_messages = history
    else:
        db_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        context_messages = [MessageDto(role=m.role, content=m.content) for m in db_messages]

    # Trim to max context (keep most recent messages)
    if len(context_messages) > max_context:
        context_messages = context_messages[-max_context:]

    messages.extend(context_messages)
    return messages


async def agent_chat(db: Session, request: ChatRequest) -> dict:
    """
    Main agent entry point for non-streaming chat.
    Orchestrates: conversation management → context building → LLM routing → response persistence.
    """
    config = get_config()
    agent_cfg = config.get("agent", {})

    conversation_id = request.conversation_id

    # Create conversation if new
    if conversation_id is None:
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        conv = Conversation(title=title or "New Chat")
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id

    # Save user message
    user_msg = Message(conversation_id=conversation_id, role="user", content=request.message)
    db.add(user_msg)
    db.commit()

    # Build agent-managed messages
    messages = _build_agent_messages(db, conversation_id, request.message, request.history)

    # Route through fallback system
    result: RouteResult = await route_chat(
        db=db,
        messages=messages,
        requested_model=request.model,
        temperature=request.temperature or agent_cfg.get("default_temperature"),
        max_tokens=request.max_tokens or agent_cfg.get("default_max_tokens"),
    )

    # Save assistant response
    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=result.content,
        model_used=result.model_id,
        platform_used=result.platform,
    )
    db.add(assistant_msg)

    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv:
        conv.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {
        "conversation_id": conversation_id,
        "content": result.content,
        "model": result.display_name,
        "platform": result.platform,
        "fallback_attempts": result.attempts,
    }


async def agent_stream_chat(db: Session, request: ChatRequest) -> tuple[int, StreamRouteResult]:
    """
    Main agent entry point for streaming chat.
    Returns (conversation_id, StreamRouteResult) so the caller can handle SSE.
    """
    config = get_config()
    agent_cfg = config.get("agent", {})

    conversation_id = request.conversation_id

    if conversation_id is None:
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        conv = Conversation(title=title or "New Chat")
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id

    user_msg = Message(conversation_id=conversation_id, role="user", content=request.message)
    db.add(user_msg)
    db.commit()

    messages = _build_agent_messages(db, conversation_id, request.message, request.history)

    result: StreamRouteResult = await route_stream_chat(
        db=db,
        messages=messages,
        requested_model=request.model,
        temperature=request.temperature or agent_cfg.get("default_temperature"),
        max_tokens=request.max_tokens or agent_cfg.get("default_max_tokens"),
    )

    return conversation_id, result
