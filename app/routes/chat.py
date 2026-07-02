"""
Chat endpoints — delegates to the Agent orchestration layer.
"""
import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from ..database import get_db
from ..models.schemas import ChatRequest, ChatResponse
from ..models.db_models import Conversation, Message
from ..services.agent import agent_chat, agent_stream_chat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """Send a message via the agent orchestrator (non-streaming)."""
    try:
        result = await agent_chat(db, request)

        return ChatResponse(
            conversation_id=result["conversation_id"],
            content=result["content"],
            model=result["model"],
            platform=result["platform"],
            fallback_attempts=result["fallback_attempts"],
        )

    except Exception as e:
        logger.error(f"[Chat] Error: {e}")
        return JSONResponse(
            status_code=503,
            content={"conversation_id": None, "content": f"Error: {e}",
                     "model": None, "platform": None, "fallback_attempts": 0},
        )


@router.post("/stream")
async def stream_message(request: ChatRequest, db: Session = Depends(get_db)):
    """Send a message via the agent orchestrator (streaming SSE)."""
    try:
        conversation_id, result = await agent_stream_chat(db, request)

        async def event_generator():
            full_content = ""
            try:
                async for chunk in result.stream:
                    full_content += chunk
                    yield json.dumps({
                        "content": chunk,
                        "model": result.display_name,
                        "platform": result.platform,
                        "done": False,
                    })

                # Persist the complete response
                assistant_msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_content,
                    model_used=result.model_id,
                    platform_used=result.platform,
                )
                db.add(assistant_msg)
                conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
                if conv:
                    conv.updated_at = datetime.now(timezone.utc)
                db.commit()

                yield json.dumps({
                    "content": "",
                    "model": result.display_name,
                    "platform": result.platform,
                    "conversationId": conversation_id,
                    "done": True,
                })
            except Exception as e:
                yield json.dumps({"error": str(e), "done": True})

        return EventSourceResponse(event_generator())

    except Exception as e:
        logger.error(f"[Chat/Stream] Error: {e}")

        async def error_gen():
            yield json.dumps({"error": str(e), "done": True})

        return EventSourceResponse(error_gen())
