"""
Chat endpoints — delegates to the Agent orchestration layer.
"""
import asyncio
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from ..database import get_db
from ..models.schemas import ChatRequest, ChatResponse
from ..models.db_models import Conversation, Message, Account
from ..services.agent import agent_chat, agent_stream_chat
from ..services.auth import get_current_account
from ..services import request_context

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


def _assert_conversation_access(db: Session, conversation_id, account_id: int) -> None:
    """Deny cross-account access to an existing conversation (authorization)."""
    if conversation_id is None:
        return
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is not None and conv.owner_id is not None and conv.owner_id != account_id:
        raise HTTPException(status_code=403, detail="You do not have access to this conversation")


@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest, db: Session = Depends(get_db),
                       account: Account = Depends(get_current_account)):
    """Send a message via the agent orchestrator (non-streaming)."""
    logger.info(f"[ChatRouter] Received request: message='{request.message[:60]}', model={request.model}")
    _assert_conversation_access(db, request.conversation_id, account.id)
    request_context.set_owner_id(account.id)   # scope provider keys to this user
    try:
        result = await agent_chat(db, request, owner_id=account.id)

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
async def stream_message(request: ChatRequest, db: Session = Depends(get_db),
                         account: Account = Depends(get_current_account)):
    """Send a message via the agent orchestrator (streaming SSE)."""
    logger.info(f"[ChatRouter/Stream] Received request: message='{request.message[:60]}', model={request.model}")
    _assert_conversation_access(db, request.conversation_id, account.id)
    request_context.set_owner_id(account.id)   # scope provider keys to this user
    try:
        conversation_id, result, citations_text = await agent_stream_chat(db, request, owner_id=account.id)

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

                # Append citations as a final non-streaming chunk after stream ends
                if citations_text:
                    full_content += f"\n\n{citations_text}"
                    yield json.dumps({
                        "content": f"\n\n{citations_text}",
                        "model": result.display_name,
                        "platform": result.platform,
                        "done": False,
                    })

                # Persist the complete response (including citations)
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
            except asyncio.CancelledError:
                logger.warning(f"[ChatRouter/Stream] Client disconnected during streaming")
                # Don't yield error, just let the connection close gracefully
                return
            except Exception as e:
                logger.error(f"[ChatRouter/Stream] Error in stream generator: {e}", exc_info=True)
                yield json.dumps({"error": str(e), "done": True})

        return EventSourceResponse(event_generator())

    except asyncio.CancelledError:
        logger.warning(f"[ChatRouter/Stream] Request cancelled before streaming started")
        # Return empty response on cancellation
        async def cancelled_gen():
            return
        return EventSourceResponse(cancelled_gen())
    except Exception as e:
        logger.error(f"[Chat/Stream] Error: {e}", exc_info=True)

        async def error_gen():
            yield json.dumps({"error": str(e), "done": True})

        return EventSourceResponse(error_gen())
