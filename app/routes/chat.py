"""
Chat endpoints — delegates to the Agent orchestration layer.
"""
import asyncio
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel

from ..database import get_db
from fastapi import Response
from ..models.schemas import (
    ChatRequest, ChatResponse, ClarifyRequest, ClarifyResponse,
    SuggestRequest, SuggestResponse, DocumentDecisionRequest,
    DocumentDecisionResponse, ExportRequest,
)
from ..models.db_models import Conversation, Message, Account
from ..services.agent import agent_chat, agent_stream_chat
from ..services.multimodal_chat import multimodal_stream_chat
from ..services.clarifier import assess_clarification
from ..services.suggester import suggest_followups
from ..services.starters import generate_starters
from ..services.document_decision import decide_document
from ..services.document_export import export_document
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


class FeedbackRequest(BaseModel):
    conversation_id: Optional[int] = None
    message_index: Optional[int] = None
    rating: int                          # +1 up, -1 down, 0 clears
    user_text: Optional[str] = None
    assistant_text: Optional[str] = None


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest, db: Session = Depends(get_db),
                          account: Account = Depends(get_current_account)):
    """Capture a 👍/👎 on an assistant message (Part D). One row per (owner,
    conversation, message_index); the Reflector reads these to learn what lands."""
    from ..models.db_models import MessageFeedback
    _assert_conversation_access(db, body.conversation_id, account.id)
    rating = 1 if body.rating > 0 else (-1 if body.rating < 0 else 0)
    row = (db.query(MessageFeedback)
           .filter(MessageFeedback.owner_id == account.id,
                   MessageFeedback.conversation_id == body.conversation_id,
                   MessageFeedback.message_index == body.message_index).first())
    if rating == 0:                      # toggled off
        if row:
            db.delete(row)
            db.commit()
        return {"ok": True, "rating": 0}
    if row:
        row.rating = rating
        row.assistant_text = body.assistant_text or row.assistant_text
        row.user_text = body.user_text or row.user_text
        row.updated_at = datetime.now(timezone.utc)
    else:
        db.add(MessageFeedback(
            owner_id=account.id, conversation_id=body.conversation_id,
            message_index=body.message_index, rating=rating,
            user_text=body.user_text, assistant_text=body.assistant_text))
    db.commit()
    return {"ok": True, "rating": rating}


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


@router.post("/clarify", response_model=ClarifyResponse)
async def clarify(request: ClarifyRequest, db: Session = Depends(get_db),
                  account: Account = Depends(get_current_account)):
    """Pre-flight clarification gate (chat-module A.2): decide whether this turn
    needs a blocking clarifying question before the answer streams. Fails open
    (clarify=false) so a clarifier hiccup never blocks a chat."""
    request_context.set_owner_id(account.id)
    try:
        result = await assess_clarification(
            db, request.message, history=request.history,
            owner_id=account.id, model=request.model,
        )
    except Exception as e:
        logger.warning(f"[Clarify] failed, proceeding without: {e}")
        return ClarifyResponse(clarify=False)
    qs = result.get("questions") or []
    # Keep the legacy single `question` populated with the first, for any client
    # that still reads it.
    return ClarifyResponse(clarify=bool(qs), questions=qs,
                           question=qs[0] if qs else None)


@router.post("/suggest", response_model=SuggestResponse)
async def suggest(request: SuggestRequest, db: Session = Depends(get_db),
                  account: Account = Depends(get_current_account)):
    """Post-turn follow-up suggestions (chat-module A.2 · the Suggester agent).
    Fails open with an empty list so it never disrupts the chat."""
    request_context.set_owner_id(account.id)
    try:
        items = await suggest_followups(
            db, request.conversation_id, owner_id=account.id, model=request.model)
    except Exception as e:
        logger.warning(f"[Suggest] failed: {e}")
        items = []
    return SuggestResponse(suggestions=items)


@router.get("/starters")
async def starters(model: Optional[str] = Query(None),
                   db: Session = Depends(get_db),
                   account: Account = Depends(get_current_account)):
    """Fresh, LLM-generated conversation starters for the new-chat empty state:
    a current-tech/coding one, a globally-relevant idea, and a writing task.
    Regenerated each call; fails open to a varied default set."""
    request_context.set_owner_id(account.id)
    try:
        items = await generate_starters(db, owner_id=account.id, model=model)
    except Exception as e:
        logger.warning(f"[Starters] failed: {e}")
        items = []
    return {"starters": items}


@router.post("/document-decision", response_model=DocumentDecisionResponse)
async def document_decision(request: DocumentDecisionRequest, db: Session = Depends(get_db),
                            account: Account = Depends(get_current_account)):
    """Backend-authoritative document triage (A.4): whether the last answer is
    export-worthy and in which format(s). Fails open (document=false)."""
    try:
        if request.content is not None:
            from ..services.document_decision import classify_content
            d = classify_content(request.content)
        elif request.conversation_id is not None:
            d = decide_document(db, request.conversation_id, account.id)
        else:
            d = {"document": False, "format": None, "formats": []}
    except Exception as e:
        logger.warning(f"[DocDecision] failed: {e}")
        d = {"document": False, "format": None, "formats": []}
    return DocumentDecisionResponse(**d)


@router.post("/export")
async def export(request: ExportRequest, account: Account = Depends(get_current_account)):
    """Generate a downloadable document (A.4) from answer content in the chosen
    format. Returns the file bytes with a Content-Disposition attachment."""
    from urllib.parse import quote
    try:
        data, mime, filename = export_document(
            request.content, request.format, request.title or "document",
            clean=request.clean)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Export] failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
    return Response(
        content=data, media_type=mime,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "X-Filename": filename,
        },
    )


@router.post("/stream")
async def stream_message(request: ChatRequest, db: Session = Depends(get_db),
                         account: Account = Depends(get_current_account)):
    """Send a message via the agent orchestrator (streaming SSE)."""
    logger.info(f"[ChatRouter/Stream] Received request: message='{request.message[:60]}', model={request.model}")
    _assert_conversation_access(db, request.conversation_id, account.id)
    request_context.set_owner_id(account.id)   # scope provider keys to this user
    try:
        # Attachments (images/documents) take an isolated multimodal path that
        # extracts document text and routes images to a vision model; everything
        # else goes through the normal agent orchestration.
        if request.attachments:
            conversation_id, result, citations_text = await multimodal_stream_chat(
                db, request, owner_id=account.id)
        else:
            conversation_id, result, citations_text = await agent_stream_chat(db, request, owner_id=account.id)

        async def event_generator():
            full_content = ""
            try:
                # Reveal the conversation id up front so the client binds to THIS
                # conversation even if generation fails mid-stream — a Retry then
                # re-runs in the same chat instead of spawning a duplicate.
                yield json.dumps({
                    "content": "",
                    "conversationId": conversation_id,
                    "done": False,
                })
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
                # Persist the PARTIAL answer so a reload shows the same stopped
                # response on every device (consistent with the WS path), instead
                # of dropping it (mobile would show nothing) or keeping a full one.
                if full_content.strip():
                    try:
                        db.add(Message(
                            conversation_id=conversation_id, role="assistant",
                            content=full_content, model_used=result.model_id,
                            platform_used=result.platform, stopped=True))
                        conv = db.query(Conversation).filter(
                            Conversation.id == conversation_id).first()
                        if conv:
                            conv.updated_at = datetime.now(timezone.utc)
                        db.commit()
                    except Exception:
                        db.rollback()
                return
            except Exception as e:
                logger.error(f"[ChatRouter/Stream] Error in stream generator: {e}", exc_info=True)
                # Carry the conversation id on the error too, so a Retry reuses
                # this conversation rather than creating a new one.
                yield json.dumps({"error": str(e), "conversationId": conversation_id, "done": True})

        return EventSourceResponse(event_generator())

    except asyncio.CancelledError:
        logger.warning(f"[ChatRouter/Stream] Request cancelled before streaming started")
        # Return empty response on cancellation
        async def cancelled_gen():
            return
        return EventSourceResponse(cancelled_gen())
    except Exception as e:
        logger.error(f"[Chat/Stream] Error: {e}", exc_info=True)
        # Capture the message now: Python clears `e` when this except block
        # exits, but error_gen runs later (while streaming the response).
        err_msg = str(e)
        # If the conversation was already created before the failure, surface its
        # id so a Retry reuses it instead of spawning a duplicate chat.
        err_cid = getattr(e, "conversation_id", None)

        async def error_gen():
            yield json.dumps({"error": err_msg, "conversationId": err_cid, "done": True})

        return EventSourceResponse(error_gen())
