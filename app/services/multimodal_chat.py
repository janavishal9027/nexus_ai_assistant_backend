"""Multimodal chat: a chat turn that carries file/image attachments.

Kept OUT of the agent-orchestration path — when a request has attachments the
chat route calls this instead of ``agent_stream_chat``. Documents have their
text extracted (reusing the RAG chunker's extractors) and added as context;
images are handed to a vision-capable model via the provider's multimodal
message format. Returns the SAME ``(conversation_id, StreamRouteResult,
citations_text)`` tuple as ``agent_stream_chat`` so the SSE route generator is
reused unchanged.
"""
import asyncio
import base64
import binascii
import logging
from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..models.schemas import ChatRequest, MessageDto
from ..models.db_models import Conversation, Message, ChatModel, ApiKey
from .fallback_router import route_stream_chat, _is_chatty
from .rag_chunking import extract_text, clean_text
from .rag_ingestion import ingest_conversation_document
from .agent import RESPONSE_FOLLOWUP_GUIDE

logger = logging.getLogger(__name__)

_MAX_DOC_CHARS = 16000        # per document
_MAX_TOTAL_DOC_CHARS = 40000  # across all documents in one turn
_MAX_IMAGES = 6

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".heif")


def _is_image(att) -> bool:
    if (att.mime_type or "").lower().startswith("image/"):
        return True
    return (att.filename or "").lower().endswith(_IMAGE_EXTS)


def _active_platforms(db: Session, owner_id: Optional[int]) -> set[str]:
    q = db.query(ApiKey.platform).filter(
        ApiKey.enabled == True, ApiKey.status != "error")  # noqa: E712
    if owner_id is not None:
        q = q.filter(or_(ApiKey.owner_id == owner_id, ApiKey.owner_id.is_(None)))
    return {row[0] for row in q.distinct().all()}


def _has_vision_model(db: Session, owner_id: Optional[int]) -> bool:
    """True if a vision-capable CHAT model exists for the user's keys. Applies
    the same _is_chatty filter as routing so image-generation models flagged
    supports_vision (Wan, Nano Banana, …) don't count."""
    active = _active_platforms(db, owner_id)
    if not active:
        return False
    rows = db.query(ChatModel).filter(
        ChatModel.enabled == True,            # noqa: E712
        ChatModel.supports_vision == True,    # noqa: E712
        ChatModel.platform.in_(active),
    ).all()
    return any(_is_chatty(m) for m in rows)


async def multimodal_stream_chat(db: Session, request: ChatRequest, owner_id: Optional[int] = None):
    """Stream a reply to a chat turn that has attachments."""
    attachments = request.attachments or []
    image_urls: list[str] = []
    img_names: list[str] = []
    doc_blocks: list[str] = []
    doc_names: list[str] = []
    doc_raws: list[tuple[str, bytes]] = []   # full bytes → per-conversation RAG
    total_doc = 0

    for att in attachments:
        try:
            raw = base64.b64decode(att.data, validate=False)
        except (binascii.Error, ValueError):
            logger.warning(f"[Multimodal] bad base64 for {att.filename}")
            continue

        if _is_image(att):
            if len(image_urls) >= _MAX_IMAGES:
                continue
            mime = att.mime_type or "image/jpeg"
            image_urls.append(f"data:{mime};base64,{att.data}")
            img_names.append(att.filename)
        else:
            try:
                text = clean_text(extract_text(att.filename, raw))
            except Exception as e:
                logger.warning(f"[Multimodal] extract failed for {att.filename}: {e}")
                doc_blocks.append(f"--- {att.filename} ---\n[Could not read this file: {e}]")
                doc_names.append(att.filename)
                continue
            if not text.strip():
                continue
            doc_raws.append((att.filename, raw))   # index full doc for later turns
            if total_doc >= _MAX_TOTAL_DOC_CHARS:
                continue
            text = text[:min(_MAX_DOC_CHARS, _MAX_TOTAL_DOC_CHARS - total_doc)]
            total_doc += len(text)
            doc_blocks.append(f"--- {att.filename} ---\n{text}")
            doc_names.append(att.filename)

    if image_urls and not _has_vision_model(db, owner_id):
        raise ValueError(
            "You attached an image, but no vision-capable model is available with "
            "your current keys. Add a provider that offers a vision model (e.g. "
            "OpenRouter, Google Gemini, or Mistral Pixtral) and try again."
        )

    # Get or create the conversation.
    conversation_id = request.conversation_id
    if conversation_id is None:
        title = (request.message[:50] or "New Chat") + ("..." if len(request.message) > 50 else "")
        conv = Conversation(title=title, owner_id=owner_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id

    # Persist the user turn with a compact attachment note so reloaded history
    # shows what was attached (the full extracted text is NOT stored).
    notes = []
    if img_names:
        notes.append("🖼 " + ", ".join(img_names))
    if doc_names:
        notes.append("📎 " + ", ".join(doc_names))
    stored = request.message
    if notes:
        stored = (request.message + "\n\n" + " · ".join(notes)).strip()
    db.add(Message(conversation_id=conversation_id, role="user", content=stored))
    db.commit()

    # Per-conversation RAG (A.3): index attached documents in the background so
    # they're retrievable on later turns in this conversation.
    for _fname, _raw in doc_raws:
        asyncio.create_task(
            ingest_conversation_document(conversation_id, owner_id, _fname, _raw))

    # Effective prompt: document text as grounding context before the question.
    prompt = request.message
    if doc_blocks:
        context = "\n\n".join(doc_blocks)
        prompt = (
            "The user attached the following document(s). Use their contents to "
            f"answer the question.\n\n{context}\n\n---\n\nUser question: "
            f"{request.message or '(describe / summarize the attached file)'}"
        )
    elif image_urls and not request.message.strip():
        prompt = "Describe the attached image(s)."

    messages: list[MessageDto] = [
        MessageDto(role="system",
                   content="You are a helpful assistant." + RESPONSE_FOLLOWUP_GUIDE),
    ]
    for m in (request.history or []):
        if m.role in ("user", "assistant"):
            messages.append(MessageDto(role=m.role, content=m.content))
    messages.append(MessageDto(role="user", content=prompt, images=image_urls or None))

    try:
        result = await route_stream_chat(
            db, messages,
            requested_model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            require_vision=bool(image_urls),
        )
    except Exception as e:
        if image_urls:
            # Reframe a routing failure for the image case: the user's only
            # vision models may be on an unavailable provider (e.g. Vercel needs
            # a card). Point them at a free vision provider.
            raise ValueError(
                "Couldn't reach a vision model to read your image. Your "
                "vision-capable models may be unavailable right now (for example "
                "Vercel AI Gateway requires a credit card). Add a provider that "
                "offers a free vision model — such as OpenRouter or Groq (Llama "
                f"Vision) — then try again.\n\n(technical detail: {e})"
            ) from e
        raise
    return conversation_id, result, ""
