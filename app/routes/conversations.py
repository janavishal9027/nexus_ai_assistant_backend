from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from ..database import get_db
from ..models.db_models import Conversation, Message, ChatModel
from ..models.schemas import ConversationDto, MessageDto

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("/")
def get_all(db: Session = Depends(get_db)):
    conversations = (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return [
        ConversationDto(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in conversations
    ]


@router.get("/{conversation_id}")
def get_by_id(conversation_id: int, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    # Resolve stored model ids to their friendly display names so the reloaded
    # per-message badge matches what was shown live (e.g. "Z.ai: GLM 5.2").
    name_by_id = {
        mid: dname
        for mid, dname in db.query(ChatModel.model_id, ChatModel.display_name).all()
    }

    return ConversationDto(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageDto(
                role=m.role,
                content=m.content,
                model=(name_by_id.get(m.model_used, m.model_used)
                       if m.model_used else None),
                platform=m.platform_used,
            )
            for m in messages
        ],
    )


@router.delete("/{conversation_id}")
def delete(conversation_id: int, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv:
        db.delete(conv)
        db.commit()
    return {"success": True}


class UpdateTitle(BaseModel):
    title: Optional[str] = None


@router.patch("/{conversation_id}")
def update_title(conversation_id: int, body: UpdateTitle, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if body.title:
        conv.title = body.title
        db.commit()
    return {"success": True}
