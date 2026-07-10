from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from ..database import get_db
from ..models.db_models import Conversation, Message, ChatModel, Account
from ..models.schemas import ConversationDto, MessageDto
from ..services.auth import get_current_account

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _owned_or_404(db: Session, conversation_id: int, account_id: int) -> Conversation:
    """Fetch a conversation only if it belongs to the account (else 404, so
    existence isn't leaked across accounts)."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None or conv.owner_id != account_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/")
def get_all(db: Session = Depends(get_db), account: Account = Depends(get_current_account)):
    conversations = (
        db.query(Conversation)
        .filter(Conversation.owner_id == account.id)
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
def get_by_id(conversation_id: int, db: Session = Depends(get_db),
              account: Account = Depends(get_current_account)):
    conv = _owned_or_404(db, conversation_id, account.id)

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
def delete(conversation_id: int, db: Session = Depends(get_db),
           account: Account = Depends(get_current_account)):
    conv = _owned_or_404(db, conversation_id, account.id)
    if conv:
        # Remove rows added by the agent feature that reference this conversation
        # via a foreign key (memory_chunks). Without this the DELETE violates the
        # FK constraint and fails with a 500, and the row reappears on next load.
        try:
            from ..models.db_models import MemoryChunk
            db.query(MemoryChunk).filter(
                MemoryChunk.conversation_id == conversation_id
            ).delete(synchronize_session=False)
        except Exception:
            db.rollback()
        db.delete(conv)          # messages cascade via the ORM relationship
        db.commit()
    return {"success": True}


class TruncateBody(BaseModel):
    keep: int  # keep the first N messages (oldest first); delete the rest


@router.post("/{conversation_id}/truncate")
def truncate(conversation_id: int, body: TruncateBody, db: Session = Depends(get_db),
             account: Account = Depends(get_current_account)):
    """Delete all messages in a conversation beyond the first `keep` (ordered
    oldest-first). Used when a user edits an earlier message: the old message and
    everything after it are removed so the regenerated turn — and future reloads
    — reflect the edit instead of duplicating history."""
    _owned_or_404(db, conversation_id, account.id)
    keep = max(0, body.keep)
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    to_delete = messages[keep:]
    for m in to_delete:
        db.delete(m)
    if to_delete:
        db.commit()
    return {"success": True, "deleted": len(to_delete)}


class UpdateTitle(BaseModel):
    title: Optional[str] = None


@router.patch("/{conversation_id}")
def update_title(conversation_id: int, body: UpdateTitle, db: Session = Depends(get_db),
                 account: Account = Depends(get_current_account)):
    conv = _owned_or_404(db, conversation_id, account.id)
    if body.title:
        conv.title = body.title
        db.commit()
    return {"success": True}
