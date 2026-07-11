from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, timedelta

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
            parent_id=c.parent_id,
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
        parent_id=conv.parent_id,
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
    _owned_or_404(db, conversation_id, account.id)
    # Cascade to descendant branches: delete this chat and every branch nested
    # under it (children, grandchildren, …). A branch the user wants to keep is
    # detached (promoted to top-level) first, so it's no longer a descendant and
    # survives this delete.
    to_delete = [conversation_id]
    frontier = [conversation_id]
    while frontier:
        cur = frontier.pop()
        for (kid,) in db.query(Conversation.id).filter(
                Conversation.parent_id == cur).all():
            to_delete.append(kid)
            frontier.append(kid)
    # Remove agent-feature rows (memory_chunks) that FK to these conversations,
    # else the DELETE violates the constraint and 500s.
    try:
        from ..models.db_models import MemoryChunk
        db.query(MemoryChunk).filter(
            MemoryChunk.conversation_id.in_(to_delete)
        ).delete(synchronize_session=False)
    except Exception:
        db.rollback()
    for c in db.query(Conversation).filter(Conversation.id.in_(to_delete)).all():
        db.delete(c)  # messages cascade via the ORM relationship
    db.commit()
    return {"success": True, "deleted": to_delete}


@router.post("/{conversation_id}/detach")
def detach(conversation_id: int, db: Session = Depends(get_db),
           account: Account = Depends(get_current_account)):
    """Promote a branch to a top-level chat by clearing its parent link. Used
    when the user unchecks a branch in the parent's delete dialog so it survives
    and becomes its own chat."""
    conv = _owned_or_404(db, conversation_id, account.id)
    conv.parent_id = None
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


class BranchBody(BaseModel):
    up_to: int  # copy messages [0 .. up_to] inclusive (oldest-first index)
    target_conversation_id: Optional[int] = None  # None → create a new chat


@router.post("/{conversation_id}/branch")
def branch(conversation_id: int, body: BranchBody, db: Session = Depends(get_db),
           account: Account = Depends(get_current_account)):
    """Branch a conversation: copy its history up to (and including) message
    index `up_to` into another chat — a brand-new one, or an existing target the
    user picked from the dropdown (the messages are appended there). Returns the
    target conversation's id so the client can switch to it."""
    src = _owned_or_404(db, conversation_id, account.id)
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    upto = messages[:max(0, body.up_to) + 1]
    if not upto:
        raise HTTPException(status_code=400, detail="Nothing to branch")

    if body.target_conversation_id is not None:
        target = _owned_or_404(db, body.target_conversation_id, account.id)
    else:
        target = Conversation(
            title=(f"Branch · {src.title}")[:120],
            owner_id=account.id,
            parent_id=conversation_id)
        db.add(target)
        db.flush()  # assign target.id

    # New timestamps (increasing) so the copies keep their order and land AFTER
    # any messages already in an existing target.
    base = datetime.now(timezone.utc)
    for i, m in enumerate(upto):
        db.add(Message(
            conversation_id=target.id,
            role=m.role,
            content=m.content,
            model_used=m.model_used,
            platform_used=m.platform_used,
            created_at=base + timedelta(milliseconds=i),
        ))
    target.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(target)
    return {"conversation_id": target.id, "title": target.title, "copied": len(upto)}


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
