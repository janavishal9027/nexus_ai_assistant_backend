"""Authentication routes — signup / login / me."""
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import (
    Account, ApiKey, AppearancePrefs, Conversation, KgEdge, MemoryChunk,
    MemoryEdge, MemoryPrefs, Message, MessageFeedback, Project,
    ProjectBrainEntry, Skill,
)
from ..models.rag_models import (Document, DocumentChunk, IngestionJob,
                                 KnowledgeBase)
from ..models.schemas import (
    SignupRequest, LoginRequest, AuthResponse, AccountDto, ChangePasswordRequest,
    UpdateProfileRequest,
)
from ..services import auth as auth_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


@router.post("/signup", response_model=AuthResponse)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    email = _normalize_email(body.email)
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Please enter a valid email address")
    if not body.password or len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters")
    if db.query(Account).filter(Account.email == email).first() is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    account = Account(
        email=email,
        name=(body.name or "").strip() or None,
        password_hash=auth_service.hash_password(body.password),
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    logger.info(f"[Auth] New account created: id={account.id} email={email}")
    token = auth_service.create_token(account.id, account.email)
    return AuthResponse(token=token, account=AccountDto.model_validate(account))


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    email = _normalize_email(body.email)
    account = db.query(Account).filter(Account.email == email).first()
    # Same error for unknown email and wrong password (avoids account enumeration).
    if account is None or not auth_service.verify_password(body.password, account.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = auth_service.create_token(account.id, account.email)
    return AuthResponse(token=token, account=AccountDto.model_validate(account))


@router.get("/me", response_model=AccountDto)
def me(account: Account = Depends(auth_service.get_current_account)):
    return AccountDto.model_validate(account)


@router.patch("/me", response_model=AccountDto)
def update_profile(body: UpdateProfileRequest, db: Session = Depends(get_db),
                   account: Account = Depends(auth_service.get_current_account)):
    """Update the current account's name and/or email."""
    if body.email is not None:
        email = _normalize_email(body.email)
        if not _EMAIL_RE.match(email):
            raise HTTPException(status_code=422, detail="Please enter a valid email address")
        clash = (
            db.query(Account)
            .filter(Account.email == email, Account.id != account.id)
            .first()
        )
        if clash is not None:
            raise HTTPException(status_code=409, detail="That email is already in use")
        account.email = email
    if body.name is not None:
        account.name = body.name.strip() or None
    db.commit()
    db.refresh(account)
    logger.info(f"[Auth] Profile updated for account id={account.id}")
    return AccountDto.model_validate(account)


@router.post("/change-password")
def change_password(body: ChangePasswordRequest, db: Session = Depends(get_db),
                    account: Account = Depends(auth_service.get_current_account)):
    if not auth_service.verify_password(body.current_password, account.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    if not body.new_password or len(body.new_password) < 6:
        raise HTTPException(status_code=422, detail="New password must be at least 6 characters")
    account.password_hash = auth_service.hash_password(body.new_password)
    db.commit()
    logger.info(f"[Auth] Password changed for account id={account.id}")
    return {"success": True}


# Every owner-scoped table, as (model, label). Account deletion must clear ALL
# of them: account ids are sequential, so anything left behind would be
# inherited by the next account to take this id. Add new owner-scoped tables
# here — leaving one out silently leaks a deleted user's data into a new one.
# tests/integration/test_account_deletion.py asserts this list covers every
# model with an owner_id, so a new table can't quietly go missing.
# Ordered children-before-parents (documents FK to knowledge_bases, chunks and
# jobs to documents) so a delete can't trip a foreign key.
_OWNER_SCOPED = [
    (MemoryChunk, "episodic"),          # raw Q&A log (+ embeddings)
    (Skill, "skills"),                  # distilled preferences/lessons
    (MessageFeedback, "feedback"),      # 👍/👎 ratings
    (MemoryEdge, "memory_graph"),       # personal people/orgs + tools/tech graph
    (KgEdge, "knowledge_graph"),        # content entity/relation facts
    (ProjectBrainEntry, "project_brain"),
    (Project, "projects"),
    (MemoryPrefs, "memory_prefs"),
    (AppearancePrefs, "appearance_prefs"),
    (ApiKey, "api_keys"),
    # RAG: the user's uploaded documents and everything derived from them.
    (DocumentChunk, "document_chunks"),  # text + embeddings of uploads
    (IngestionJob, "ingestion_jobs"),
    (Document, "documents"),
    (KnowledgeBase, "knowledge_bases"),
]


@router.delete("/me")
def delete_account(db: Session = Depends(get_db),
                   account: Account = Depends(auth_service.get_current_account)):
    """Permanently delete the account and every row it owns — conversations and
    their messages, all memory layers (episodic, skills, feedback, personal
    graph, content knowledge graph, project brains), projects, memory
    preferences, and private provider keys.

    Deletes are owner-scoped and run in ONE transaction, so this either erases
    everything or nothing.
    """
    acc_id = account.id
    removed: dict[str, int] = {}

    # Conversations + their messages. Messages have no owner_id, so they're
    # reached through the conversation ids.
    conv_ids = [cid for (cid,) in
                db.query(Conversation.id).filter(Conversation.owner_id == acc_id).all()]
    if conv_ids:
        removed["messages"] = db.query(Message).filter(
            Message.conversation_id.in_(conv_ids)).delete(synchronize_session=False)
        removed["conversations"] = db.query(Conversation).filter(
            Conversation.id.in_(conv_ids)).delete(synchronize_session=False)

    # Everything else is owner-scoped. NB: memory chunks are deleted by
    # owner_id, not conversation_id — a chunk whose conversation was already
    # deleted would otherwise survive its owner.
    for model, label in _OWNER_SCOPED:
        removed[label] = db.query(model).filter(
            model.owner_id == acc_id).delete(synchronize_session=False)

    db.delete(account)
    db.commit()
    logger.info(f"[Auth] Account deleted id={acc_id} removed="
                f"{ {k: v for k, v in removed.items() if v} }")
    return {"success": True}
