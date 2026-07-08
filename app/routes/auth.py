"""Authentication routes — signup / login / me."""
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import Account, Conversation, Message, MemoryChunk, ApiKey
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


@router.delete("/me")
def delete_account(db: Session = Depends(get_db),
                   account: Account = Depends(auth_service.get_current_account)):
    """Permanently delete the account and all of its data (conversations, their
    messages + memory chunks, and the account's private provider keys)."""
    acc_id = account.id
    conv_ids = [cid for (cid,) in db.query(Conversation.id).filter(Conversation.owner_id == acc_id).all()]
    if conv_ids:
        db.query(MemoryChunk).filter(MemoryChunk.conversation_id.in_(conv_ids)).delete(synchronize_session=False)
        db.query(Message).filter(Message.conversation_id.in_(conv_ids)).delete(synchronize_session=False)
        db.query(Conversation).filter(Conversation.id.in_(conv_ids)).delete(synchronize_session=False)
    db.query(ApiKey).filter(ApiKey.owner_id == acc_id).delete(synchronize_session=False)
    db.delete(account)
    db.commit()
    logger.info(f"[Auth] Account deleted id={acc_id} ({len(conv_ids)} conversations removed)")
    return {"success": True}
