"""Authentication utilities: password hashing + HS256 JWT + auth dependency.

Dependency-free by design:
- Passwords are hashed with stdlib PBKDF2-HMAC-SHA256 (per-user salt).
- Tokens are HS256 JWTs built with stdlib hmac/hashlib/base64 (no PyJWT).

This avoids native wheels (bcrypt/argon2/PyJWT) that can be blocked by OS
application-control policies.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..database import get_db

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 240_000
_PBKDF2_ALGO = "sha256"


# ─── Password hashing (PBKDF2-HMAC-SHA256) ──────────────────────────────────
def hash_password(password: str) -> str:
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(_PBKDF2_ALGO, password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_{_PBKDF2_ALGO}${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, hash_hex = stored.split("$")
        if not scheme.startswith("pbkdf2_"):
            return False
        algo = scheme.split("_", 1)[1]
        dk = hashlib.pbkdf2_hmac(algo, password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# ─── JWT (HS256) ────────────────────────────────────────────────────────────
def _jwt_secret() -> str:
    try:
        from ..config import get_settings
        secret = get_settings().jwt_secret
    except Exception:
        secret = os.environ.get("JWT_SECRET", "")
    return secret or "dev-insecure-jwt-secret-change-me"


def _jwt_ttl_seconds() -> int:
    try:
        from ..config import get_settings
        return int(get_settings().jwt_expires_hours) * 3600
    except Exception:
        return 720 * 3600


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def create_token(account_id: int, email: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"sub": str(account_id), "email": email, "iat": now, "exp": now + _jwt_ttl_seconds()}
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(payload).encode())}"
    sig = hmac.new(_jwt_secret().encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


def verify_token(token: str) -> Optional[dict]:
    try:
        header_seg, payload_seg, sig_seg = token.split(".")
        signing_input = f"{header_seg}.{payload_seg}"
        expected = hmac.new(_jwt_secret().encode(), signing_input.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(sig_seg), expected):
            return None
        payload = json.loads(_b64url_decode(payload_seg))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


# ─── Request helpers / FastAPI dependency ───────────────────────────────────
def extract_bearer(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return request.query_params.get("token")


def account_from_token(token: Optional[str], db: Session):
    from ..models.db_models import Account
    if not token:
        return None
    payload = verify_token(token)
    if not payload:
        return None
    try:
        account_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        return None
    return db.query(Account).filter(Account.id == account_id).first()


def get_current_account(request: Request, db: Session = Depends(get_db)):
    """FastAPI dependency — returns the authenticated Account or raises 401."""
    account = account_from_token(extract_bearer(request), db)
    if account is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return account
