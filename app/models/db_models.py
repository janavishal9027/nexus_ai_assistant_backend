from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..database import Base

# pgvector is an optional dependency. When it (or its native extension) is
# unavailable the app must still import and boot with memory features disabled
# (req 15.8). We fall back to a JSON column so table metadata stays valid.
try:
    from pgvector.sqlalchemy import Vector as _Vector
    HAS_PGVECTOR = True

    def _embedding_column(dim: int):
        return Column(_Vector(dim), nullable=False)
except Exception:  # pragma: no cover - exercised only when pgvector is absent
    HAS_PGVECTOR = False

    def _embedding_column(dim: int):
        # Fallback: store the raw vector as JSON. Cosine search is unavailable
        # in this mode; memory features require a real pgvector deployment.
        return Column(JSON, nullable=False)


# Embedding dimension for text-embedding-3-small. Must match the Vector(dim)
# column and the embedding model configured via EMBEDDING_MODEL (req 15.5).
EMBEDDING_DIM = 1536


class ChatModel(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, nullable=False, index=True)
    model_id = Column(String, nullable=False)
    display_name = Column(String)
    intelligence_rank = Column(Integer, default=50)
    speed_rank = Column(Integer, default=50)
    size_label = Column(String)  # Frontier, Large, Medium, Small
    rpm_limit = Column(Integer, nullable=True)
    rpd_limit = Column(Integer, nullable=True)
    tpm_limit = Column(Integer, nullable=True)
    tpd_limit = Column(Integer, nullable=True)
    context_window = Column(Integer, nullable=True)
    enabled = Column(Boolean, default=True)
    supports_vision = Column(Boolean, default=False)
    supports_tools = Column(Boolean, default=False)
    priority = Column(Integer, default=100)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, nullable=False, index=True)
    api_key = Column(String, nullable=False)
    label = Column(String, default="")
    enabled = Column(Boolean, default=True)
    status = Column(String, default="unknown")  # healthy, error, unknown
    # Owning account. NULL = shared/global key usable by everyone (e.g. the
    # seeded free-tier keys); a set value = a user's private key. Added to
    # existing databases via ALTER TABLE at startup.
    owner_id = Column(Integer, index=True, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_checked_at = Column(DateTime, nullable=True)


class Account(Base):
    """Authenticated end-user account (login/signup). Distinct from the
    password-less agent `User` business entity."""
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    # Owning account. Nullable so pre-auth/legacy rows remain valid; the column
    # is added to existing databases via ALTER TABLE at startup.
    owner_id = Column(Integer, index=True, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan",
                            order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    model_used = Column(String, nullable=True)
    platform_used = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")


# ─── Full-Stack Agent Orchestration models (additive; new tables only) ──────

class User(Base):
    """Business user entity served by the User_Service_Tool.

    NOTE: password / password_hash / api_key fields are intentionally absent
    and must never be added here (req 5.9 / 19.4).
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(64), default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    tasks = relationship("Task", back_populates="assignee")
    fcm_tokens = relationship("FCMToken", back_populates="user")


class Task(Base):
    """Task entity served by the Task_Service_Tool."""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(64), default="pending", index=True)
    assignee_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    due_date = Column(DateTime, nullable=True)
    priority = Column(String(32), default="medium")
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    assignee = relationship("User", back_populates="tasks")


class MemoryChunk(Base):
    """Semantic-memory chunk: raw text plus its pgvector embedding (req 8)."""
    __tablename__ = "memory_chunks"

    id = Column(Integer, primary_key=True, index=True)
    # ON DELETE CASCADE so deleting a conversation cleans up its memory chunks
    # (applies to freshly created databases; existing DBs rely on the explicit
    # cleanup in the delete-conversation route).
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"),
                             nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    text = Column(Text, nullable=False)
    embedding = _embedding_column(EMBEDDING_DIM)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # IVFFlat cosine index created by the Alembic migration (req 8.1).


class FCMToken(Base):
    """FCM device token registered for push notifications (req 11.4)."""
    __tablename__ = "fcm_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    device_token = Column(String(512), nullable=False, unique=True)
    device_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="fcm_tokens")


class AuditLog(Base):
    """Audit trail for privileged write operations (req 19.7)."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    correlation_id = Column(String(36), nullable=False, index=True)
    tool_name = Column(String(128), nullable=False)
    acting_user_id = Column(Integer, nullable=True)
    target_resource = Column(String(255), nullable=True)
    outcome = Column(String(32), nullable=False)  # success | error
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
