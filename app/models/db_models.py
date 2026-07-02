from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..database import Base


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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_checked_at = Column(DateTime, nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
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
