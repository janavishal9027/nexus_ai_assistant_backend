from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class MessageDto(BaseModel):
    role: str
    content: str
    model: Optional[str] = None
    platform: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    history: Optional[list[MessageDto]] = None


class ChatResponse(BaseModel):
    conversation_id: Optional[int] = None
    content: str
    model: Optional[str] = None
    platform: Optional[str] = None
    fallback_attempts: int = 0


class StreamChunk(BaseModel):
    content: str = ""
    model: Optional[str] = None
    platform: Optional[str] = None
    done: bool = False
    error: Optional[str] = None


class ConversationDto(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: Optional[list[MessageDto]] = None

    class Config:
        from_attributes = True


class AddKeyRequest(BaseModel):
    platform: str
    key: str
    label: Optional[str] = ""
