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
    # Deep Research / adaptive mode: route ONLY to large (>=400B parameter)
    # models and always gather live web context for a thorough, cited answer.
    deep_research: bool = False
    # Web Search mode: force a live web search for this turn (bypasses the
    # needs_web_search heuristic) so the answer is grounded in fresh results.
    web_search: bool = False


class ChatResponse(BaseModel):
    conversation_id: Optional[int] = None
    content: str
    model: Optional[str] = None
    platform: Optional[str] = None
    fallback_attempts: int = 0
    tool_calls_made: int = 0
    tool_rounds: int = 0


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
    parent_id: Optional[int] = None
    messages: Optional[list[MessageDto]] = None

    class Config:
        from_attributes = True


class AddKeyRequest(BaseModel):
    platform: str
    key: str
    label: Optional[str] = ""


# ─── Authentication schemas ─────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AccountDto(BaseModel):
    id: int
    email: str
    name: Optional[str] = None

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    token: str
    account: AccountDto


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


# ─── Full-Stack Agent Orchestration schemas (additive) ──────────────────────
from typing import Literal


class AgentChatRequest(BaseModel):
    """Chat request for the Agent_Gateway. Extends ChatRequest with agent fields."""
    message: str
    conversation_id: Optional[int] = None
    session_id: Optional[str] = None          # links an HTTP request to a WebSocket session
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    requires_fresh_data: bool = False         # bypasses Redis reads when True (req 20)
    history: Optional[list[MessageDto]] = None


class AgentChatResponse(BaseModel):
    conversation_id: Optional[int] = None
    content: str
    model: Optional[str] = None
    platform: Optional[str] = None
    correlation_id: str
    planner_used: bool = False
    subtask_count: int = 0
    tool_calls_made: int = 0
    requires_fresh_data: bool = False


class SubtaskStatus(BaseModel):
    index: int
    description: str
    status: Literal["completed", "skipped", "failed", "pending"]
    output: Optional[dict] = None
    failure_reason: Optional[str] = None


class PlanSummary(BaseModel):
    subtask_count: int
    subtasks: list[SubtaskStatus]


class HealthComponent(BaseModel):
    status: Literal["healthy", "degraded", "timeout", "disabled"]
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    components: dict[str, HealthComponent]


# User / Task DTOs — deliberately contain NO password / api_key fields (req 5.9)
class UserDto(BaseModel):
    id: int
    name: str
    email: str
    role: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TaskDto(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    status: str
    assignee_id: Optional[int] = None
    due_date: Optional[datetime] = None
    priority: str
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PaginatedUsers(BaseModel):
    items: list[UserDto]
    page: int
    page_size: int
    total_count: int


class PaginatedTasks(BaseModel):
    items: list[TaskDto]
    page_size: int
    total_count: int
    next_page_token: Optional[str] = None
