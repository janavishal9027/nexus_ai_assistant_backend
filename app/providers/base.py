from abc import ABC, abstractmethod
from typing import AsyncGenerator
from ..models.schemas import MessageDto


class LlmProvider(ABC):
    """Base class for all LLM providers."""

    @property
    @abstractmethod
    def platform(self) -> str:
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        ...

    @abstractmethod
    async def chat_completion(
        self,
        api_key: str,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the full response text."""
        ...

    @abstractmethod
    async def stream_chat_completion(
        self,
        api_key: str,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Send a streaming chat completion and yield text chunks."""
        ...

    @abstractmethod
    async def validate_key(self, api_key: str) -> bool:
        """Check if the API key is valid."""
        ...
