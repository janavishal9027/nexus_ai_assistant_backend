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

    async def stream_chat_completion_ex(
        self,
        api_key: str,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Streaming variant that also reports why generation stopped.

        Yields ``{"type": "content", "text": <str>}`` for each delta and a final
        ``{"type": "finish", "reason": <str|None>}`` where reason is ``"length"``
        (hit the token cap → truncated), ``"stop"`` (finished), or None (unknown).

        Default wraps ``stream_chat_completion`` and reports no reason; providers
        that expose finish_reason (OpenAI-compatible) override this so the Deep
        Research orchestrator can auto-continue across models."""
        async for chunk in self.stream_chat_completion(
            api_key=api_key,
            messages=messages,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield {"type": "content", "text": chunk}
        yield {"type": "finish", "reason": None}

    @abstractmethod
    async def validate_key(self, api_key: str) -> bool:
        """Check if the API key is valid."""
        ...
