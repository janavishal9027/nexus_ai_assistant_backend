import httpx
import json
import logging
from typing import AsyncGenerator
from .base import LlmProvider
from ..models.schemas import MessageDto

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LlmProvider):
    """
    Generic provider for platforms with OpenAI-compatible APIs:
    OpenRouter, Groq, NVIDIA NIM, HuggingFace Router.
    """

    def __init__(self, platform_name: str, base_url: str, extra_headers: dict | None = None):
        self._platform = platform_name
        self._base_url = base_url
        self._extra_headers = extra_headers or {}

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def base_url(self) -> str:
        return self._base_url

    async def chat_completion(
        self,
        api_key: str,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body = self._build_body(messages, model_id, temperature, max_tokens, stream=False)
        headers = self._build_headers(api_key)

        logger.info(f"[{self._platform}] POST /chat/completions model={model_id}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers=headers,
            )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                raise RuntimeError(
                    f"{self._platform} API error {resp.status_code}: {error_text}"
                )

            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if not content:
                    content = message.get("reasoning_content", "")
                if content:
                    return content

            raise RuntimeError(f"{self._platform} returned no content")

    async def stream_chat_completion(
        self,
        api_key: str,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        body = self._build_body(messages, model_id, temperature, max_tokens, stream=True)
        headers = self._build_headers(api_key)

        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=body,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    raise RuntimeError(
                        f"{self._platform} stream error {resp.status_code}: {error.decode()[:300]}"
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    async def stream_chat_completion_ex(
        self,
        api_key: str,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Streaming with completion metadata: yields content deltas and a final
        finish event carrying finish_reason ("length" = truncated, "stop" = done)."""
        body = self._build_body(messages, model_id, temperature, max_tokens, stream=True)
        headers = self._build_headers(api_key)
        finish_reason: str | None = None

        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=body,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    raise RuntimeError(
                        f"{self._platform} stream error {resp.status_code}: {error.decode()[:300]}"
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield {"type": "content", "text": content}

        yield {"type": "finish", "reason": finish_reason}

    async def validate_key(self, api_key: str) -> bool:
        headers = self._build_headers(api_key)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"{self._base_url}/models", headers=headers)
                return resp.status_code not in (401, 403)
        except Exception:
            return False

    def _build_headers(self, api_key: str) -> dict:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        return headers

    def _build_body(
        self,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict:
        body: dict = {
            "model": model_id,
            "stream": stream,
            "messages": [{"role": self._api_role(m.role), "content": m.content} for m in messages],
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens and max_tokens > 0:
            body["max_tokens"] = max_tokens
        return body

    @staticmethod
    def _api_role(role: str) -> str:
        """Map an internal message role to an OpenAI-compatible API role.

        OpenAI-compatible APIs only accept ``system``/``user``/``assistant`` unless
        the full native tool-calling protocol is used (an assistant message with
        ``tool_calls`` followed by ``tool`` messages carrying ``tool_call_id``).
        This app represents tool output as plain context messages via the
        DecisionEngine (no ``tool_call_id``), so any non-standard role — notably
        ``tool`` — is mapped to ``user`` to satisfy strict providers like Groq,
        which otherwise reject a ``role:"tool"`` message that lacks ``tool_call_id``.
        """
        return role if role in ("system", "user", "assistant") else "user"
