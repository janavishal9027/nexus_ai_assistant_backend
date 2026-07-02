import httpx
import json
import logging
from typing import AsyncGenerator
from .base import LlmProvider
from ..models.schemas import MessageDto

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleProvider(LlmProvider):
    """Google AI Studio provider (Gemini API)."""

    @property
    def platform(self) -> str:
        return "google"

    @property
    def base_url(self) -> str:
        return BASE_URL

    async def chat_completion(
        self,
        api_key: str,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body = self._build_gemini_body(messages, temperature, max_tokens)
        url = f"{BASE_URL}/models/{model_id}:generateContent?key={api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body)

            if resp.status_code != 200:
                raise RuntimeError(f"Google API error {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)
                if text:
                    return text

            raise RuntimeError("Google API returned no content")

    async def stream_chat_completion(
        self,
        api_key: str,
        messages: list[MessageDto],
        model_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        body = self._build_gemini_body(messages, temperature, max_tokens)
        url = f"{BASE_URL}/models/{model_id}:streamGenerateContent?alt=sse&key={api_key}"

        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    raise RuntimeError(f"Google stream error {resp.status_code}: {error.decode()[:300]}")

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts)
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue

    async def validate_key(self, api_key: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{BASE_URL}/models?key={api_key}")
                return resp.status_code not in (401, 403)
        except Exception:
            return False

    def _build_gemini_body(
        self, messages: list[MessageDto], temperature: float | None, max_tokens: int | None
    ) -> dict:
        system_parts = []
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        body: dict = {"contents": contents}

        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        generation_config: dict = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens and max_tokens > 0:
            generation_config["maxOutputTokens"] = max_tokens
        if generation_config:
            body["generationConfig"] = generation_config

        return body
