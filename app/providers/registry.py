from .base import LlmProvider
from .openai_compat import OpenAICompatProvider
from .google import GoogleProvider


class ProviderRegistry:
    """Registry of all supported LLM providers."""

    def __init__(self):
        self._providers: dict[str, LlmProvider] = {}
        self._init_providers()

    def _init_providers(self):
        # OpenRouter - OpenAI-compatible with extra headers
        self._register(OpenAICompatProvider(
            platform_name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "http://localhost:8080",
                "X-Title": "ChatApp",
            },
        ))

        # Groq - OpenAI-compatible (ultra-fast inference)
        self._register(OpenAICompatProvider(
            platform_name="groq",
            base_url="https://api.groq.com/openai/v1",
        ))

        # NVIDIA NIM - OpenAI-compatible
        self._register(OpenAICompatProvider(
            platform_name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
        ))

        # HuggingFace Router - OpenAI-compatible
        self._register(OpenAICompatProvider(
            platform_name="huggingface",
            base_url="https://router.huggingface.co/v1",
        ))

        # Mistral AI - OpenAI-compatible
        self._register(OpenAICompatProvider(
            platform_name="mistral",
            base_url="https://api.mistral.ai/v1",
        ))

        # Cerebras - OpenAI-compatible (fast inference)
        self._register(OpenAICompatProvider(
            platform_name="cerebras",
            base_url="https://api.cerebras.ai/v1",
        ))

        # SambaNova - OpenAI-compatible
        self._register(OpenAICompatProvider(
            platform_name="sambanova",
            base_url="https://api.sambanova.ai/v1",
        ))

        # Vercel AI Gateway - OpenAI-compatible
        self._register(OpenAICompatProvider(
            platform_name="vercel",
            base_url="https://ai-gateway.vercel.sh/v1",
        ))

        # Z.ai (GLM) - OpenAI-compatible
        self._register(OpenAICompatProvider(
            platform_name="zai",
            base_url="https://api.z.ai/api/paas/v4",
        ))

        # Google AI Studio - Gemini API
        self._register(GoogleProvider())

    def _register(self, provider: LlmProvider):
        self._providers[provider.platform] = provider

    def get(self, platform: str) -> LlmProvider | None:
        return self._providers.get(platform)

    def has(self, platform: str) -> bool:
        return platform in self._providers

    def all_platforms(self) -> list[str]:
        return list(self._providers.keys())


# Singleton
provider_registry = ProviderRegistry()
