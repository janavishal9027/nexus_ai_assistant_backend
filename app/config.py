from pydantic_settings import BaseSettings
from functools import lru_cache
from urllib.parse import quote_plus


class Settings(BaseSettings):
    # PostgreSQL config
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "chatapp"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # Provider API keys (can also be added dynamically via API)
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    nvidia_api_key: str = ""
    huggingface_api_key: str = ""
    google_api_key: str = ""

    # Real-time web search. Optional: with no key, a keyless DuckDuckGo
    # fallback is used. A free Tavily key gives cleaner, LLM-optimized results.
    tavily_api_key: str = ""

    # Fallback config
    fallback_max_retries: int = 10
    cooldown_duration_ms: int = 90000

    # ─── Agent Orchestration Infrastructure (full-stack-agent-orchestration) ───
    # Redis (live session state + fast current-state cache). Req 15.1
    redis_url: str = "redis://localhost:6379"
    # Kafka (agent event streaming + inter-service commands). Req 15.2
    kafka_bootstrap_servers: str = "localhost:9092"
    # Firebase Cloud Messaging service-account credentials path. Req 15.4
    fcm_credentials_path: str = ""
    # Embedding model for pgvector semantic memory. Req 15.5
    embedding_model: str = "text-embedding-3-small"
    # Cosine similarity cut-off for memory_search. Req 8.9 / 15
    memory_similarity_threshold: float = 0.7
    # Bounded per-topic ring-buffer size for the Real-Time Events tool. Req 18.4
    realtime_event_buffer_size: int = 500
    # Comma-separated agent feature flags: planner,redis_cache,kafka,fcm,websocket. Req 15.7
    agent_features: str = ""

    # ─── Authentication ─────────────────────────────────────────────────────
    # HS256 signing secret for JWTs. MUST be overridden in production via env.
    jwt_secret: str = "dev-insecure-jwt-secret-change-me"
    # Token lifetime in hours (default 30 days).
    jwt_expires_hours: int = 720

    @property
    def database_url(self) -> str:
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
