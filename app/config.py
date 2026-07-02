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

    # Fallback config
    fallback_max_retries: int = 10
    cooldown_duration_ms: int = 90000

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
