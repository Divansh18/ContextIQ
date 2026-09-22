"""Environment-backed application settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration values loaded from defaults, `.env`, and the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CONTEXTIQ_",
        extra="ignore",
    )

    app_name: str = "ContextIQ"
    app_version: str = "0.1.0"
    preview_character_limit: int = Field(default=200, ge=1, le=2_000)


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the application process."""
    return Settings()

