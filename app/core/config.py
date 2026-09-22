"""Environment-backed application settings."""

from functools import lru_cache
from typing import Self

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEARCH_TOP_K = 3
MAX_SEARCH_TOP_K = 100
DEFAULT_SEARCH_NUM_CANDIDATES_MULTIPLIER = 10
DEFAULT_GENERATION_MODEL = "gpt-5.6-luna"
DEFAULT_GENERATION_MAX_OUTPUT_TOKENS = 300
MAX_ELASTICSEARCH_NUM_CANDIDATES = 10_000
MAX_ELASTICSEARCH_EMBEDDING_DIMENSIONS = 4_096


def validate_chunk_configuration(*, chunk_size: int, chunk_overlap: int) -> None:
    """Reject sizes that cannot produce forward-moving chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


class Settings(BaseSettings):
    """Configuration values loaded from defaults, `.env`, and the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CONTEXTIQ_",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "ContextIQ"
    app_version: str = "0.1.0"
    preview_character_limit: int = Field(default=200, ge=1, le=2_000)
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = Field(
        default="contextiq-document-chunks",
        min_length=1,
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "CONTEXTIQ_OPENAI_API_KEY",
        ),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        validation_alias=AliasChoices(
            "EMBEDDING_MODEL",
            "CONTEXTIQ_EMBEDDING_MODEL",
        ),
    )
    embedding_dimensions: int = Field(
        default=1_536,
        ge=1,
        le=MAX_ELASTICSEARCH_EMBEDDING_DIMENSIONS,
        validation_alias=AliasChoices(
            "EMBEDDING_DIMENSIONS",
            "CONTEXTIQ_EMBEDDING_DIMENSIONS",
        ),
    )
    search_default_top_k: int = Field(
        default=DEFAULT_SEARCH_TOP_K,
        ge=1,
        le=MAX_SEARCH_TOP_K,
    )
    search_num_candidates_multiplier: int = Field(
        default=DEFAULT_SEARCH_NUM_CANDIDATES_MULTIPLIER,
        ge=2,
        le=100,
    )
    generation_model: str = Field(
        default=DEFAULT_GENERATION_MODEL,
        min_length=1,
        validation_alias=AliasChoices(
            "GENERATION_MODEL",
            "CONTEXTIQ_GENERATION_MODEL",
        ),
    )
    generation_max_output_tokens: int = Field(
        default=DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
        ge=1,
        le=4_000,
    )

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> Self:
        """Ensure each chunk advances through the source text."""
        validate_chunk_configuration(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the application process."""
    return Settings()
