"""Tests for environment-backed application settings."""

import pytest

from app.core.config import Settings


def test_embedding_settings_use_requested_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1_536
    assert settings.search_default_top_k == 3
    assert settings.search_num_candidates_multiplier == 10
    assert settings.generation_model == "gpt-5.6-luna"
    assert settings.generation_max_output_tokens == 300


def test_openai_api_key_uses_standard_environment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(settings)
