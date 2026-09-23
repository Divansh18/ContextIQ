"""Tests for environment-backed application settings."""

import pytest

from app.core.config import Settings


def test_embedding_settings_use_requested_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.frontend_origin == "http://localhost:3000"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1_536
    assert settings.search_default_top_k == 3
    assert settings.search_num_candidates_multiplier == 10
    assert settings.generation_model == "gpt-5.6-luna"
    assert settings.generation_max_output_tokens == 300
    assert settings.rag_relevance_threshold == pytest.approx(0.60)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_relevance_threshold_must_be_a_normalized_score(
    threshold: float,
) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, rag_relevance_threshold=threshold)


def test_openai_api_key_uses_standard_environment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(settings)


def test_frontend_origin_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXTIQ_FRONTEND_ORIGIN", "http://127.0.0.1:3001")

    settings = Settings(_env_file=None)

    assert settings.frontend_origin == "http://127.0.0.1:3001"
