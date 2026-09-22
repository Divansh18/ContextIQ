"""Unit tests for OpenAI embedding generation."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from openai import OpenAIError

from app.rag.embeddings import (
    EmbeddingAPIError,
    MissingOpenAIAPIKeyError,
    UnexpectedEmbeddingDimensionError,
    UnexpectedEmbeddingResponseError,
    embed_text,
    embed_texts,
)


def _embedding_item(index: int, values: list[float]) -> SimpleNamespace:
    return SimpleNamespace(index=index, embedding=values)


@patch("app.rag.embeddings.OpenAI")
def test_embed_text_generates_one_embedding(openai_class: Mock) -> None:
    client = openai_class.return_value
    client.embeddings.create.return_value = SimpleNamespace(
        data=[_embedding_item(0, [0.1, 0.2, 0.3])]
    )

    embedding = embed_text(
        "ContextIQ",
        api_key="test-key",
        model="text-embedding-3-small",
        expected_dimensions=3,
    )

    assert embedding == [0.1, 0.2, 0.3]
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["ContextIQ"],
        dimensions=3,
        encoding_format="float",
    )
    client.close.assert_called_once_with()


@patch("app.rag.embeddings.OpenAI")
def test_embed_texts_batches_inputs_in_one_request(openai_class: Mock) -> None:
    client = openai_class.return_value
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            _embedding_item(1, [0.4, 0.5]),
            _embedding_item(0, [0.1, 0.2]),
        ]
    )

    embeddings = embed_texts(
        ["first chunk", "second chunk"],
        api_key="test-key",
        model="text-embedding-3-small",
        expected_dimensions=2,
    )

    assert embeddings == [[0.1, 0.2], [0.4, 0.5]]
    client.embeddings.create.assert_called_once()
    assert client.embeddings.create.call_args.kwargs["input"] == [
        "first chunk",
        "second chunk",
    ]


@patch("app.rag.embeddings.OpenAI")
def test_embedding_dimension_mismatch_is_rejected(openai_class: Mock) -> None:
    client = openai_class.return_value
    client.embeddings.create.return_value = SimpleNamespace(
        data=[_embedding_item(0, [0.1, 0.2])]
    )

    with pytest.raises(UnexpectedEmbeddingDimensionError, match="2 dimensions"):
        embed_text(
            "ContextIQ",
            api_key="test-key",
            model="text-embedding-3-small",
            expected_dimensions=3,
        )


@patch("app.rag.embeddings.OpenAI")
def test_missing_api_key_fails_before_client_creation(openai_class: Mock) -> None:
    with pytest.raises(MissingOpenAIAPIKeyError, match="OPENAI_API_KEY"):
        embed_text(
            "ContextIQ",
            api_key=None,
            model="text-embedding-3-small",
            expected_dimensions=1_536,
        )

    openai_class.assert_not_called()


@patch("app.rag.embeddings.OpenAI")
def test_openai_api_failure_is_wrapped(openai_class: Mock) -> None:
    client = openai_class.return_value
    client.embeddings.create.side_effect = OpenAIError("request failed")

    with pytest.raises(EmbeddingAPIError, match="generation failed"):
        embed_texts(
            ["first", "second"],
            api_key="test-key",
            model="text-embedding-3-small",
            expected_dimensions=2,
        )

    client.close.assert_called_once_with()


@patch("app.rag.embeddings.OpenAI")
def test_missing_embedding_result_is_rejected(openai_class: Mock) -> None:
    client = openai_class.return_value
    client.embeddings.create.return_value = SimpleNamespace(
        data=[_embedding_item(0, [0.1, 0.2])]
    )

    with pytest.raises(
        UnexpectedEmbeddingResponseError,
        match="unexpected number or ordering",
    ):
        embed_texts(
            ["first", "second"],
            api_key="test-key",
            model="text-embedding-3-small",
            expected_dimensions=2,
        )


@patch("app.rag.embeddings.OpenAI")
def test_empty_batch_avoids_api_request(openai_class: Mock) -> None:
    assert (
        embed_texts(
            [],
            api_key=None,
            model="text-embedding-3-small",
            expected_dimensions=1_536,
        )
        == []
    )
    openai_class.assert_not_called()
