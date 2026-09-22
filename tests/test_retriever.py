"""Unit tests for semantic retrieval orchestration."""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.rag.embeddings import EmbeddingAPIError
from app.rag.retriever import retrieve_semantic_chunks
from app.schemas.search import SemanticSearchResult

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=SecretStr("test-key"),
        embedding_dimensions=3,
        search_default_top_k=3,
        search_num_candidates_multiplier=10,
    )


def _result() -> SemanticSearchResult:
    return SemanticSearchResult(
        document_id=DOCUMENT_ID,
        filename="knowledge.pdf",
        chunk_index=0,
        text="Containers can run without managing virtual machines.",
        character_count=52,
        score=0.91,
    )


@patch("app.rag.retriever.search_similar_chunks")
@patch("app.rag.retriever.embed_text")
def test_retriever_embeds_query_and_uses_configured_default_top_k(
    embed_query: Mock,
    search_chunks: Mock,
) -> None:
    client = Mock()
    embed_query.return_value = [0.1, 0.2, 0.3]
    search_chunks.return_value = [_result()]

    results = retrieve_semantic_chunks(
        query="  run containers  ",
        top_k=None,
        document_id=None,
        settings=_settings(),
        elasticsearch_client=client,
    )

    assert results == [_result()]
    embed_query.assert_called_once_with(
        "run containers",
        api_key="test-key",
        model="text-embedding-3-small",
        expected_dimensions=3,
    )
    search_chunks.assert_called_once_with(
        client,
        index_name="contextiq-document-chunks",
        query_embedding=[0.1, 0.2, 0.3],
        top_k=3,
        num_candidates=30,
        document_id=None,
    )


@patch("app.rag.retriever.search_similar_chunks")
@patch("app.rag.retriever.embed_text", return_value=[0.1, 0.2, 0.3])
def test_retriever_preserves_explicit_top_k_and_document_filter(
    embed_query: Mock,
    search_chunks: Mock,
) -> None:
    retrieve_semantic_chunks(
        query="object storage",
        top_k=5,
        document_id=DOCUMENT_ID,
        settings=_settings(),
        elasticsearch_client=Mock(),
    )

    assert search_chunks.call_args.kwargs["top_k"] == 5
    assert search_chunks.call_args.kwargs["num_candidates"] == 50
    assert search_chunks.call_args.kwargs["document_id"] == DOCUMENT_ID


@patch("app.rag.retriever.search_similar_chunks")
@patch("app.rag.retriever.embed_text")
def test_embedding_failure_stops_retrieval(
    embed_query: Mock,
    search_chunks: Mock,
) -> None:
    embed_query.side_effect = EmbeddingAPIError("embedding failed")

    with pytest.raises(EmbeddingAPIError, match="embedding failed"):
        retrieve_semantic_chunks(
            query="containers",
            top_k=3,
            document_id=None,
            settings=_settings(),
            elasticsearch_client=Mock(),
        )

    search_chunks.assert_not_called()


@patch("app.rag.retriever.search_similar_chunks")
@patch("app.rag.retriever.embed_text")
def test_retriever_rejects_empty_query_before_external_calls(
    embed_query: Mock,
    search_chunks: Mock,
) -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        retrieve_semantic_chunks(
            query="   ",
            top_k=3,
            document_id=None,
            settings=_settings(),
            elasticsearch_client=Mock(),
        )

    embed_query.assert_not_called()
    search_chunks.assert_not_called()
