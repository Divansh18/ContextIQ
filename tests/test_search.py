"""API tests for semantic search."""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.rag.embeddings import EmbeddingAPIError, MissingOpenAIAPIKeyError
from app.rag.vector_store import VectorSearchError

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


@patch("app.api.routes.search.retrieve_semantic_chunks")
def test_search_returns_ranked_chunks_without_embeddings(
    retrieve_chunks: Mock,
    client: TestClient,
) -> None:
    retrieve_chunks.return_value = [
        {
            "document_id": DOCUMENT_ID,
            "filename": "knowledge.pdf",
            "chunk_index": 2,
            "text": "Amazon S3 provides object storage.",
            "character_count": 34,
            "score": 0.91,
            "embedding": [0.1, 0.2, 0.3],
        }
    ]

    response = client.post(
        "/search",
        json={
            "query": "  where should uploaded files be stored?  ",
            "top_k": 2,
            "document_id": str(DOCUMENT_ID),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "where should uploaded files be stored?",
        "results": [
            {
                "document_id": str(DOCUMENT_ID),
                "filename": "knowledge.pdf",
                "chunk_index": 2,
                "text": "Amazon S3 provides object storage.",
                "character_count": 34,
                "score": 0.91,
            }
        ],
    }
    assert "embedding" not in response.json()["results"][0]
    assert retrieve_chunks.call_args.kwargs["top_k"] == 2
    assert retrieve_chunks.call_args.kwargs["document_id"] == DOCUMENT_ID


@patch("app.api.routes.search.retrieve_semantic_chunks")
def test_search_uses_configured_default_when_top_k_is_omitted(
    retrieve_chunks: Mock,
    client: TestClient,
) -> None:
    retrieve_chunks.return_value = []

    response = client.post("/search", json={"query": "containers"})

    assert response.status_code == 200
    assert retrieve_chunks.call_args.kwargs["top_k"] is None


@patch("app.api.routes.search.retrieve_semantic_chunks")
def test_search_rejects_empty_query(
    retrieve_chunks: Mock,
    client: TestClient,
) -> None:
    response = client.post("/search", json={"query": "   "})

    assert response.status_code == 422
    retrieve_chunks.assert_not_called()


@pytest.mark.parametrize("top_k", [0, 101])
@patch("app.api.routes.search.retrieve_semantic_chunks")
def test_search_rejects_invalid_top_k(
    retrieve_chunks: Mock,
    client: TestClient,
    top_k: int,
) -> None:
    response = client.post(
        "/search",
        json={"query": "containers", "top_k": top_k},
    )

    assert response.status_code == 422
    retrieve_chunks.assert_not_called()


def test_search_rejects_invalid_document_id(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={"query": "containers", "document_id": "not-a-uuid"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (MissingOpenAIAPIKeyError("OPENAI_API_KEY is required."), 503),
        (EmbeddingAPIError("embedding failed"), 502),
        (VectorSearchError("Elasticsearch vector search failed."), 503),
    ],
)
@patch("app.api.routes.search.retrieve_semantic_chunks")
def test_search_reports_external_dependency_failures(
    retrieve_chunks: Mock,
    client: TestClient,
    failure: Exception,
    expected_status: int,
) -> None:
    retrieve_chunks.side_effect = failure

    response = client.post("/search", json={"query": "containers"})

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(failure)}
