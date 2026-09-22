"""API tests for grounded question answering."""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.rag.embeddings import EmbeddingAPIError, MissingOpenAIAPIKeyError
from app.rag.generation import GenerationAPIError, InvalidGenerationResponseError
from app.rag.vector_store import VectorSearchError

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


@patch("app.api.routes.ask.answer_question")
def test_ask_returns_answer_and_sources_without_embeddings(
    answer_with_context: Mock,
    client: TestClient,
) -> None:
    answer_with_context.return_value = {
        "question": "Which service runs containers?",
        "answer": "Amazon ECS with AWS Fargate.",
        "sources": [
            {
                "document_id": DOCUMENT_ID,
                "filename": "knowledge.pdf",
                "chunk_index": 1,
                "score": 0.91,
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
    }

    response = client.post(
        "/ask",
        json={
            "question": "  Which service runs containers?  ",
            "top_k": 2,
            "document_id": str(DOCUMENT_ID),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "Which service runs containers?",
        "answer": "Amazon ECS with AWS Fargate.",
        "sources": [
            {
                "document_id": str(DOCUMENT_ID),
                "filename": "knowledge.pdf",
                "chunk_index": 1,
                "score": 0.91,
            }
        ],
    }
    assert "embedding" not in response.json()["sources"][0]
    assert answer_with_context.call_args.kwargs["top_k"] == 2
    assert answer_with_context.call_args.kwargs["document_id"] == DOCUMENT_ID


@patch("app.api.routes.ask.answer_question")
def test_ask_passes_omitted_top_k_to_configured_default(
    answer_with_context: Mock,
    client: TestClient,
) -> None:
    answer_with_context.return_value = {
        "question": "Question",
        "answer": "Answer",
        "sources": [],
    }

    response = client.post("/ask", json={"question": "Question"})

    assert response.status_code == 200
    assert answer_with_context.call_args.kwargs["top_k"] is None


@patch("app.api.routes.ask.answer_question")
def test_ask_rejects_empty_question(
    answer_with_context: Mock,
    client: TestClient,
) -> None:
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
    answer_with_context.assert_not_called()


@pytest.mark.parametrize("top_k", [0, 101])
@patch("app.api.routes.ask.answer_question")
def test_ask_rejects_invalid_top_k(
    answer_with_context: Mock,
    client: TestClient,
    top_k: int,
) -> None:
    response = client.post(
        "/ask",
        json={"question": "Question", "top_k": top_k},
    )

    assert response.status_code == 422
    answer_with_context.assert_not_called()


def test_ask_rejects_invalid_document_id(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json={"question": "Question", "document_id": "not-a-uuid"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (MissingOpenAIAPIKeyError("OPENAI_API_KEY is required."), 503),
        (EmbeddingAPIError("embedding failed"), 502),
        (GenerationAPIError("generation failed"), 502),
        (InvalidGenerationResponseError("empty answer"), 502),
        (VectorSearchError("Elasticsearch vector search failed."), 503),
    ],
)
@patch("app.api.routes.ask.answer_question")
def test_ask_reports_pipeline_failures(
    answer_with_context: Mock,
    client: TestClient,
    failure: Exception,
    expected_status: int,
) -> None:
    answer_with_context.side_effect = failure

    response = client.post("/ask", json={"question": "Question"})

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(failure)}
