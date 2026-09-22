"""Unit tests for RAG question-answer orchestration."""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.rag.generation import GenerationAPIError, INSUFFICIENT_CONTEXT_ANSWER
from app.schemas.search import SemanticSearchResult
from app.services.question_service import answer_question

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=SecretStr("test-key"),
        embedding_dimensions=3,
        generation_model="gpt-test",
        generation_max_output_tokens=250,
    )


def _chunk() -> SemanticSearchResult:
    text = "AWS Fargate runs containers without managing EC2 virtual machines."
    return SemanticSearchResult(
        document_id=DOCUMENT_ID,
        filename="knowledge.pdf",
        chunk_index=1,
        text=text,
        character_count=len(text),
        score=0.91,
    )


@patch("app.services.question_service.generate_grounded_answer")
@patch("app.services.question_service.retrieve_semantic_chunks")
def test_question_service_retrieves_generates_and_preserves_sources(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    retrieve_chunks.return_value = [_chunk()]
    generate_answer.return_value = "Use Amazon ECS with AWS Fargate."
    client = Mock()

    response = answer_question(
        question="  Which service runs containers?  ",
        top_k=2,
        document_id=DOCUMENT_ID,
        settings=_settings(),
        elasticsearch_client=client,
    )

    retrieve_chunks.assert_called_once_with(
        query="Which service runs containers?",
        top_k=2,
        document_id=DOCUMENT_ID,
        settings=_settings(),
        elasticsearch_client=client,
    )
    generation_call = generate_answer.call_args.kwargs
    assert generation_call["question"] == "Which service runs containers?"
    assert "AWS Fargate runs containers" in generation_call["context"]
    assert generation_call["api_key"] == "test-key"
    assert generation_call["model"] == "gpt-test"
    assert generation_call["max_output_tokens"] == 250
    assert response.answer == "Use Amazon ECS with AWS Fargate."
    assert response.sources[0].document_id == DOCUMENT_ID
    assert response.sources[0].chunk_index == 1
    assert response.sources[0].score == pytest.approx(0.91)
    assert "embedding" not in response.sources[0].model_dump()


@patch("app.services.question_service.generate_grounded_answer")
@patch("app.services.question_service.retrieve_semantic_chunks", return_value=[])
def test_no_retrieval_results_return_grounded_fallback_without_generation(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    response = answer_question(
        question="What is not in the documents?",
        top_k=3,
        document_id=None,
        settings=_settings(),
        elasticsearch_client=Mock(),
    )

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.sources == []
    generate_answer.assert_not_called()


@patch("app.services.question_service.generate_grounded_answer")
@patch("app.services.question_service.retrieve_semantic_chunks")
def test_empty_retrieved_context_returns_grounded_fallback(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    retrieve_chunks.return_value = [
        SemanticSearchResult(
            document_id=DOCUMENT_ID,
            filename="empty.pdf",
            chunk_index=0,
            text=" ",
            character_count=1,
            score=0.2,
        )
    ]

    response = answer_question(
        question="What is not in the documents?",
        top_k=3,
        document_id=None,
        settings=_settings(),
        elasticsearch_client=Mock(),
    )

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.sources == []
    generate_answer.assert_not_called()


@patch("app.services.question_service.generate_grounded_answer")
@patch("app.services.question_service.retrieve_semantic_chunks")
def test_generation_failure_is_not_swallowed(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    retrieve_chunks.return_value = [_chunk()]
    generate_answer.side_effect = GenerationAPIError("generation failed")

    with pytest.raises(GenerationAPIError, match="generation failed"):
        answer_question(
            question="Which service runs containers?",
            top_k=3,
            document_id=None,
            settings=_settings(),
            elasticsearch_client=Mock(),
        )
