"""Unit tests for RAG question-answer response orchestration."""

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


def _chunk(*, score: float = 0.91) -> SemanticSearchResult:
    text = "AWS Fargate runs containers without managing EC2 virtual machines."
    return SemanticSearchResult(
        document_id=DOCUMENT_ID,
        filename="knowledge.pdf",
        chunk_index=1,
        text=text,
        character_count=len(text),
        score=score,
    )


@patch("app.services.question_service.run_rag_workflow")
def test_question_service_runs_graph_and_preserves_sources(
    run_workflow: Mock,
) -> None:
    chunk = _chunk()
    run_workflow.return_value = {
        "question": "Which service runs containers?",
        "top_k": 2,
        "document_id": DOCUMENT_ID,
        "retrieved_chunks": [chunk],
        "enough_context": True,
        "answer": "Use Amazon ECS with AWS Fargate.",
    }
    client = Mock()
    settings = _settings()

    response = answer_question(
        question="  Which service runs containers?  ",
        top_k=2,
        document_id=DOCUMENT_ID,
        settings=settings,
        elasticsearch_client=client,
    )

    run_workflow.assert_called_once_with(
        question="Which service runs containers?",
        top_k=2,
        document_id=DOCUMENT_ID,
        settings=settings,
        elasticsearch_client=client,
    )
    assert response.answer == "Use Amazon ECS with AWS Fargate."
    assert response.sources[0].document_id == DOCUMENT_ID
    assert response.sources[0].chunk_index == 1
    assert response.sources[0].score == pytest.approx(0.91)
    assert "embedding" not in response.sources[0].model_dump()


@patch("app.services.question_service.run_rag_workflow")
def test_fallback_preserves_weak_retrieval_sources(
    run_workflow: Mock,
) -> None:
    weak_chunk = _chunk(score=0.52)
    run_workflow.return_value = {
        "question": "What is the capital of France?",
        "top_k": 3,
        "document_id": None,
        "retrieved_chunks": [weak_chunk],
        "enough_context": False,
        "answer": INSUFFICIENT_CONTEXT_ANSWER,
    }

    response = answer_question(
        question="What is the capital of France?",
        top_k=3,
        document_id=None,
        settings=_settings(),
        elasticsearch_client=Mock(),
    )

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert [source.chunk_index for source in response.sources] == [1]
    assert response.sources[0].score == pytest.approx(0.52)


@patch("app.services.question_service.run_rag_workflow")
def test_empty_retrieval_returns_no_sources(run_workflow: Mock) -> None:
    run_workflow.return_value = {
        "question": "Question",
        "top_k": None,
        "document_id": None,
        "retrieved_chunks": [],
        "enough_context": False,
        "answer": INSUFFICIENT_CONTEXT_ANSWER,
    }

    response = answer_question(
        question="Question",
        top_k=None,
        document_id=None,
        settings=_settings(),
        elasticsearch_client=Mock(),
    )

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.sources == []


@patch("app.services.question_service.run_rag_workflow")
def test_workflow_failure_is_not_swallowed(run_workflow: Mock) -> None:
    run_workflow.side_effect = GenerationAPIError("generation failed")

    with pytest.raises(GenerationAPIError, match="generation failed"):
        answer_question(
            question="Which service runs containers?",
            top_k=3,
            document_id=None,
            settings=_settings(),
            elasticsearch_client=Mock(),
        )
