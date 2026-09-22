"""Unit tests for grounded context construction and answer generation."""

from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from openai import OpenAIError

from app.rag.embeddings import MissingOpenAIAPIKeyError
from app.rag.generation import (
    GROUNDING_INSTRUCTIONS,
    GenerationAPIError,
    InvalidGenerationResponseError,
    build_generation_input,
    build_retrieved_context,
    generate_grounded_answer,
)
from app.schemas.search import SemanticSearchResult

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


def _chunk(*, index: int, text: str, score: float) -> SemanticSearchResult:
    return SemanticSearchResult(
        document_id=DOCUMENT_ID,
        filename="knowledge.pdf",
        chunk_index=index,
        text=text,
        character_count=len(text),
        score=score,
    )


def test_retrieved_chunks_are_formatted_as_deterministic_context() -> None:
    context = build_retrieved_context(
        [
            _chunk(index=1, text="AWS Fargate runs containers.", score=0.91),
            _chunk(index=2, text="Amazon S3 stores files.", score=0.82),
        ]
    )

    assert context == (
        "[SOURCE 1]\n"
        f"document_id: {DOCUMENT_ID}\n"
        "filename: knowledge.pdf\n"
        "chunk_index: 1\n"
        "retrieval_score: 0.910000\n"
        "content:\n"
        "AWS Fargate runs containers.\n"
        "[/SOURCE 1]\n\n"
        "[SOURCE 2]\n"
        f"document_id: {DOCUMENT_ID}\n"
        "filename: knowledge.pdf\n"
        "chunk_index: 2\n"
        "retrieval_score: 0.820000\n"
        "content:\n"
        "Amazon S3 stores files.\n"
        "[/SOURCE 2]"
    )


@patch("app.rag.generation.OpenAI")
def test_generation_uses_responses_api_with_question_and_context(
    openai_class: Mock,
) -> None:
    client = openai_class.return_value
    client.responses.create.return_value = SimpleNamespace(
        output_text="AWS Fargate runs containers without managed virtual machines."
    )
    context = build_retrieved_context(
        [_chunk(index=1, text="AWS Fargate runs containers.", score=0.91)]
    )

    answer = generate_grounded_answer(
        question="Which service runs containers?",
        context=context,
        api_key="test-key",
        model="gpt-test",
        max_output_tokens=300,
    )

    assert answer == (
        "AWS Fargate runs containers without managed virtual machines."
    )
    client.responses.create.assert_called_once_with(
        model="gpt-test",
        instructions=GROUNDING_INSTRUCTIONS,
        input=build_generation_input(
            question="Which service runs containers?",
            context=context,
        ),
        max_output_tokens=300,
    )
    client.close.assert_called_once_with()


@patch("app.rag.generation.OpenAI")
def test_missing_api_key_stops_generation_before_client_creation(
    openai_class: Mock,
) -> None:
    with pytest.raises(MissingOpenAIAPIKeyError, match="OPENAI_API_KEY"):
        generate_grounded_answer(
            question="Which service runs containers?",
            context="[SOURCE 1]\ncontent:\nFargate",
            api_key=None,
            model="gpt-test",
            max_output_tokens=300,
        )

    openai_class.assert_not_called()


@patch("app.rag.generation.OpenAI")
def test_generation_provider_failure_is_wrapped(openai_class: Mock) -> None:
    client = openai_class.return_value
    client.responses.create.side_effect = OpenAIError("request failed")

    with pytest.raises(GenerationAPIError, match="generation failed"):
        generate_grounded_answer(
            question="Which service runs containers?",
            context="[SOURCE 1]\ncontent:\nFargate",
            api_key="test-key",
            model="gpt-test",
            max_output_tokens=300,
        )

    client.close.assert_called_once_with()


@pytest.mark.parametrize("output_text", [None, "", "   "])
@patch("app.rag.generation.OpenAI")
def test_empty_or_malformed_generation_result_is_rejected(
    openai_class: Mock,
    output_text: str | None,
) -> None:
    openai_class.return_value.responses.create.return_value = SimpleNamespace(
        output_text=output_text
    )

    with pytest.raises(InvalidGenerationResponseError, match="empty or malformed"):
        generate_grounded_answer(
            question="Which service runs containers?",
            context="[SOURCE 1]\ncontent:\nFargate",
            api_key="test-key",
            model="gpt-test",
            max_output_tokens=300,
        )
