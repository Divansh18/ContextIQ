"""Unit tests for LangChain context adaptation and answer generation."""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from openai import OpenAIError

from app.rag.embeddings import MissingOpenAIAPIKeyError
from app.rag.generation import (
    GROUNDING_INSTRUCTIONS,
    GenerationAPIError,
    InvalidGenerationResponseError,
    build_retrieved_context,
    generate_grounded_answer,
    to_langchain_documents,
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


def test_chunks_are_adapted_to_langchain_documents_with_metadata() -> None:
    documents = to_langchain_documents(
        [_chunk(index=1, text="AWS Fargate runs containers.", score=0.91)]
    )

    assert len(documents) == 1
    assert documents[0].page_content == "AWS Fargate runs containers."
    assert documents[0].metadata == {
        "document_id": str(DOCUMENT_ID),
        "filename": "knowledge.pdf",
        "chunk_index": 1,
        "score": 0.91,
    }
    assert "embedding" not in documents[0].metadata


def test_langchain_documents_are_formatted_as_deterministic_context() -> None:
    documents = to_langchain_documents(
        [
            _chunk(index=1, text="AWS Fargate runs containers.", score=0.91),
            _chunk(index=2, text="Amazon S3 stores files.", score=0.82),
        ]
    )

    context = build_retrieved_context(documents)

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


@patch("app.rag.generation.ChatOpenAI")
def test_lcel_pipeline_inserts_prompt_values_and_parses_answer(
    chat_openai: Mock,
) -> None:
    captured: dict[str, object] = {}

    def respond(prompt_value: object) -> AIMessage:
        captured["messages"] = prompt_value.to_messages()  # type: ignore[attr-defined]
        return AIMessage(
            content="AWS Fargate runs containers without managed virtual machines."
        )

    chat_openai.return_value = RunnableLambda(respond)
    context = build_retrieved_context(
        to_langchain_documents(
            [_chunk(index=1, text="AWS Fargate runs containers.", score=0.91)]
        )
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
    chat_openai.assert_called_once()
    model_arguments = chat_openai.call_args.kwargs
    assert model_arguments["model"] == "gpt-test"
    assert model_arguments["api_key"].get_secret_value() == "test-key"
    assert model_arguments["max_completion_tokens"] == 300
    assert model_arguments["use_responses_api"] is True
    assert model_arguments["output_version"] == "responses/v1"

    messages = captured["messages"]
    assert messages[0].content == GROUNDING_INSTRUCTIONS  # type: ignore[index,union-attr]
    human_content = messages[1].content  # type: ignore[index,union-attr]
    assert "QUESTION:\nWhich service runs containers?" in human_content
    assert "BEGIN RETRIEVED CONTEXT" in human_content
    assert "AWS Fargate runs containers." in human_content
    assert "END RETRIEVED CONTEXT" in human_content


@patch("app.rag.generation.ChatOpenAI")
def test_missing_api_key_stops_generation_before_model_creation(
    chat_openai: Mock,
) -> None:
    with pytest.raises(MissingOpenAIAPIKeyError, match="OPENAI_API_KEY"):
        generate_grounded_answer(
            question="Which service runs containers?",
            context="[SOURCE 1]\ncontent:\nFargate",
            api_key=None,
            model="gpt-test",
            max_output_tokens=300,
        )

    chat_openai.assert_not_called()


@patch("app.rag.generation.ChatOpenAI")
def test_generation_provider_failure_is_wrapped(chat_openai: Mock) -> None:
    def fail(_: object) -> str:
        raise OpenAIError("request failed")

    chat_openai.return_value = RunnableLambda(fail)

    with pytest.raises(GenerationAPIError, match="generation failed"):
        generate_grounded_answer(
            question="Which service runs containers?",
            context="[SOURCE 1]\ncontent:\nFargate",
            api_key="test-key",
            model="gpt-test",
            max_output_tokens=300,
        )


@pytest.mark.parametrize("output_text", [None, "", "   "])
@patch("app.rag.generation.build_generation_chain")
def test_empty_or_malformed_generation_result_is_rejected(
    build_chain: Mock,
    output_text: str | None,
) -> None:
    build_chain.return_value.invoke.return_value = output_text

    with pytest.raises(InvalidGenerationResponseError, match="empty or malformed"):
        generate_grounded_answer(
            question="Which service runs containers?",
            context="[SOURCE 1]\ncontent:\nFargate",
            api_key="test-key",
            model="gpt-test",
            max_output_tokens=300,
        )
